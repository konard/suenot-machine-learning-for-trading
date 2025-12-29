//! Модели market impact

use super::params::ImpactParams;

/// Трейт для моделей market impact
pub trait ImpactModel: Send + Sync {
    /// Рассчитать временное воздействие (temporary impact)
    fn temporary_impact(&self, quantity: f64, params: &ImpactParams) -> f64;

    /// Рассчитать постоянное воздействие (permanent impact)
    fn permanent_impact(&self, quantity: f64, params: &ImpactParams) -> f64;

    /// Рассчитать полное воздействие
    fn total_impact(&self, quantity: f64, params: &ImpactParams) -> f64 {
        self.temporary_impact(quantity, params) + self.permanent_impact(quantity, params)
    }

    /// Рассчитать стоимость исполнения
    fn execution_cost(
        &self,
        quantity: f64,
        arrival_price: f64,
        params: &ImpactParams,
    ) -> f64 {
        let impact = self.total_impact(quantity, params);
        quantity * impact * arrival_price
    }

    /// Название модели
    fn name(&self) -> &'static str;
}

/// Линейная модель impact
///
/// temporary_impact = eta * (quantity / ADV)
/// permanent_impact = gamma * (quantity / ADV)
#[derive(Debug, Clone, Default)]
pub struct LinearImpact;

impl ImpactModel for LinearImpact {
    fn temporary_impact(&self, quantity: f64, params: &ImpactParams) -> f64 {
        params.eta * quantity.abs() / params.adv
    }

    fn permanent_impact(&self, quantity: f64, params: &ImpactParams) -> f64 {
        params.gamma * quantity / params.adv
    }

    fn name(&self) -> &'static str {
        "Linear"
    }
}

/// Модель квадратного корня (Square Root Impact)
///
/// Эмпирически обоснованная модель, наблюдаемая на реальных рынках.
/// temporary_impact = eta * sign(q) * sqrt(|q| / ADV)
/// permanent_impact = gamma * sign(q) * sqrt(|q| / ADV)
#[derive(Debug, Clone, Default)]
pub struct SquareRootImpact;

impl ImpactModel for SquareRootImpact {
    fn temporary_impact(&self, quantity: f64, params: &ImpactParams) -> f64 {
        let normalized = (quantity.abs() / params.adv).sqrt();
        params.eta * normalized
    }

    fn permanent_impact(&self, quantity: f64, params: &ImpactParams) -> f64 {
        let sign = if quantity >= 0.0 { 1.0 } else { -1.0 };
        let normalized = (quantity.abs() / params.adv).sqrt();
        params.gamma * sign * normalized
    }

    fn name(&self) -> &'static str {
        "SquareRoot"
    }
}

/// Затухающее воздействие (Transient Impact)
///
/// Воздействие, которое затухает со временем.
/// Используется для моделирования временной ликвидности.
#[derive(Debug, Clone)]
pub struct TransientImpact {
    /// История предыдущих сделок и их оставшегося воздействия
    history: Vec<(f64, f64)>, // (remaining_impact, decay_factor)
}

impl Default for TransientImpact {
    fn default() -> Self {
        Self::new()
    }
}

impl TransientImpact {
    /// Создать новую модель
    pub fn new() -> Self {
        Self { history: Vec::new() }
    }

    /// Добавить сделку в историю
    pub fn add_trade(&mut self, quantity: f64, params: &ImpactParams) {
        let impact = params.eta * (quantity.abs() / params.adv).sqrt();
        self.history.push((impact, params.decay_rate));
    }

    /// Обновить историю (применить затухание)
    pub fn decay(&mut self) {
        self.history.retain_mut(|(impact, decay)| {
            *impact *= *decay;
            *impact > 0.0001 // Удаляем слишком маленькие воздействия
        });
    }

    /// Получить накопленное transient impact
    pub fn accumulated_impact(&self) -> f64 {
        self.history.iter().map(|(impact, _)| *impact).sum()
    }

    /// Сбросить историю
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

impl ImpactModel for TransientImpact {
    fn temporary_impact(&self, quantity: f64, params: &ImpactParams) -> f64 {
        let direct = params.eta * (quantity.abs() / params.adv).sqrt();
        direct + self.accumulated_impact()
    }

    fn permanent_impact(&self, quantity: f64, params: &ImpactParams) -> f64 {
        let sign = if quantity >= 0.0 { 1.0 } else { -1.0 };
        params.gamma * sign * (quantity.abs() / params.adv).sqrt()
    }

    fn name(&self) -> &'static str {
        "Transient"
    }
}

/// Комбинированная модель impact
///
/// Позволяет комбинировать различные модели с весами.
#[derive(Debug, Clone)]
pub struct CombinedImpact {
    /// Базовая модель для temporary impact
    temp_model: Box<dyn ImpactModel>,
    /// Базовая модель для permanent impact
    perm_model: Box<dyn ImpactModel>,
    /// Дополнительные компоненты (спред, комиссии)
    spread_cost: f64,
    commission: f64,
}

impl CombinedImpact {
    /// Создать комбинированную модель
    pub fn new(spread_cost: f64, commission: f64) -> Self {
        Self {
            temp_model: Box::new(SquareRootImpact),
            perm_model: Box::new(SquareRootImpact),
            spread_cost,
            commission,
        }
    }

    /// Установить модель временного воздействия
    pub fn with_temp_model<M: ImpactModel + 'static>(mut self, model: M) -> Self {
        self.temp_model = Box::new(model);
        self
    }

    /// Установить модель постоянного воздействия
    pub fn with_perm_model<M: ImpactModel + 'static>(mut self, model: M) -> Self {
        self.perm_model = Box::new(model);
        self
    }
}

impl ImpactModel for CombinedImpact {
    fn temporary_impact(&self, quantity: f64, params: &ImpactParams) -> f64 {
        self.temp_model.temporary_impact(quantity, params)
            + self.spread_cost / 2.0 // Половина спреда
            + self.commission
    }

    fn permanent_impact(&self, quantity: f64, params: &ImpactParams) -> f64 {
        self.perm_model.permanent_impact(quantity, params)
    }

    fn name(&self) -> &'static str {
        "Combined"
    }
}

// Для Box<dyn ImpactModel>
impl<T: ImpactModel + ?Sized> ImpactModel for Box<T> {
    fn temporary_impact(&self, quantity: f64, params: &ImpactParams) -> f64 {
        (**self).temporary_impact(quantity, params)
    }

    fn permanent_impact(&self, quantity: f64, params: &ImpactParams) -> f64 {
        (**self).permanent_impact(quantity, params)
    }

    fn name(&self) -> &'static str {
        (**self).name()
    }
}

impl Clone for Box<dyn ImpactModel> {
    fn clone(&self) -> Self {
        // Создаём новый SquareRootImpact как fallback
        Box::new(SquareRootImpact)
    }
}

impl std::fmt::Debug for Box<dyn ImpactModel> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ImpactModel({})", self.name())
    }
}

/// Рассчитать полную стоимость исполнения траектории
pub fn trajectory_cost<M: ImpactModel>(
    model: &M,
    schedule: &[f64],
    arrival_price: f64,
    params: &ImpactParams,
) -> f64 {
    let mut total_cost = 0.0;
    let mut price = arrival_price;

    for &quantity in schedule {
        let temp = model.temporary_impact(quantity, params);
        let perm = model.permanent_impact(quantity, params);

        // Цена исполнения = текущая цена + временное + половина постоянного
        let execution_price = price * (1.0 + temp + perm / 2.0);

        // Стоимость = объём * (цена исполнения - arrival price)
        total_cost += quantity * (execution_price - arrival_price);

        // Обновляем цену для следующей сделки
        price *= 1.0 + perm;
    }

    total_cost
}

/// Рассчитать implementation shortfall
pub fn implementation_shortfall<M: ImpactModel>(
    model: &M,
    schedule: &[f64],
    arrival_price: f64,
    params: &ImpactParams,
) -> f64 {
    let total_quantity: f64 = schedule.iter().sum();
    if total_quantity == 0.0 {
        return 0.0;
    }

    let cost = trajectory_cost(model, schedule, arrival_price, params);
    cost / (total_quantity * arrival_price) // Как доля от общей стоимости
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_impact() {
        let model = LinearImpact;
        let params = ImpactParams {
            adv: 1_000_000.0,
            eta: 0.0001,
            gamma: 0.0005,
            volatility: 0.02,
            spread: 0.0001,
            decay_rate: 0.5,
        };

        let temp = model.temporary_impact(10_000.0, &params);
        let perm = model.permanent_impact(10_000.0, &params);

        assert!(temp > 0.0);
        assert!(perm > 0.0);
        assert!(temp < perm); // eta < gamma
    }

    #[test]
    fn test_sqrt_impact() {
        let model = SquareRootImpact;
        let params = ImpactParams::crypto_default();

        // Проверяем субаддитивность (impact растёт медленнее чем линейно)
        let impact_100 = model.temporary_impact(100.0, &params);
        let impact_400 = model.temporary_impact(400.0, &params);

        // sqrt(400) = 2 * sqrt(100), поэтому impact_400 = 2 * impact_100
        assert!((impact_400 / impact_100 - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_transient_impact() {
        let mut model = TransientImpact::new();
        let params = ImpactParams::crypto_default();

        // Добавляем сделку
        model.add_trade(1000.0, &params);
        let impact1 = model.accumulated_impact();

        // После затухания impact должен уменьшиться
        model.decay();
        let impact2 = model.accumulated_impact();

        assert!(impact2 < impact1);
    }

    #[test]
    fn test_trajectory_cost() {
        let model = SquareRootImpact;
        let params = ImpactParams::crypto_default();

        // Равномерное исполнение
        let schedule = vec![100.0, 100.0, 100.0, 100.0, 100.0];
        let cost = trajectory_cost(&model, &schedule, 50000.0, &params);

        assert!(cost > 0.0);
    }
}
