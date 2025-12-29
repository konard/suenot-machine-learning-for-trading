//! Типы данных для NLP и криптотрейдинга

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Анонс/новость от Bybit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Announcement {
    /// Уникальный идентификатор
    pub id: String,
    /// Заголовок
    pub title: String,
    /// Описание/контент
    pub description: String,
    /// Тип анонса
    pub announcement_type: AnnouncementType,
    /// Время публикации
    pub publish_time: DateTime<Utc>,
    /// Связанные символы (если есть)
    pub symbols: Vec<String>,
    /// URL для полного текста
    pub url: Option<String>,
}

/// Тип анонса
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AnnouncementType {
    /// Листинг новой монеты
    NewListing,
    /// Делистинг
    Delisting,
    /// Обновление продукта
    ProductUpdate,
    /// Техническое обслуживание
    Maintenance,
    /// Акции и промо
    Promotion,
    /// Другое
    Other,
}

impl Default for AnnouncementType {
    fn default() -> Self {
        Self::Other
    }
}

/// Рыночные данные (свеча)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Время открытия свечи
    pub open_time: DateTime<Utc>,
    /// Цена открытия
    pub open: f64,
    /// Максимальная цена
    pub high: f64,
    /// Минимальная цена
    pub low: f64,
    /// Цена закрытия
    pub close: f64,
    /// Объём
    pub volume: f64,
    /// Объём в USDT
    pub turnover: f64,
}

/// Результат анализа настроений
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentResult {
    /// Оригинальный текст
    pub text: String,
    /// Полярность настроения
    pub polarity: Polarity,
    /// Оценка от -1.0 до 1.0
    pub score: f64,
    /// Уверенность в оценке (0.0 - 1.0)
    pub confidence: f64,
    /// Ключевые слова, повлиявшие на оценку
    pub key_words: Vec<ScoredWord>,
}

/// Слово с оценкой
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredWord {
    pub word: String,
    pub score: f64,
}

/// Полярность настроения
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Polarity {
    /// Позитивное настроение
    Positive,
    /// Нейтральное настроение
    Neutral,
    /// Негативное настроение
    Negative,
}

impl Polarity {
    pub fn from_score(score: f64) -> Self {
        if score > 0.1 {
            Polarity::Positive
        } else if score < -0.1 {
            Polarity::Negative
        } else {
            Polarity::Neutral
        }
    }
}

/// Торговый сигнал
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Символ торговой пары
    pub symbol: String,
    /// Время генерации сигнала
    pub timestamp: DateTime<Utc>,
    /// Рекомендуемое действие
    pub action: SignalAction,
    /// Сила сигнала (0.0 - 1.0)
    pub strength: f64,
    /// Уверенность в сигнале (0.0 - 1.0)
    pub confidence: f64,
    /// Агрегированное настроение
    pub sentiment_score: f64,
    /// Количество проанализированных текстов
    pub texts_analyzed: usize,
    /// Причины сигнала
    pub reasons: Vec<String>,
}

/// Действие торгового сигнала
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalAction {
    /// Сильный сигнал на покупку
    StrongBuy,
    /// Сигнал на покупку
    Buy,
    /// Держать позицию
    Hold,
    /// Сигнал на продажу
    Sell,
    /// Сильный сигнал на продажу
    StrongSell,
}

impl SignalAction {
    pub fn from_score(score: f64, confidence: f64) -> Self {
        let threshold = 0.7; // минимальная уверенность для сильных сигналов

        if score > 0.5 && confidence >= threshold {
            SignalAction::StrongBuy
        } else if score > 0.2 {
            SignalAction::Buy
        } else if score < -0.5 && confidence >= threshold {
            SignalAction::StrongSell
        } else if score < -0.2 {
            SignalAction::Sell
        } else {
            SignalAction::Hold
        }
    }
}

/// Токен (результат токенизации)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Token {
    /// Оригинальная форма
    pub original: String,
    /// Нормализованная форма (lowercase, stemmed)
    pub normalized: String,
    /// Позиция в тексте
    pub position: usize,
    /// Тип токена
    pub token_type: TokenType,
}

/// Тип токена
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenType {
    /// Слово
    Word,
    /// Число
    Number,
    /// Символ криптовалюты (BTC, ETH, etc.)
    CryptoSymbol,
    /// Хэштег
    Hashtag,
    /// Упоминание (@user)
    Mention,
    /// URL
    Url,
    /// Знак препинания
    Punctuation,
    /// Эмодзи
    Emoji,
    /// Другое
    Other,
}

/// Матрица документ-терм
#[derive(Debug, Clone)]
pub struct DocumentTermMatrix {
    /// Названия документов
    pub documents: Vec<String>,
    /// Словарь терминов (слово -> индекс)
    pub vocabulary: std::collections::HashMap<String, usize>,
    /// Обратный словарь (индекс -> слово)
    pub terms: Vec<String>,
    /// Матрица частот [документы x термины]
    pub matrix: Vec<Vec<f64>>,
}

impl DocumentTermMatrix {
    pub fn new() -> Self {
        Self {
            documents: Vec::new(),
            vocabulary: std::collections::HashMap::new(),
            terms: Vec::new(),
            matrix: Vec::new(),
        }
    }

    /// Количество документов
    pub fn n_documents(&self) -> usize {
        self.documents.len()
    }

    /// Количество терминов
    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }

    /// Получить вектор документа
    pub fn get_document_vector(&self, doc_idx: usize) -> Option<&Vec<f64>> {
        self.matrix.get(doc_idx)
    }
}

impl Default for DocumentTermMatrix {
    fn default() -> Self {
        Self::new()
    }
}

/// Конфигурация API Bybit
#[derive(Debug, Clone)]
pub struct BybitConfig {
    /// Базовый URL API
    pub base_url: String,
    /// API ключ (опционально для публичных эндпоинтов)
    pub api_key: Option<String>,
    /// API секрет
    pub api_secret: Option<String>,
    /// Таймаут запроса в секундах
    pub timeout_secs: u64,
}

impl Default for BybitConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
            api_key: None,
            api_secret: None,
            timeout_secs: 30,
        }
    }
}
