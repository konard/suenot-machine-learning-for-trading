//! Модуль для работы с анонсами Bybit

use crate::models::{Announcement, AnnouncementType};
use chrono::{DateTime, Utc};

/// Фильтр для анонсов
#[derive(Debug, Clone, Default)]
pub struct AnnouncementFilter {
    /// Фильтр по типу анонса
    pub announcement_type: Option<AnnouncementType>,
    /// Фильтр по дате (не раньше)
    pub from_date: Option<DateTime<Utc>>,
    /// Фильтр по дате (не позже)
    pub to_date: Option<DateTime<Utc>>,
    /// Фильтр по символам
    pub symbols: Option<Vec<String>>,
    /// Поиск по ключевым словам
    pub keywords: Option<Vec<String>>,
}

impl AnnouncementFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_type(mut self, ann_type: AnnouncementType) -> Self {
        self.announcement_type = Some(ann_type);
        self
    }

    pub fn from_date(mut self, date: DateTime<Utc>) -> Self {
        self.from_date = Some(date);
        self
    }

    pub fn to_date(mut self, date: DateTime<Utc>) -> Self {
        self.to_date = Some(date);
        self
    }

    pub fn with_symbols(mut self, symbols: Vec<String>) -> Self {
        self.symbols = Some(symbols);
        self
    }

    pub fn with_keywords(mut self, keywords: Vec<String>) -> Self {
        self.keywords = Some(keywords);
        self
    }

    /// Применить фильтр к списку анонсов
    pub fn apply(&self, announcements: &[Announcement]) -> Vec<Announcement> {
        announcements
            .iter()
            .filter(|a| self.matches(a))
            .cloned()
            .collect()
    }

    /// Проверить, соответствует ли анонс фильтру
    fn matches(&self, announcement: &Announcement) -> bool {
        // Проверка типа
        if let Some(ref ann_type) = self.announcement_type {
            if &announcement.announcement_type != ann_type {
                return false;
            }
        }

        // Проверка даты "от"
        if let Some(from) = self.from_date {
            if announcement.publish_time < from {
                return false;
            }
        }

        // Проверка даты "до"
        if let Some(to) = self.to_date {
            if announcement.publish_time > to {
                return false;
            }
        }

        // Проверка символов
        if let Some(ref symbols) = self.symbols {
            let has_symbol = symbols
                .iter()
                .any(|s| announcement.symbols.contains(s));
            if !has_symbol && !symbols.is_empty() {
                return false;
            }
        }

        // Проверка ключевых слов
        if let Some(ref keywords) = self.keywords {
            let text = format!(
                "{} {}",
                announcement.title.to_lowercase(),
                announcement.description.to_lowercase()
            );
            let has_keyword = keywords
                .iter()
                .any(|k| text.contains(&k.to_lowercase()));
            if !has_keyword && !keywords.is_empty() {
                return false;
            }
        }

        true
    }
}

/// Категоризация анонсов по важности для трейдинга
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingRelevance {
    /// Высокая важность (листинг, делистинг)
    High,
    /// Средняя важность (обновления, партнёрства)
    Medium,
    /// Низкая важность (общие новости)
    Low,
}

/// Оценить важность анонса для трейдинга
pub fn assess_trading_relevance(announcement: &Announcement) -> TradingRelevance {
    // Высокий приоритет для листингов и делистингов
    match announcement.announcement_type {
        AnnouncementType::NewListing => return TradingRelevance::High,
        AnnouncementType::Delisting => return TradingRelevance::High,
        _ => {}
    }

    // Проверка ключевых слов
    let text = format!(
        "{} {}",
        announcement.title.to_lowercase(),
        announcement.description.to_lowercase()
    );

    let high_impact_keywords = [
        "listing",
        "delist",
        "airdrop",
        "fork",
        "halving",
        "hack",
        "exploit",
        "partnership",
        "acquisition",
    ];

    let medium_impact_keywords = [
        "update",
        "upgrade",
        "launch",
        "release",
        "integration",
        "support",
        "trading",
    ];

    for keyword in high_impact_keywords {
        if text.contains(keyword) {
            return TradingRelevance::High;
        }
    }

    for keyword in medium_impact_keywords {
        if text.contains(keyword) {
            return TradingRelevance::Medium;
        }
    }

    TradingRelevance::Low
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_announcement(title: &str, ann_type: AnnouncementType) -> Announcement {
        Announcement {
            id: "test".to_string(),
            title: title.to_string(),
            description: String::new(),
            announcement_type: ann_type,
            publish_time: Utc::now(),
            symbols: vec!["BTC".to_string()],
            url: None,
        }
    }

    #[test]
    fn test_filter_by_type() {
        let announcements = vec![
            create_test_announcement("New listing", AnnouncementType::NewListing),
            create_test_announcement("Maintenance", AnnouncementType::Maintenance),
        ];

        let filter = AnnouncementFilter::new().with_type(AnnouncementType::NewListing);
        let filtered = filter.apply(&announcements);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].title, "New listing");
    }

    #[test]
    fn test_trading_relevance() {
        let listing = create_test_announcement("New BTC listing", AnnouncementType::NewListing);
        assert_eq!(assess_trading_relevance(&listing), TradingRelevance::High);

        let maintenance =
            create_test_announcement("System maintenance", AnnouncementType::Maintenance);
        assert_eq!(
            assess_trading_relevance(&maintenance),
            TradingRelevance::Low
        );
    }
}
