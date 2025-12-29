//! # Market Impact Models
//!
//! Модели воздействия на рынок для оценки стоимости исполнения ордеров.
//!
//! ## Типы воздействия
//!
//! - **Temporary Impact** - временное воздействие, влияет только на текущую сделку
//! - **Permanent Impact** - постоянное воздействие, сдвигает цену для всех будущих сделок
//! - **Transient Impact** - затухающее воздействие, влияет на несколько следующих сделок

mod models;
mod params;

pub use models::{ImpactModel, LinearImpact, SquareRootImpact, TransientImpact, CombinedImpact};
pub use params::ImpactParams;
