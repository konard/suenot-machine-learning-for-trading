//! Difference-in-Differences estimation models.

mod did;

pub use did::{generate_synthetic_panel, DiDEstimator, DiDResults, PanelData, TreatmentEffect};
