//! Continual learning components.
//!
//! This module provides the core components for continual meta-learning:
//! - Memory buffer for experience replay
//! - Elastic Weight Consolidation (EWC) for preventing forgetting
//! - CML trainer for the main algorithm

pub mod memory;
pub mod ewc;
pub mod learner;
