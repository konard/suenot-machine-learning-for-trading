//! ResNet model module for time series
//!
//! Implements ResNet architecture adapted for 1D time series data.

mod blocks;
mod layers;
mod resnet;

pub use blocks::{BasicBlock, BottleneckBlock, ResidualBlock};
pub use layers::{BatchNorm1d, Conv1d, Linear, MaxPool1d, ReLU};
pub use resnet::{ResNet18, ResNet34, ResNet50};
