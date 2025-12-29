//! Machine learning models module
//!
//! Provides Decision Tree and Random Forest implementations.

mod decision_tree;
mod random_forest;

pub use decision_tree::{DecisionTree, TreeConfig, TreeNode};
pub use random_forest::RandomForest;
