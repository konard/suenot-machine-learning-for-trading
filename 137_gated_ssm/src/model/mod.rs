//! Model module: SSM and Gated SSM implementations.

pub mod gated_ssm;
pub mod ssm;

pub use gated_ssm::GatedSSMBlock;
pub use ssm::DiagonalSSMState;
