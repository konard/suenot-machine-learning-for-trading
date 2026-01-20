//! Execution engine and order management.
//!
//! This module provides:
//! - Parent and child order structures
//! - Execution engine for managing order flow
//! - LLM adapter for intelligent execution decisions
//! - Execution state machine

mod engine;
mod llm_adapter;
mod state_machine;
mod types;

pub use engine::{ExecutionConfig, ExecutionEngine, ExecutionError, ExecutionResult};
pub use llm_adapter::{LlmAdapter, LlmConfig, LlmDecision, LlmError};
pub use state_machine::{ExecutionState, StateTransition};
pub use types::{
    ChildOrder, ChildOrderStatus, OrderId, ParentOrder, ParentOrderStatus, Side,
};
