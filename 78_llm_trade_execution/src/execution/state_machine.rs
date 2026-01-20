//! Execution state machine for managing order lifecycle.

use crate::execution::{ChildOrderStatus, ParentOrderStatus};
use serde::{Deserialize, Serialize};
use std::fmt;

/// High-level execution state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionState {
    /// Initial state - order received
    Initialized,
    /// Validating order parameters
    Validating,
    /// Calculating execution schedule
    Scheduling,
    /// Actively executing (generating child orders)
    Executing,
    /// Waiting for child orders to fill
    WaitingForFills,
    /// Paused by user or system
    Paused,
    /// Adjusting strategy based on market conditions
    Adjusting,
    /// Completing execution (final reconciliation)
    Completing,
    /// Successfully completed
    Completed,
    /// Cancelled by user
    Cancelled,
    /// Failed due to error
    Failed,
}

impl ExecutionState {
    /// Check if this is a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            ExecutionState::Completed | ExecutionState::Cancelled | ExecutionState::Failed
        )
    }

    /// Check if this state can process new slices
    pub fn can_execute(&self) -> bool {
        matches!(
            self,
            ExecutionState::Executing | ExecutionState::Adjusting
        )
    }

    /// Check if this state is waiting for fills
    pub fn is_waiting(&self) -> bool {
        matches!(self, ExecutionState::WaitingForFills)
    }

    /// Get valid transitions from this state
    pub fn valid_transitions(&self) -> Vec<ExecutionState> {
        match self {
            ExecutionState::Initialized => vec![
                ExecutionState::Validating,
                ExecutionState::Cancelled,
                ExecutionState::Failed,
            ],
            ExecutionState::Validating => vec![
                ExecutionState::Scheduling,
                ExecutionState::Cancelled,
                ExecutionState::Failed,
            ],
            ExecutionState::Scheduling => vec![
                ExecutionState::Executing,
                ExecutionState::Cancelled,
                ExecutionState::Failed,
            ],
            ExecutionState::Executing => vec![
                ExecutionState::WaitingForFills,
                ExecutionState::Adjusting,
                ExecutionState::Paused,
                ExecutionState::Completing,
                ExecutionState::Cancelled,
                ExecutionState::Failed,
            ],
            ExecutionState::WaitingForFills => vec![
                ExecutionState::Executing,
                ExecutionState::Adjusting,
                ExecutionState::Paused,
                ExecutionState::Completing,
                ExecutionState::Cancelled,
                ExecutionState::Failed,
            ],
            ExecutionState::Paused => vec![
                ExecutionState::Executing,
                ExecutionState::Cancelled,
                ExecutionState::Failed,
            ],
            ExecutionState::Adjusting => vec![
                ExecutionState::Executing,
                ExecutionState::WaitingForFills,
                ExecutionState::Paused,
                ExecutionState::Cancelled,
                ExecutionState::Failed,
            ],
            ExecutionState::Completing => vec![
                ExecutionState::Completed,
                ExecutionState::Failed,
            ],
            ExecutionState::Completed => vec![],
            ExecutionState::Cancelled => vec![],
            ExecutionState::Failed => vec![],
        }
    }

    /// Check if a transition to the target state is valid
    pub fn can_transition_to(&self, target: ExecutionState) -> bool {
        self.valid_transitions().contains(&target)
    }
}

impl fmt::Display for ExecutionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutionState::Initialized => write!(f, "Initialized"),
            ExecutionState::Validating => write!(f, "Validating"),
            ExecutionState::Scheduling => write!(f, "Scheduling"),
            ExecutionState::Executing => write!(f, "Executing"),
            ExecutionState::WaitingForFills => write!(f, "WaitingForFills"),
            ExecutionState::Paused => write!(f, "Paused"),
            ExecutionState::Adjusting => write!(f, "Adjusting"),
            ExecutionState::Completing => write!(f, "Completing"),
            ExecutionState::Completed => write!(f, "Completed"),
            ExecutionState::Cancelled => write!(f, "Cancelled"),
            ExecutionState::Failed => write!(f, "Failed"),
        }
    }
}

impl From<ParentOrderStatus> for ExecutionState {
    fn from(status: ParentOrderStatus) -> Self {
        match status {
            ParentOrderStatus::Pending => ExecutionState::Initialized,
            ParentOrderStatus::Active => ExecutionState::Executing,
            ParentOrderStatus::Paused => ExecutionState::Paused,
            ParentOrderStatus::Completed => ExecutionState::Completed,
            ParentOrderStatus::Cancelled => ExecutionState::Cancelled,
            ParentOrderStatus::Failed => ExecutionState::Failed,
        }
    }
}

/// State transition event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// Previous state
    pub from: ExecutionState,
    /// New state
    pub to: ExecutionState,
    /// Reason for transition
    pub reason: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl StateTransition {
    /// Create a new state transition
    pub fn new(from: ExecutionState, to: ExecutionState, reason: impl Into<String>) -> Self {
        Self {
            from,
            to,
            reason: reason.into(),
            timestamp: chrono::Utc::now(),
        }
    }
}

/// State machine for execution management
#[derive(Debug, Clone)]
pub struct ExecutionStateMachine {
    current_state: ExecutionState,
    history: Vec<StateTransition>,
}

impl ExecutionStateMachine {
    /// Create a new state machine
    pub fn new() -> Self {
        Self {
            current_state: ExecutionState::Initialized,
            history: Vec::new(),
        }
    }

    /// Get the current state
    pub fn current_state(&self) -> ExecutionState {
        self.current_state
    }

    /// Get the state history
    pub fn history(&self) -> &[StateTransition] {
        &self.history
    }

    /// Attempt to transition to a new state
    pub fn transition(&mut self, to: ExecutionState, reason: impl Into<String>) -> Result<(), String> {
        if !self.current_state.can_transition_to(to) {
            return Err(format!(
                "Invalid transition from {} to {}",
                self.current_state, to
            ));
        }

        let transition = StateTransition::new(self.current_state, to, reason);
        self.history.push(transition);
        self.current_state = to;
        Ok(())
    }

    /// Force a state (bypasses validation - use with caution)
    pub fn force_state(&mut self, state: ExecutionState, reason: impl Into<String>) {
        let transition = StateTransition::new(self.current_state, state, reason);
        self.history.push(transition);
        self.current_state = state;
    }

    /// Check if the execution is in a terminal state
    pub fn is_terminal(&self) -> bool {
        self.current_state.is_terminal()
    }

    /// Check if currently executing
    pub fn is_executing(&self) -> bool {
        self.current_state.can_execute()
    }

    /// Get the time spent in the current state
    pub fn time_in_current_state(&self) -> chrono::Duration {
        if let Some(last) = self.history.last() {
            chrono::Utc::now() - last.timestamp
        } else {
            chrono::Duration::zero()
        }
    }
}

impl Default for ExecutionStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

/// Map child order status to execution state impact
pub fn child_order_state_impact(status: ChildOrderStatus) -> Option<ExecutionState> {
    match status {
        ChildOrderStatus::Rejected => Some(ExecutionState::Adjusting),
        ChildOrderStatus::Filled => None, // Continue executing
        ChildOrderStatus::Cancelled => None, // May need to reschedule
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_machine_transitions() {
        let mut sm = ExecutionStateMachine::new();
        assert_eq!(sm.current_state(), ExecutionState::Initialized);

        // Valid transitions
        assert!(sm.transition(ExecutionState::Validating, "Starting validation").is_ok());
        assert!(sm.transition(ExecutionState::Scheduling, "Validation passed").is_ok());
        assert!(sm.transition(ExecutionState::Executing, "Schedule ready").is_ok());

        assert_eq!(sm.current_state(), ExecutionState::Executing);
        assert_eq!(sm.history().len(), 3);
    }

    #[test]
    fn test_invalid_transition() {
        let mut sm = ExecutionStateMachine::new();

        // Cannot go directly from Initialized to Executing
        let result = sm.transition(ExecutionState::Executing, "Skip validation");
        assert!(result.is_err());
    }

    #[test]
    fn test_terminal_states() {
        assert!(ExecutionState::Completed.is_terminal());
        assert!(ExecutionState::Cancelled.is_terminal());
        assert!(ExecutionState::Failed.is_terminal());
        assert!(!ExecutionState::Executing.is_terminal());
    }

    #[test]
    fn test_can_execute() {
        assert!(ExecutionState::Executing.can_execute());
        assert!(ExecutionState::Adjusting.can_execute());
        assert!(!ExecutionState::Paused.can_execute());
        assert!(!ExecutionState::Completed.can_execute());
    }
}
