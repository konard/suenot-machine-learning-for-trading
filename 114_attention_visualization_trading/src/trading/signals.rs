//! Trading signal definitions.

/// Trading signal types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// Hold position (no action)
    Hold,
    /// Enter or maintain long position
    Long,
    /// Enter or maintain short position
    Short,
}

impl Signal {
    /// Get the position multiplier (-1, 0, or 1).
    pub fn multiplier(&self) -> f64 {
        match self {
            Signal::Hold => 0.0,
            Signal::Long => 1.0,
            Signal::Short => -1.0,
        }
    }

    /// Check if this is an active position.
    pub fn is_active(&self) -> bool {
        !matches!(self, Signal::Hold)
    }
}

impl Default for Signal {
    fn default() -> Self {
        Signal::Hold
    }
}

impl std::fmt::Display for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Signal::Hold => write!(f, "HOLD"),
            Signal::Long => write!(f, "LONG"),
            Signal::Short => write!(f, "SHORT"),
        }
    }
}
