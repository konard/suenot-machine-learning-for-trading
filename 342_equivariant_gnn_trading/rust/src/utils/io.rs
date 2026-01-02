//! IO Utilities

use anyhow::Result;
use serde::{de::DeserializeOwned, Serialize};
use std::{fs, path::Path};

/// Save data as JSON
pub fn save_json<T: Serialize>(data: &T, path: impl AsRef<Path>) -> Result<()> {
    let json = serde_json::to_string_pretty(data)?;
    fs::write(path, json)?;
    Ok(())
}

/// Load data from JSON
pub fn load_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let json = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&json)?)
}
