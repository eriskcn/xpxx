use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::missing::MissingValues;

/// Variable data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VariableType {
    Numeric,
    /// String with max width in bytes.
    String(u32),
}

/// SPSS measurement level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasureLevel {
    Scale,
    Ordinal,
    Nominal,
}

/// A variable (column) definition with full SPSS-style metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variable {
    /// Variable name (up to 64 chars, no spaces).
    pub name: String,
    pub var_type: VariableType,
    pub measure: MeasureLevel,
    /// Descriptive label.
    pub label: Option<String>,
    /// Value labels: numeric value → label string.
    pub value_labels: HashMap<i64, String>,
    /// User-defined missing values.
    pub missing: MissingValues,
    /// Display width.
    pub print_width: u8,
    /// Decimal places for display.
    pub print_decimals: u8,
}

impl Variable {
    /// Create a numeric variable with defaults.
    pub fn numeric(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            var_type: VariableType::Numeric,
            measure: MeasureLevel::Scale,
            label: None,
            value_labels: HashMap::new(),
            missing: MissingValues::default(),
            print_width: 8,
            print_decimals: 2,
        }
    }

    /// Create a string variable with given width.
    pub fn string(name: impl Into<String>, width: u32) -> Self {
        Self {
            name: name.into(),
            var_type: VariableType::String(width),
            measure: MeasureLevel::Nominal,
            label: None,
            value_labels: HashMap::new(),
            missing: MissingValues::default(),
            print_width: width as u8,
            print_decimals: 0,
        }
    }
}
