use serde::{Deserialize, Serialize};

/// A cell value that may be missing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Numeric(f64),
    String(String),
    /// System-missing (equivalent to SPSS SYSMIS).
    SystemMissing,
}

impl Value {
    pub fn is_missing(&self) -> bool {
        matches!(self, Value::SystemMissing)
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Numeric(v) => Some(*v),
            _ => None,
        }
    }
}

/// User-defined missing value specification per variable.
/// SPSS allows up to 3 discrete values or a range + 1 discrete.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MissingValues {
    /// Discrete missing values (up to 3).
    pub discrete: Vec<f64>,
    /// Range-based missing: (low, high) inclusive.
    pub range: Option<(f64, f64)>,
}

impl MissingValues {
    pub fn is_user_missing(&self, val: f64) -> bool {
        if self.discrete.iter().any(|&v| v == val) {
            return true;
        }
        if let Some((lo, hi)) = self.range {
            if val >= lo && val <= hi {
                return true;
            }
        }
        false
    }
}
