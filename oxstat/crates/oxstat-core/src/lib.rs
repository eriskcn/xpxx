//! oxstat-core: Core data structures for OxStat.
//!
//! Provides the in-memory data frame backed by Apache Arrow,
//! missing value system, and variable metadata.

mod dataset;
mod missing;
mod variable;

pub use dataset::Dataset;
pub use missing::{MissingValues, Value};
pub use variable::{MeasureLevel, Variable, VariableType};
