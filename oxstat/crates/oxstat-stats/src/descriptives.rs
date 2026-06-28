//! DESCRIPTIVES procedure.
//!
//! Computes mean, std deviation, min, max, skewness, kurtosis.

use oxstat_core::Dataset;
use oxstat_output::OutputItem;

/// Options for the DESCRIPTIVES command.
pub struct DescriptivesOptions {
    pub variables: Vec<String>,
    pub statistics: Vec<Statistic>,
}

/// Available statistics for DESCRIPTIVES.
#[derive(Debug, Clone, Copy)]
pub enum Statistic {
    Mean,
    StdDev,
    Min,
    Max,
    Variance,
    Skewness,
    Kurtosis,
    Sum,
    Range,
}

impl Default for DescriptivesOptions {
    fn default() -> Self {
        Self {
            variables: Vec::new(),
            statistics: vec![
                Statistic::Mean,
                Statistic::StdDev,
                Statistic::Min,
                Statistic::Max,
            ],
        }
    }
}

/// Run DESCRIPTIVES on a dataset.
pub fn run(_dataset: &Dataset, _options: &DescriptivesOptions) -> Vec<OutputItem> {
    // TODO: Compute statistics for each variable
    // TODO: Build output table
    vec![]
}
