//! DESCRIPTIVES procedure.
//!
//! Computes mean, std deviation, min, max, skewness, kurtosis.

use oxstat_core::{Dataset, VariableType};
use oxstat_output::{OutputItem, Table};
use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array};

/// Options for the DESCRIPTIVES command.
pub struct DescriptivesOptions {
    pub variables: Vec<String>,
    pub statistics: Vec<Statistic>,
}

/// Available statistics for DESCRIPTIVES.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Extract numeric values as optional f64 from any numeric array type.
fn extract_f64_values(column: &dyn Array) -> Vec<Option<f64>> {
    let mut values = Vec::with_capacity(column.len());
    if let Some(arr) = column.as_any().downcast_ref::<Float64Array>() {
        for i in 0..arr.len() {
            if arr.is_valid(i) {
                values.push(Some(arr.value(i)));
            } else {
                values.push(None);
            }
        }
    } else if let Some(arr) = column.as_any().downcast_ref::<Float32Array>() {
        for i in 0..arr.len() {
            if arr.is_valid(i) {
                values.push(Some(arr.value(i) as f64));
            } else {
                values.push(None);
            }
        }
    } else if let Some(arr) = column.as_any().downcast_ref::<Int64Array>() {
        for i in 0..arr.len() {
            if arr.is_valid(i) {
                values.push(Some(arr.value(i) as f64));
            } else {
                values.push(None);
            }
        }
    } else if let Some(arr) = column.as_any().downcast_ref::<Int32Array>() {
        for i in 0..arr.len() {
            if arr.is_valid(i) {
                values.push(Some(arr.value(i) as f64));
            } else {
                values.push(None);
            }
        }
    } else {
        // Fallback cast using arrow compute
        if let Ok(cast_arr) = arrow::compute::cast(column, &arrow::datatypes::DataType::Float64) {
            if let Some(arr) = cast_arr.as_any().downcast_ref::<Float64Array>() {
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        values.push(Some(arr.value(i)));
                    } else {
                        values.push(None);
                    }
                }
            }
        }
    }
    values
}

/// Format values as string, displaying "." for NaNs.
fn format_val(val: f64, decimals: u8) -> String {
    if val.is_nan() {
        ".".to_string()
    } else {
        format!("{:.1$}", val, decimals as usize)
    }
}

/// Run DESCRIPTIVES on a dataset.
pub fn run(dataset: &Dataset, options: &DescriptivesOptions) -> Vec<OutputItem> {
    // 1. Gather variables to run on (default to all numeric)
    let variables_to_run = if options.variables.is_empty() {
        dataset.variables.iter()
            .filter(|v| matches!(v.var_type, VariableType::Numeric))
            .map(|v| v.name.clone())
            .collect::<Vec<String>>()
    } else {
        options.variables.clone()
    };

    // Pre-retrieve column data for listwise N calculation
    let mut col_data = Vec::new();
    for var_name in &variables_to_run {
        if let Some(col_idx) = dataset.variable_index(var_name) {
            if let Some(var) = dataset.variable(var_name) {
                if matches!(var.var_type, VariableType::Numeric) {
                    let mut values = Vec::new();
                    for batch in &dataset.batches {
                        let column = batch.column(col_idx);
                        values.extend(extract_f64_values(column.as_ref()));
                    }
                    col_data.push((var.clone(), values));
                }
            }
        }
    }

    // Compute listwise N
    let n_cases = dataset.n_cases();
    let mut listwise_n = 0;
    if !col_data.is_empty() {
        for r in 0..n_cases {
            let mut row_valid = true;
            for (var, values) in &col_data {
                if r < values.len() {
                    match values[r] {
                        Some(val) => {
                            if var.missing.is_user_missing(val) {
                                row_valid = false;
                                break;
                            }
                        }
                        None => {
                            row_valid = false;
                            break;
                        }
                    }
                } else {
                    row_valid = false;
                    break;
                }
            }
            if row_valid {
                listwise_n += 1;
            }
        }
    }

    // Build headers
    let mut column_headers = vec!["Variable".to_string(), "N".to_string()];
    for stat in &options.statistics {
        let name = match stat {
            Statistic::Mean => "Mean",
            Statistic::StdDev => "Std. Deviation",
            Statistic::Min => "Minimum",
            Statistic::Max => "Maximum",
            Statistic::Variance => "Variance",
            Statistic::Skewness => "Skewness",
            Statistic::Kurtosis => "Kurtosis",
            Statistic::Sum => "Sum",
            Statistic::Range => "Range",
        };
        column_headers.push(name.to_string());
    }

    let mut rows = Vec::new();

    // Compute stats for each variable
    for var_name in &variables_to_run {
        let col_idx = match dataset.variable_index(var_name) {
            Some(idx) => idx,
            None => continue,
        };
        let var = match dataset.variable(var_name) {
            Some(v) => v,
            None => continue,
        };
        if !matches!(var.var_type, VariableType::Numeric) {
            continue;
        }

        // Get non-missing values
        let mut valid_values = Vec::new();
        for batch in &dataset.batches {
            let column = batch.column(col_idx);
            let raw_values = extract_f64_values(column.as_ref());
            for val_opt in raw_values {
                if let Some(val) = val_opt {
                    if !var.missing.is_user_missing(val) {
                        valid_values.push(val);
                    }
                }
            }
        }

        let n = valid_values.len();
        let sum: f64 = valid_values.iter().sum();
        let mean = if n > 0 { sum / (n as f64) } else { f64::NAN };

        let mut min = f64::NAN;
        let mut max = f64::NAN;
        let mut range = f64::NAN;
        if n > 0 {
            min = valid_values.iter().copied().fold(f64::INFINITY, f64::min);
            max = valid_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            range = max - min;
        }

        let mut variance = f64::NAN;
        let mut std_dev = f64::NAN;
        if n > 1 {
            let sq_sum: f64 = valid_values.iter().map(|&x| (x - mean).powi(2)).sum();
            variance = sq_sum / ((n - 1) as f64);
            std_dev = variance.sqrt();
        }

        let mut skewness = f64::NAN;
        if n > 2 && std_dev > 0.0 {
            let cubed_sum: f64 = valid_values.iter().map(|&x| ((x - mean) / std_dev).powi(3)).sum();
            skewness = (n as f64) / (((n - 1) as f64) * ((n - 2) as f64)) * cubed_sum;
        }

        let mut kurtosis = f64::NAN;
        if n > 3 && std_dev > 0.0 {
            let fourth_sum: f64 = valid_values.iter().map(|&x| ((x - mean) / std_dev).powi(4)).sum();
            let term1 = ((n * (n + 1)) as f64) / (((n - 1) * (n - 2) * (n - 3)) as f64) * fourth_sum;
            let term2 = (3.0 * ((n - 1) as f64).powi(2)) / (((n - 2) * (n - 3)) as f64);
            kurtosis = term1 - term2;
        }

        let display_name = var.label.clone().unwrap_or_else(|| var.name.clone());
        let mut row = vec![display_name, format!("{}", n)];

        for stat in &options.statistics {
            let formatted = match stat {
                Statistic::Mean => format_val(mean, var.print_decimals),
                Statistic::StdDev => format_val(std_dev, var.print_decimals),
                Statistic::Min => format_val(min, var.print_decimals),
                Statistic::Max => format_val(max, var.print_decimals),
                Statistic::Variance => format_val(variance, var.print_decimals),
                Statistic::Skewness => format_val(skewness, 3), // Skewness standard is 3 decimals
                Statistic::Kurtosis => format_val(kurtosis, 3), // Kurtosis standard is 3 decimals
                Statistic::Sum => format_val(sum, var.print_decimals),
                Statistic::Range => format_val(range, var.print_decimals),
            };
            row.push(formatted);
        }
        rows.push(row);
    }

    let table = Table {
        title: "Descriptive Statistics".to_string(),
        column_headers,
        rows,
        footnotes: vec![format!("Valid N (listwise) = {}", listwise_n)],
    };

    vec![OutputItem::Table(table)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxstat_core::{Dataset, Variable};
    use arrow::array::{RecordBatch, Float64Array};
    use arrow::datatypes::{Schema, Field, DataType};
    use std::sync::Arc;

    #[test]
    fn test_descriptives_calculations() {
        // Create schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
        ]));

        // Create data: 1.0, 2.0, 3.0, 4.0, 5.0 (Mean = 3.0, StdDev = sqrt(2.5) = 1.5811)
        let x_array = Arc::new(Float64Array::from(vec![
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(4.0),
            Some(5.0),
        ])) as Arc<dyn Array>;

        let batch = RecordBatch::try_new(schema, vec![x_array]).unwrap();

        let mut dataset = Dataset::new();
        dataset.batches.push(batch);

        let mut var = Variable::numeric("x");
        var.print_decimals = 2;
        dataset.add_variable(var);

        // Run descriptives
        let options = DescriptivesOptions {
            variables: vec!["x".to_string()],
            statistics: vec![
                Statistic::Mean,
                Statistic::StdDev,
                Statistic::Min,
                Statistic::Max,
                Statistic::Sum,
                Statistic::Variance,
                Statistic::Skewness,
                Statistic::Kurtosis,
            ],
        };

        let output = run(&dataset, &options);
        assert_eq!(output.len(), 1);

        if let OutputItem::Table(table) = &output[0] {
            assert_eq!(table.rows.len(), 1);
            let row = &table.rows[0];
            assert_eq!(row[0], "x"); // Variable name
            assert_eq!(row[1], "5"); // N
            assert_eq!(row[2], "3.00"); // Mean
            assert_eq!(row[3], "1.58"); // StdDev (approx 1.5811)
            assert_eq!(row[4], "1.00"); // Min
            assert_eq!(row[5], "5.00"); // Max
            assert_eq!(row[6], "15.00"); // Sum
            assert_eq!(row[7], "2.50"); // Variance
            assert_eq!(row[8], "0.000"); // Skewness (symmetric)
            assert_eq!(row[9], "-1.200"); // Kurtosis (flat)
        } else {
            panic!("Expected OutputItem::Table");
        }
    }
}
