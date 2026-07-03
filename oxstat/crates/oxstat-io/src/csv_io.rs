//! CSV import/export for OxStat datasets.

use std::fs::File;
use std::io::Seek;
use std::path::Path;
use std::sync::Arc;

use arrow::array::Array;
#[allow(deprecated)]
use arrow::csv::reader::infer_file_schema;
#[allow(deprecated)]
use arrow::csv::{ReaderBuilder, Writer};
use oxstat_core::{Dataset, MeasureLevel, Variable, VariableType};

/// Read a CSV file into a Dataset.
pub fn read_csv<P: AsRef<Path>>(path: P) -> Result<Dataset, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;

    // Infer schema from first 100 records
    let (schema, _) = infer_file_schema(&mut file, b',', Some(100), true)?;
    file.rewind()?;

    // Build Arrow CSV reader
    let reader = ReaderBuilder::new(Arc::new(schema.clone()))
        .has_header(true)
        .build(file)?;

    let mut batches = Vec::new();
    for batch in reader {
        batches.push(batch?);
    }

    let mut dataset = Dataset::new();
    dataset.batches = batches;

    // Create variables from fields
    for field in schema.fields() {
        let name = field.name().clone();

        // Determine type (Numeric or String)
        let var_type = match field.data_type() {
            arrow::datatypes::DataType::Utf8 | arrow::datatypes::DataType::LargeUtf8 => {
                let mut max_width = 0;
                if let Some(col_idx) = schema.index_of(&name).ok() {
                    for batch in &dataset.batches {
                        let column = batch.column(col_idx);
                        if let Some(arr) = column.as_any().downcast_ref::<arrow::array::StringArray>() {
                            for i in 0..arr.len() {
                                if !arr.is_null(i) {
                                    max_width = max_width.max(arr.value(i).len());
                                }
                            }
                        } else if let Some(arr) = column.as_any().downcast_ref::<arrow::array::LargeStringArray>() {
                            for i in 0..arr.len() {
                                if !arr.is_null(i) {
                                    max_width = max_width.max(arr.value(i).len());
                                }
                            }
                        }
                    }
                }
                let final_width = if max_width == 0 { 8 } else { max_width };
                VariableType::String(final_width as u32)
            }
            _ => VariableType::Numeric,
        };

        let mut var = match var_type {
            VariableType::Numeric => Variable::numeric(name),
            VariableType::String(w) => Variable::string(name, w),
        };

        var.measure = match var_type {
            VariableType::Numeric => MeasureLevel::Scale,
            VariableType::String(_) => MeasureLevel::Nominal,
        };

        dataset.add_variable(var);
    }

    Ok(dataset)
}

/// Write a Dataset to a CSV file.
pub fn write_csv<P: AsRef<Path>>(dataset: &Dataset, path: P) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut writer = Writer::new(file);
    for batch in &dataset.batches {
        writer.write(batch)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_read_write() {
        let temp_dir = std::env::temp_dir();
        let csv_path = temp_dir.join("test_input.csv");
        let output_path = temp_dir.join("test_output.csv");

        // Write sample CSV file
        let csv_data = "id,name,score\n1,Alice,95.5\n2,Bob,88.0\n3,Charlie,92.3\n";
        std::fs::write(&csv_path, csv_data).unwrap();

        // Read CSV
        let dataset = read_csv(&csv_path).unwrap();

        // Verify variables
        assert_eq!(dataset.n_variables(), 3);
        assert_eq!(dataset.n_cases(), 3);

        let var_id = dataset.variable("id").unwrap();
        assert_eq!(var_id.var_type, VariableType::Numeric);
        assert_eq!(var_id.measure, MeasureLevel::Scale);

        let var_name = dataset.variable("name").unwrap();
        assert_eq!(var_name.var_type, VariableType::String(7)); // "Charlie" is 7 bytes
        assert_eq!(var_name.measure, MeasureLevel::Nominal);

        let var_score = dataset.variable("score").unwrap();
        assert_eq!(var_score.var_type, VariableType::Numeric);
        assert_eq!(var_score.measure, MeasureLevel::Scale);

        // Write CSV
        write_csv(&dataset, &output_path).unwrap();

        // Read written CSV back
        let dataset2 = read_csv(&output_path).unwrap();
        assert_eq!(dataset2.n_variables(), 3);
        assert_eq!(dataset2.n_cases(), 3);

        // Clean up
        let _ = std::fs::remove_file(csv_path);
        let _ = std::fs::remove_file(output_path);
    }
}

