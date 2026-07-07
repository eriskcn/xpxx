//! Parquet import/export for OxStat datasets.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use arrow::array::Array;
use arrow::datatypes::{Schema, Field, DataType as ArrowDataType};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;

use oxstat_core::{Dataset, MeasureLevel, Variable, VariableType, MissingValues};

#[derive(Serialize, Deserialize)]
struct VariableMetadata {
    label: Option<String>,
    value_labels: HashMap<i64, String>,
    missing_discrete: Vec<f64>,
    missing_range: Option<(f64, f64)>,
    print_width: u8,
    print_decimals: u8,
    measure: MeasureLevel,
}

/// Read Parquet file into Dataset.
pub fn read_parquet<P: AsRef<Path>>(path: P) -> Result<Dataset, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let reader = builder.build()?;

    let mut batches = Vec::new();
    for batch in reader {
        batches.push(batch?);
    }

    let mut dataset = Dataset::new();
    dataset.batches = batches;

    for field in schema.fields() {
        let name = field.name().clone();
        let var = if let Some(meta_str) = field.metadata().get("oxstat:variable_metadata") {
            if let Ok(meta) = serde_json::from_str::<VariableMetadata>(meta_str) {
                let var_type = match field.data_type() {
                    ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 => {
                        VariableType::String(meta.print_width as u32)
                    }
                    _ => VariableType::Numeric,
                };
                Variable {
                    name: name.clone(),
                    var_type,
                    measure: meta.measure,
                    label: meta.label,
                    value_labels: meta.value_labels,
                    missing: MissingValues {
                        discrete: meta.missing_discrete,
                        range: meta.missing_range,
                    },
                    print_width: meta.print_width,
                    print_decimals: meta.print_decimals,
                }
            } else {
                default_variable_from_field(field, &dataset.batches)
            }
        } else {
            default_variable_from_field(field, &dataset.batches)
        };
        dataset.add_variable(var);
    }

    Ok(dataset)
}

fn default_variable_from_field(field: &Field, batches: &[RecordBatch]) -> Variable {
    let name = field.name().clone();
    let var_type = match field.data_type() {
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 => {
            let mut max_width = 0;
            if !batches.is_empty() {
                if let Some(col_idx) = batches[0].schema().index_of(&name).ok() {
                    for batch in batches {
                        let column = batch.column(col_idx);
                        if let Some(arr) = column.as_any().downcast_ref::<arrow::array::StringArray>() {
                            for i in 0..arr.len() {
                                if !arr.is_null(i) {
                                    max_width = max_width.max(arr.value(i).len());
                                }
                            }
                        }
                    }
                }
            }
            let width = if max_width == 0 { 8 } else { max_width };
            VariableType::String(width as u32)
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
    var
}

/// Write Dataset to Parquet file.
pub fn write_parquet<P: AsRef<Path>>(dataset: &Dataset, path: P) -> Result<(), Box<dyn std::error::Error>> {
    if dataset.batches.is_empty() {
        return Err("Dataset has no record batches".into());
    }

    let original_schema = dataset.batches[0].schema();
    let mut new_fields = Vec::new();

    for (idx, field) in original_schema.fields().iter().enumerate() {
        let mut field_metadata = field.metadata().clone();
        if let Some(var) = dataset.variables.get(idx) {
            let meta = VariableMetadata {
                label: var.label.clone(),
                value_labels: var.value_labels.clone(),
                missing_discrete: var.missing.discrete.clone(),
                missing_range: var.missing.range,
                print_width: var.print_width,
                print_decimals: var.print_decimals,
                measure: var.measure,
            };
            let json_str = serde_json::to_string(&meta)?;
            field_metadata.insert("oxstat:variable_metadata".to_string(), json_str);
        }
        let field_owned = field.as_ref().clone().with_metadata(field_metadata);
        new_fields.push(field_owned);
    }

    let new_schema = Arc::new(Schema::new(new_fields));
    let mut updated_batches = Vec::with_capacity(dataset.batches.len());
    for batch in &dataset.batches {
        let new_batch = RecordBatch::try_new(new_schema.clone(), batch.columns().to_vec())?;
        updated_batches.push(new_batch);
    }

    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, new_schema, Some(props))?;

    for batch in &updated_batches {
        writer.write(batch)?;
    }
    writer.close()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;

    #[test]
    fn test_parquet_read_write() {
        let temp_dir = std::env::temp_dir();
        let parquet_path = temp_dir.join("test_data.parquet");

        let schema = Arc::new(Schema::new(vec![
            Field::new("score", ArrowDataType::Float64, true),
        ]));
        let score_array = Arc::new(Float64Array::from(vec![Some(99.0), Some(88.0)])) as Arc<dyn arrow::array::Array>;
        let batch = RecordBatch::try_new(schema, vec![score_array]).unwrap();

        let mut dataset = Dataset::new();
        dataset.batches.push(batch);

        let mut var = Variable::numeric("score");
        var.label = Some("Test Score".to_string());
        dataset.add_variable(var);

        write_parquet(&dataset, &parquet_path).unwrap();

        let loaded = read_parquet(&parquet_path).unwrap();
        assert_eq!(loaded.n_variables(), 1);
        assert_eq!(loaded.n_cases(), 2);

        let var_loaded = loaded.variable("score").unwrap();
        assert_eq!(var_loaded.label, Some("Test Score".to_string()));

        let _ = std::fs::remove_file(parquet_path);
    }
}
