//! Excel (.xlsx, .xls) import for OxStat datasets.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanBuilder, Float64Builder, StringBuilder};
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use calamine::{Data, Reader, open_workbook_auto, Range};
use oxstat_core::{Dataset, MeasureLevel, Variable, VariableType};

/// Read an Excel file sheet into a Dataset.
pub fn read_excel<P: AsRef<Path>>(path: P, sheet_name: Option<&str>) -> Result<Dataset, Box<dyn std::error::Error>> {
    let mut workbook = open_workbook_auto(path)?;
    let target_sheet = match sheet_name {
        Some(name) => name.to_string(),
        None => workbook.sheet_names()
            .first()
            .ok_or("Workbook has no sheets")?
            .clone(),
    };

    let range = workbook.worksheet_range(&target_sheet)?;
    read_range(&range)
}

/// Convert a calamine Range to a Dataset.
pub fn read_range(range: &Range<Data>) -> Result<Dataset, Box<dyn std::error::Error>> {
    let (height, width) = range.get_size();
    if height < 2 || width == 0 {
        return Err("Worksheet has no data or columns".into());
    }

    // Read headers
    let mut col_names = Vec::with_capacity(width);
    for i in 0..width {
        let cell = range.get_value((0, i as u32)).unwrap_or(&Data::Empty);
        let name = match cell {
            Data::String(s) => s.trim().to_string(),
            Data::Empty => format!("col_{}", i),
            other => other.to_string().trim().to_string(),
        };
        col_names.push(name);
    }

    // Infer Arrow types
    let mut arrow_types = vec![ArrowDataType::Float64; width];
    for col_idx in 0..width {
        let mut has_string = false;
        let mut has_bool = false;
        let mut has_numeric = false;
        for row_idx in 1..height {
            if let Some(cell) = range.get_value((row_idx as u32, col_idx as u32)) {
                match cell {
                    Data::String(_) | Data::DateTime(_) | Data::DateTimeIso(_) => has_string = true,
                    Data::Int(_) | Data::Float(_) => has_numeric = true,
                    Data::Bool(_) => has_bool = true,
                    _ => {}
                }
            }
        }
        arrow_types[col_idx] = if has_string {
            ArrowDataType::Utf8
        } else if has_numeric {
            ArrowDataType::Float64
        } else if has_bool {
            ArrowDataType::Boolean
        } else {
            ArrowDataType::Utf8
        };
    }

    let mut fields = Vec::with_capacity(width);
    let mut columns = Vec::with_capacity(width);

    // Build columns
    for col_idx in 0..width {
        let name = &col_names[col_idx];
        let dtype = arrow_types[col_idx].clone();
        fields.push(Field::new(name, dtype.clone(), true));

        match dtype {
            ArrowDataType::Float64 => {
                let mut builder = Float64Builder::with_capacity(height - 1);
                for row_idx in 1..height {
                    if let Some(cell) = range.get_value((row_idx as u32, col_idx as u32)) {
                        match cell {
                            Data::Int(val) => builder.append_value(*val as f64),
                            Data::Float(val) => builder.append_value(*val),
                            _ => builder.append_null(),
                        }
                    } else {
                        builder.append_null();
                    }
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef);
            }
            ArrowDataType::Boolean => {
                let mut builder = BooleanBuilder::with_capacity(height - 1);
                for row_idx in 1..height {
                    if let Some(cell) = range.get_value((row_idx as u32, col_idx as u32)) {
                        match cell {
                            Data::Bool(val) => builder.append_value(*val),
                            _ => builder.append_null(),
                        }
                    } else {
                        builder.append_null();
                    }
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef);
            }
            _ => {
                let mut builder = StringBuilder::with_capacity(height - 1, 1024);
                for row_idx in 1..height {
                    if let Some(cell) = range.get_value((row_idx as u32, col_idx as u32)) {
                        match cell {
                            Data::Empty | Data::Error(_) => builder.append_null(),
                            Data::String(val) => builder.append_value(val),
                            Data::DateTimeIso(val) => builder.append_value(val),
                            other => builder.append_value(&other.to_string()),
                        }
                    } else {
                        builder.append_null();
                    }
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef);
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), columns)?;

    let mut dataset = Dataset::new();
    dataset.batches = vec![batch];

    // Build variables
    for field in schema.fields() {
        let name = field.name().clone();
        let var_type = match field.data_type() {
            ArrowDataType::Utf8 => {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_excel_error() {
        let res = read_excel("nonexistent_file.xlsx", None);
        assert!(res.is_err());
    }
}
