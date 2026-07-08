//! oxstat-transform: Data transformations.
//!
//! SELECT IF, SORT CASES, AGGREGATE, MATCH FILES / ADD FILES, RESHAPE.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Builder, StringBuilder, RecordBatch, ArrayBuilder};
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use oxstat_core::{Dataset, MeasureLevel, MissingValues, Value, Variable, VariableType};
use oxstat_expr::Expr;

// Helper to get case value at (batch, row) for a variable index
fn get_cell_value(dataset: &Dataset, batch_idx: usize, row_idx: usize, var_idx: usize) -> Value {
    let var = &dataset.variables[var_idx];
    let batch = &dataset.batches[batch_idx];
    let column = batch.column(var_idx);

    match var.var_type {
        VariableType::Numeric => {
            if column.is_null(row_idx) {
                Value::SystemMissing
            } else {
                let array = column
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .unwrap();
                let val = array.value(row_idx);
                if var.missing.is_user_missing(val) {
                    Value::SystemMissing
                } else {
                    Value::Numeric(val)
                }
            }
        }
        VariableType::String(_) => {
            if column.is_null(row_idx) {
                Value::SystemMissing
            } else {
                let array = column
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .unwrap();
                Value::String(array.value(row_idx).to_string())
            }
        }
    }
}

// Reusable array from Value list builder
fn array_from_values(values: &[Value]) -> (ArrayRef, VariableType) {
    let is_string = values.iter().any(|v| matches!(v, Value::String(_)));
    if is_string {
        let mut builder = StringBuilder::new();
        let mut max_width = 0;
        for val in values {
            match val {
                Value::String(s) => {
                    builder.append_value(s);
                    max_width = max_width.max(s.len());
                }
                _ => builder.append_null(),
            }
        }
        let array = Arc::new(builder.finish()) as ArrayRef;
        let width = if max_width == 0 { 8 } else { max_width };
        (array, VariableType::String(width as u32))
    } else {
        let mut builder = Float64Builder::new();
        for val in values {
            match val {
                Value::Numeric(n) => builder.append_value(*n),
                _ => builder.append_null(),
            }
        }
        let array = Arc::new(builder.finish()) as ArrayRef;
        (array, VariableType::Numeric)
    }
}

// Resolves a flat row index to (batch_idx, row_in_batch)
fn get_batch_row_index(dataset: &Dataset, mut row_idx: usize) -> (usize, usize) {
    let mut batch_idx = 0;
    while row_idx >= dataset.batches[batch_idx].num_rows() {
        row_idx -= dataset.batches[batch_idx].num_rows();
        batch_idx += 1;
    }
    (batch_idx, row_idx)
}

// Build a column array from a list of positions
fn build_column_from_positions(
    dataset: &Dataset,
    var_idx: usize,
    positions: &[(usize, usize)],
) -> ArrayRef {
    let var = &dataset.variables[var_idx];
    match var.var_type {
        VariableType::Numeric => {
            let mut builder = Float64Builder::with_capacity(positions.len());
            for &(b, r) in positions {
                let col = dataset.batches[b].column(var_idx);
                if col.is_null(r) {
                    builder.append_null();
                } else {
                    let arr = col.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap();
                    builder.append_value(arr.value(r));
                }
            }
            Arc::new(builder.finish()) as ArrayRef
        }
        VariableType::String(_) => {
            let mut builder = StringBuilder::with_capacity(positions.len(), positions.len() * 8);
            for &(b, r) in positions {
                let col = dataset.batches[b].column(var_idx);
                if col.is_null(r) {
                    builder.append_null();
                } else {
                    let arr = col.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
                    builder.append_value(arr.value(r));
                }
            }
            Arc::new(builder.finish()) as ArrayRef
        }
    }
}

// Build null array
fn create_null_array(var_type: &VariableType, length: usize) -> ArrayRef {
    match var_type {
        VariableType::Numeric => {
            let mut builder = Float64Builder::with_capacity(length);
            for _ in 0..length {
                builder.append_null();
            }
            Arc::new(builder.finish()) as ArrayRef
        }
        VariableType::String(_) => {
            let mut builder = StringBuilder::with_capacity(length, length * 8);
            for _ in 0..length {
                builder.append_null();
            }
            Arc::new(builder.finish()) as ArrayRef
        }
    }
}

// Concatenate batches for a single column
fn concat_column_batches(dataset: &Dataset, col_idx: usize) -> Result<ArrayRef, String> {
    let arrays: Vec<&dyn arrow::array::Array> = dataset.batches.iter()
        .map(|b| b.column(col_idx).as_ref())
        .collect();
    arrow::compute::concat(&arrays).map_err(|e| e.to_string())
}

// Group Key Generator
fn get_group_key(dataset: &Dataset, batch_idx: usize, row_idx: usize, break_indices: &[usize]) -> String {
    let mut key = String::new();
    for &var_idx in break_indices {
        let val = get_cell_value(dataset, batch_idx, row_idx, var_idx);
        match val {
            Value::Numeric(n) => {
                key.push_str(&format!("N:{:?},", n.to_bits()));
            }
            Value::String(s) => {
                key.push_str(&format!("S:{},", s));
            }
            Value::SystemMissing => {
                key.push_str("M,");
            }
        }
    }
    key
}

/// 1.5.1 SELECT IF: deletes rows matching condition.
pub fn select_if(dataset: &Dataset, expr: &Expr) -> Result<Dataset, String> {
    if dataset.batches.is_empty() {
        return Ok(Dataset::new());
    }

    let mut filtered_batches = Vec::new();
    for (batch_idx, batch) in dataset.batches.iter().enumerate() {
        let num_rows = batch.num_rows();
        let mut filter_builder = arrow::array::BooleanBuilder::with_capacity(num_rows);

        for row_idx in 0..num_rows {
            let val = expr.eval(dataset, batch_idx, row_idx);
            let keep = match val {
                Value::Numeric(n) => n != 0.0,
                _ => false,
            };
            filter_builder.append_value(keep);
        }

        let filter_array = filter_builder.finish();
        let filtered_batch = arrow::compute::filter_record_batch(batch, &filter_array)
            .map_err(|e| e.to_string())?;

        if filtered_batch.num_rows() > 0 {
            filtered_batches.push(filtered_batch);
        }
    }

    let mut out = Dataset::new();
    out.batches = filtered_batches;
    for var in &dataset.variables {
        out.add_variable(var.clone());
    }
    Ok(out)
}

/// 1.5.2 SORT CASES
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    Ascending,
    Descending,
}

#[derive(Debug, Clone)]
pub struct SortKey {
    pub var_name: String,
    pub order: SortOrder,
}

pub fn sort_cases(dataset: &Dataset, keys: &[SortKey]) -> Result<Dataset, String> {
    if dataset.batches.is_empty() {
        return Ok(Dataset::new());
    }

    let sort_columns = keys.iter().map(|key| {
        let idx = dataset.variable_index(&key.var_name)
            .ok_or_else(|| format!("Var not found: {}", key.var_name))?;
        let concatenated = concat_column_batches(dataset, idx)?;
        let options = arrow::compute::SortOptions {
            descending: key.order == SortOrder::Descending,
            nulls_first: key.order == SortOrder::Ascending,
        };
        Ok(arrow::compute::SortColumn {
            values: concatenated,
            options: Some(options),
        })
    }).collect::<Result<Vec<_>, String>>()?;

    let indices = arrow::compute::lexsort_to_indices(&sort_columns, None)
        .map_err(|e| e.to_string())?;

    let mut sorted_columns = Vec::new();
    let num_vars = dataset.n_variables();
    for idx in 0..num_vars {
        let concatenated = concat_column_batches(dataset, idx)?;
        let sorted_array = arrow::compute::take(&concatenated, &indices, None)
            .map_err(|e| e.to_string())?;
        sorted_columns.push(sorted_array);
    }

    let fields: Vec<arrow::datatypes::FieldRef> = dataset.variables.iter()
        .map(|v| {
            let dt = match v.var_type {
                VariableType::Numeric => ArrowDataType::Float64,
                VariableType::String(_) => ArrowDataType::Utf8,
            };
            std::sync::Arc::new(Field::new(&v.name, dt, true))
        }).collect();
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, sorted_columns).map_err(|e| e.to_string())?;

    let mut out = Dataset::new();
    out.batches.push(batch);
    for var in &dataset.variables {
        out.add_variable(var.clone());
    }
    Ok(out)
}

/// 1.5.3 AGGREGATE
#[derive(Debug, Clone)]
pub enum AggFunction {
    Sum(String),
    Mean(String),
    Min(String),
    Max(String),
    Count,
}

#[derive(Debug, Clone)]
pub struct AggSpec {
    pub target_var: String,
    pub function: AggFunction,
}

pub fn aggregate(
    dataset: &Dataset,
    break_vars: &[String],
    specs: &[AggSpec],
) -> Result<Dataset, String> {
    if dataset.batches.is_empty() {
        return Ok(Dataset::new());
    }

    let break_indices: Vec<usize> = break_vars.iter()
        .map(|name| dataset.variable_index(name).ok_or_else(|| format!("Break var not found: {}", name)))
        .collect::<Result<_, String>>()?;

    let mut groups: HashMap<String, (Vec<Value>, Vec<(usize, usize)>)> = HashMap::new();

    for batch_idx in 0..dataset.batches.len() {
        let n_rows = dataset.batches[batch_idx].num_rows();
        for row_idx in 0..n_rows {
            let key = get_group_key(dataset, batch_idx, row_idx, &break_indices);
            groups.entry(key).or_insert_with(|| {
                let break_vals = break_indices.iter()
                    .map(|&idx| get_cell_value(dataset, batch_idx, row_idx, idx))
                    .collect();
                (break_vals, Vec::new())
            }).1.push((batch_idx, row_idx));
        }
    }

    // Build outputs
    let n_groups = groups.len();
    let mut break_builders: Vec<Box<dyn ArrayBuilder>> = Vec::new();
    for &idx in &break_indices {
        match dataset.variables[idx].var_type {
            VariableType::Numeric => break_builders.push(Box::new(Float64Builder::new())),
            VariableType::String(_) => break_builders.push(Box::new(StringBuilder::new())),
        }
    }

    let mut agg_builders: Vec<Float64Builder> = specs.iter()
        .map(|_| Float64Builder::new())
        .collect();

    for (_key, (break_vals, positions)) in groups {
        for (i, val) in break_vals.into_iter().enumerate() {
            match val {
                Value::Numeric(n) => {
                    let b = break_builders[i].as_any_mut().downcast_mut::<Float64Builder>().unwrap();
                    b.append_value(n);
                }
                Value::String(s) => {
                    let b = break_builders[i].as_any_mut().downcast_mut::<StringBuilder>().unwrap();
                    b.append_value(s);
                }
                Value::SystemMissing => {
                    match dataset.variables[break_indices[i]].var_type {
                        VariableType::Numeric => {
                            let b = break_builders[i].as_any_mut().downcast_mut::<Float64Builder>().unwrap();
                            b.append_null();
                        }
                        VariableType::String(_) => {
                            let b = break_builders[i].as_any_mut().downcast_mut::<StringBuilder>().unwrap();
                            b.append_null();
                        }
                    }
                }
            }
        }

        for (i, spec) in specs.iter().enumerate() {
            let agg_val = match &spec.function {
                AggFunction::Count => Value::Numeric(positions.len() as f64),
                AggFunction::Sum(src_name) => {
                    let src_idx = dataset.variable_index(src_name).ok_or_else(|| format!("Var not found: {}", src_name))?;
                    let nums = get_group_numeric_values(dataset, &positions, src_idx);
                    if nums.is_empty() { Value::SystemMissing } else { Value::Numeric(nums.iter().sum()) }
                }
                AggFunction::Mean(src_name) => {
                    let src_idx = dataset.variable_index(src_name).ok_or_else(|| format!("Var not found: {}", src_name))?;
                    let nums = get_group_numeric_values(dataset, &positions, src_idx);
                    if nums.is_empty() { Value::SystemMissing } else { Value::Numeric(nums.iter().sum::<f64>() / nums.len() as f64) }
                }
                AggFunction::Min(src_name) => {
                    let src_idx = dataset.variable_index(src_name).ok_or_else(|| format!("Var not found: {}", src_name))?;
                    let nums = get_group_numeric_values(dataset, &positions, src_idx);
                    if nums.is_empty() { Value::SystemMissing } else { Value::Numeric(nums.iter().copied().fold(f64::INFINITY, f64::min)) }
                }
                AggFunction::Max(src_name) => {
                    let src_idx = dataset.variable_index(src_name).ok_or_else(|| format!("Var not found: {}", src_name))?;
                    let nums = get_group_numeric_values(dataset, &positions, src_idx);
                    if nums.is_empty() { Value::SystemMissing } else { Value::Numeric(nums.iter().copied().fold(f64::NEG_INFINITY, f64::max)) }
                }
            };

            match agg_val {
                Value::Numeric(n) => agg_builders[i].append_value(n),
                _ => agg_builders[i].append_null(),
            }
        }
    }

    let mut out_fields = Vec::new();
    let mut out_columns: Vec<ArrayRef> = Vec::new();

    // Add break variables to schema
    for (i, &idx) in break_indices.iter().enumerate() {
        let var = &dataset.variables[idx];
        let (dt, array) = match var.var_type {
            VariableType::Numeric => {
                let mut b = break_builders[i].as_any_mut().downcast_mut::<Float64Builder>().unwrap();
                (ArrowDataType::Float64, Arc::new(b.finish()) as ArrayRef)
            }
            VariableType::String(_) => {
                let mut b = break_builders[i].as_any_mut().downcast_mut::<StringBuilder>().unwrap();
                (ArrowDataType::Utf8, Arc::new(b.finish()) as ArrayRef)
            }
        };
        out_fields.push(Field::new(&var.name, dt, true));
        out_columns.push(array);
    }

    // Add aggregated variables to schema
    for (i, spec) in specs.iter().enumerate() {
        out_fields.push(Field::new(&spec.target_var, ArrowDataType::Float64, true));
        let array = Arc::new(agg_builders[i].finish()) as ArrayRef;
        out_columns.push(array);
    }

    let schema = Arc::new(Schema::new(out_fields));
    let batch = RecordBatch::try_new(schema, out_columns).map_err(|e| e.to_string())?;

    let mut out = Dataset::new();
    out.batches.push(batch);

    for &idx in &break_indices {
        out.add_variable(dataset.variables[idx].clone());
    }
    for spec in specs {
        out.add_variable(Variable::numeric(&spec.target_var));
    }

    Ok(out)
}

fn get_group_numeric_values(dataset: &Dataset, positions: &[(usize, usize)], var_idx: usize) -> Vec<f64> {
    positions.iter()
        .map(|&(b, r)| get_cell_value(dataset, b, r, var_idx))
        .filter_map(|v| match v {
            Value::Numeric(n) => Some(n),
            _ => None,
        })
        .collect()
}

/// 1.5.4 ADD FILES
pub fn add_files(datasets: &[&Dataset]) -> Result<Dataset, String> {
    if datasets.is_empty() {
        return Ok(Dataset::new());
    }

    // Find union of variables
    let mut var_map = HashMap::new();
    let mut ordered_var_names = Vec::new();

    for ds in datasets {
        for var in &ds.variables {
            let key = var.name.to_uppercase();
            if !var_map.contains_key(&key) {
                var_map.insert(key.clone(), var.clone());
                ordered_var_names.push(key);
            }
        }
    }

    let total_len: usize = datasets.iter().map(|ds| ds.n_cases()).sum();
    let mut out_columns: Vec<ArrayRef> = Vec::new();

    for var_key in &ordered_var_names {
        let var_def = var_map.get(var_key).unwrap();
        let mut arrays = Vec::with_capacity(datasets.len());

        for ds in datasets {
            if let Some(idx) = ds.variable_index(&var_def.name) {
                let col = concat_column_batches(ds, idx)?;
                arrays.push(col);
            } else {
                let null_arr = create_null_array(&var_def.var_type, ds.n_cases());
                arrays.push(null_arr);
            }
        }

        let dyn_arrays: Vec<&dyn arrow::array::Array> = arrays.iter().map(|a| a.as_ref()).collect();
        let concatenated = arrow::compute::concat(&dyn_arrays).map_err(|e| e.to_string())?;
        out_columns.push(concatenated);
    }

    let fields: Vec<arrow::datatypes::FieldRef> = ordered_var_names.iter()
        .map(|k| {
            let var = var_map.get(k).unwrap();
            let dt = match var.var_type {
                VariableType::Numeric => ArrowDataType::Float64,
                VariableType::String(_) => ArrowDataType::Utf8,
            };
            std::sync::Arc::new(Field::new(&var.name, dt, true))
        }).collect();
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, out_columns).map_err(|e| e.to_string())?;

    let mut out = Dataset::new();
    out.batches.push(batch);
    for key in &ordered_var_names {
        out.add_variable(var_map.get(key).unwrap().clone());
    }

    Ok(out)
}

/// 1.5.4 MATCH FILES: One-to-One
pub fn match_files_one_to_one(datasets: &[&Dataset]) -> Result<Dataset, String> {
    if datasets.is_empty() {
        return Ok(Dataset::new());
    }

    let max_cases = datasets.iter().map(|d| d.n_cases()).max().unwrap();
    let mut out_columns = Vec::new();
    let mut out_variables = Vec::new();
    let mut seen_vars = HashMap::new();

    for ds in datasets {
        let n_cases = ds.n_cases();
        for (i, var) in ds.variables.iter().enumerate() {
            let key = var.name.to_uppercase();
            if !seen_vars.contains_key(&key) {
                seen_vars.insert(key, true);
                out_variables.push(var.clone());

                let col = concat_column_batches(ds, i)?;
                if n_cases < max_cases {
                    let pad = create_null_array(&var.var_type, max_cases - n_cases);
                    let concatenated = arrow::compute::concat(&[col.as_ref(), pad.as_ref()]).map_err(|e| e.to_string())?;
                    out_columns.push(concatenated);
                } else {
                    out_columns.push(col);
                }
            }
        }
    }

    let fields: Vec<arrow::datatypes::FieldRef> = out_variables.iter()
        .map(|v| {
            let dt = match v.var_type {
                VariableType::Numeric => ArrowDataType::Float64,
                VariableType::String(_) => ArrowDataType::Utf8,
            };
            std::sync::Arc::new(Field::new(&v.name, dt, true))
        }).collect();
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, out_columns).map_err(|e| e.to_string())?;

    let mut out = Dataset::new();
    out.batches.push(batch);
    for var in out_variables {
        out.add_variable(var);
    }
    Ok(out)
}

/// 1.5.4 MATCH FILES: Key Lookup Join
pub fn match_files_by_key(
    primary: &Dataset,
    lookup: &Dataset,
    by_vars: &[String],
) -> Result<Dataset, String> {
    if primary.batches.is_empty() {
        return Ok(Dataset::new());
    }

    let pri_indices: Vec<usize> = by_vars.iter()
        .map(|name| primary.variable_index(name).ok_or_else(|| format!("Key not found in primary: {}", name)))
        .collect::<Result<_, String>>()?;

    let lkp_indices: Vec<usize> = by_vars.iter()
        .map(|name| lookup.variable_index(name).ok_or_else(|| format!("Key not found in lookup: {}", name)))
        .collect::<Result<_, String>>()?;

    // Build hash map from lookup keys to lookup row position (batch, row)
    let mut lkp_map = HashMap::new();
    for b_idx in 0..lookup.batches.len() {
        let n_rows = lookup.batches[b_idx].num_rows();
        for r_idx in 0..n_rows {
            let key = get_group_key(lookup, b_idx, r_idx, &lkp_indices);
            lkp_map.entry(key).or_insert((b_idx, r_idx));
        }
    }

    let n_primary_cases = primary.n_cases();

    // Map each primary row to matching lookup row position
    let mut match_positions = Vec::with_capacity(n_primary_cases);
    for b_idx in 0..primary.batches.len() {
        let n_rows = primary.batches[b_idx].num_rows();
        for r_idx in 0..n_rows {
            let key = get_group_key(primary, b_idx, r_idx, &pri_indices);
            let pos = lkp_map.get(&key).copied();
            match_positions.push(pos);
        }
    }

    // Build columns list
    let mut out_variables = Vec::new();
    let mut out_columns = Vec::new();
    let mut seen_vars = HashMap::new();

    // Primary columns are copied entirely
    for (i, var) in primary.variables.iter().enumerate() {
        let key = var.name.to_uppercase();
        seen_vars.insert(key, true);
        out_variables.push(var.clone());
        let col = concat_column_batches(primary, i)?;
        out_columns.push(col);
    }

    // Aligned lookup columns
    for (i, var) in lookup.variables.iter().enumerate() {
        let key = var.name.to_uppercase();
        if !seen_vars.contains_key(&key) {
            seen_vars.insert(key, true);
            out_variables.push(var.clone());

            // Build aligned column array
            let arr = match var.var_type {
                VariableType::Numeric => {
                    let mut builder = Float64Builder::with_capacity(n_primary_cases);
                    for pos in &match_positions {
                        if let Some((b, r)) = *pos {
                            let col = lookup.batches[b].column(i);
                            if col.is_null(r) {
                                builder.append_null();
                            } else {
                                let val_arr = col.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap();
                                builder.append_value(val_arr.value(r));
                            }
                        } else {
                            builder.append_null();
                        }
                    }
                    Arc::new(builder.finish()) as ArrayRef
                }
                VariableType::String(_) => {
                    let mut builder = StringBuilder::with_capacity(n_primary_cases, n_primary_cases * 8);
                    for pos in &match_positions {
                        if let Some((b, r)) = *pos {
                            let col = lookup.batches[b].column(i);
                            if col.is_null(r) {
                                builder.append_null();
                            } else {
                                let val_arr = col.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
                                builder.append_value(val_arr.value(r));
                            }
                        } else {
                            builder.append_null();
                        }
                    }
                    Arc::new(builder.finish()) as ArrayRef
                }
            };
            out_columns.push(arr);
        }
    }

    let fields: Vec<arrow::datatypes::FieldRef> = out_variables.iter()
        .map(|v| {
            let dt = match v.var_type {
                VariableType::Numeric => ArrowDataType::Float64,
                VariableType::String(_) => ArrowDataType::Utf8,
            };
            std::sync::Arc::new(Field::new(&v.name, dt, true))
        }).collect();
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, out_columns).map_err(|e| e.to_string())?;

    let mut out = Dataset::new();
    out.batches.push(batch);
    for var in out_variables {
        out.add_variable(var);
    }
    Ok(out)
}

/// 1.5.5 RESHAPE: VARSTOCASES (Wide to Long)
#[derive(Debug, Clone)]
pub struct TransposeSpec {
    pub target_var: String,
    pub source_vars: Vec<String>,
}

pub fn vars_to_cases(
    dataset: &Dataset,
    specs: &[TransposeSpec],
    index_var: Option<&str>,
    keep_vars: &[String],
) -> Result<Dataset, String> {
    if dataset.batches.is_empty() {
        return Ok(Dataset::new());
    }

    let n_cases = dataset.n_cases();
    let n_transposed = specs[0].source_vars.len();
    let output_len = n_cases * n_transposed;

    // Generate output row indices mapping to original row indices (replicated keep columns)
    let mut keep_positions = Vec::with_capacity(output_len);
    for i in 0..n_cases {
        let pos = get_batch_row_index(dataset, i);
        for _ in 0..n_transposed {
            keep_positions.push(pos);
        }
    }

    let mut out_variables = Vec::new();
    let mut out_columns = Vec::new();

    // 1. Replicated keep variables
    for name in keep_vars {
        let idx = dataset.variable_index(name).ok_or_else(|| format!("Keep var not found: {}", name))?;
        out_variables.push(dataset.variables[idx].clone());
        let array = build_column_from_positions(dataset, idx, &keep_positions);
        out_columns.push(array);
    }

    // 2. Transposed specs columns
    for spec in specs {
        let mut values = Vec::with_capacity(output_len);
        for i in 0..n_cases {
            let (b, r) = get_batch_row_index(dataset, i);
            for j in 0..n_transposed {
                let src_name = &spec.source_vars[j];
                let src_idx = dataset.variable_index(src_name).ok_or_else(|| format!("Source var not found: {}", src_name))?;
                let val = get_cell_value(dataset, b, r, src_idx);
                values.push(val);
            }
        }
        let (array, var_type) = array_from_values(&values);
        out_variables.push(match var_type {
            VariableType::Numeric => Variable::numeric(&spec.target_var),
            VariableType::String(w) => Variable::string(&spec.target_var, w),
        });
        out_columns.push(array);
    }

    // 3. Generate index variable
    if let Some(index_name) = index_var {
        out_variables.push(Variable::numeric(index_name));
        let mut builder = Float64Builder::with_capacity(output_len);
        for _ in 0..n_cases {
            for j in 0..n_transposed {
                builder.append_value((j + 1) as f64);
            }
        }
        let array = Arc::new(builder.finish()) as ArrayRef;
        out_columns.push(array);
    }

    let fields: Vec<arrow::datatypes::FieldRef> = out_variables.iter()
        .map(|v| {
            let dt = match v.var_type {
                VariableType::Numeric => ArrowDataType::Float64,
                VariableType::String(_) => ArrowDataType::Utf8,
            };
            std::sync::Arc::new(Field::new(&v.name, dt, true))
        }).collect();
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, out_columns).map_err(|e| e.to_string())?;

    let mut out = Dataset::new();
    out.batches.push(batch);
    for var in out_variables {
        out.add_variable(var);
    }
    Ok(out)
}

/// 1.5.5 RESHAPE: CASESTOVARS (Long to Wide)
pub fn cases_to_vars(
    dataset: &Dataset,
    id_vars: &[String],
    index_var: &str,
    group_vars: &[String],
) -> Result<Dataset, String> {
    if dataset.batches.is_empty() {
        return Ok(Dataset::new());
    }

    let id_indices: Vec<usize> = id_vars.iter()
        .map(|name| dataset.variable_index(name).ok_or_else(|| format!("ID var not found: {}", name)))
        .collect::<Result<_, String>>()?;

    let idx_var_index = dataset.variable_index(index_var)
        .ok_or_else(|| format!("Index var not found: {}", index_var))?;

    // Group rows by unique ID
    let mut groups: HashMap<String, (Vec<Value>, Vec<(usize, usize)>)> = HashMap::new();
    for b_idx in 0..dataset.batches.len() {
        let n_rows = dataset.batches[b_idx].num_rows();
        for r_idx in 0..n_rows {
            let key = get_group_key(dataset, b_idx, r_idx, &id_indices);
            groups.entry(key).or_insert_with(|| {
                let id_vals = id_indices.iter()
                    .map(|&idx| get_cell_value(dataset, b_idx, r_idx, idx))
                    .collect();
                (id_vals, Vec::new())
            }).1.push((b_idx, r_idx));
        }
    }

    // Collect all unique index values (sorted)
    let mut index_val_map = HashMap::new();
    for b_idx in 0..dataset.batches.len() {
        let n_rows = dataset.batches[b_idx].num_rows();
        for r_idx in 0..n_rows {
            let val = get_cell_value(dataset, b_idx, r_idx, idx_var_index);
            match val {
                Value::Numeric(n) => {
                    index_val_map.insert(format!("{}", n), val);
                }
                Value::String(s) => {
                    index_val_map.insert(s.clone(), Value::String(s));
                }
                Value::SystemMissing => {}
            }
        }
    }
    let mut index_keys: Vec<String> = index_val_map.keys().cloned().collect();
    index_keys.sort();

    let n_groups = groups.len();

    // Prepare id variables builders
    let mut id_builders: Vec<Box<dyn ArrayBuilder>> = Vec::new();
    for &idx in &id_indices {
        match dataset.variables[idx].var_type {
            VariableType::Numeric => id_builders.push(Box::new(Float64Builder::new())),
            VariableType::String(_) => id_builders.push(Box::new(StringBuilder::new())),
        }
    }

    // Group values map: target variable name -> array of Values
    let mut target_values_map: HashMap<String, Vec<Value>> = HashMap::new();
    let mut target_var_types = HashMap::new();

    // Map each group to its columns
    for (id_vals, positions) in groups.values() {
        // Append ID values
        for (i, val) in id_vals.iter().enumerate() {
            match val {
                Value::Numeric(n) => {
                    let b = id_builders[i].as_any_mut().downcast_mut::<Float64Builder>().unwrap();
                    b.append_value(*n);
                }
                Value::String(s) => {
                    let b = id_builders[i].as_any_mut().downcast_mut::<StringBuilder>().unwrap();
                    b.append_value(s.clone());
                }
                Value::SystemMissing => {
                    match dataset.variables[id_indices[i]].var_type {
                        VariableType::Numeric => {
                            let b = id_builders[i].as_any_mut().downcast_mut::<Float64Builder>().unwrap();
                            b.append_null();
                        }
                        VariableType::String(_) => {
                            let b = id_builders[i].as_any_mut().downcast_mut::<StringBuilder>().unwrap();
                            b.append_null();
                        }
                    }
                }
            }
        }

        // Map index value to row position in this group
        let mut local_index_map = HashMap::new();
        for &(b, r) in positions {
            let val = get_cell_value(dataset, b, r, idx_var_index);
            let k = match val {
                Value::Numeric(n) => format!("{}", n),
                Value::String(s) => s,
                _ => continue,
            };
            local_index_map.insert(k, (b, r));
        }

        // Transpose each grouped variable
        for group_var_name in group_vars {
            let grp_idx = dataset.variable_index(group_var_name).ok_or_else(|| format!("Group var not found: {}", group_var_name))?;
            for idx_key in &index_keys {
                let target_name = format!("{}_{}", group_var_name, idx_key);
                let values_vec = target_values_map.entry(target_name.clone()).or_insert_with(Vec::new);

                if let Some(&(b, r)) = local_index_map.get(idx_key) {
                    let val = get_cell_value(dataset, b, r, grp_idx);
                    target_var_types.insert(target_name, dataset.variables[grp_idx].var_type);
                    values_vec.push(val);
                } else {
                    values_vec.push(Value::SystemMissing);
                }
            }
        }
    }

    let mut out_variables = Vec::new();
    let mut out_columns = Vec::new();

    // 1. Add ID variables
    for (i, &idx) in id_indices.iter().enumerate() {
        let var = &dataset.variables[idx];
        let (dt, array) = match var.var_type {
            VariableType::Numeric => {
                let mut b = id_builders[i].as_any_mut().downcast_mut::<Float64Builder>().unwrap();
                (ArrowDataType::Float64, Arc::new(b.finish()) as ArrayRef)
            }
            VariableType::String(_) => {
                let mut b = id_builders[i].as_any_mut().downcast_mut::<StringBuilder>().unwrap();
                (ArrowDataType::Utf8, Arc::new(b.finish()) as ArrayRef)
            }
        };
        out_variables.push(var.clone());
        out_columns.push(array);
    }

    // 2. Add transposed columns
    for group_var_name in group_vars {
        for idx_key in &index_keys {
            let target_name = format!("{}_{}", group_var_name, idx_key);
            let values = target_values_map.get(&target_name).unwrap();
            let (array, var_type) = array_from_values(values);

            out_variables.push(match var_type {
                VariableType::Numeric => Variable::numeric(&target_name),
                VariableType::String(w) => Variable::string(&target_name, w),
            });
            out_columns.push(array);
        }
    }

    let fields: Vec<arrow::datatypes::FieldRef> = out_variables.iter()
        .map(|v| {
            let dt = match v.var_type {
                VariableType::Numeric => ArrowDataType::Float64,
                VariableType::String(_) => ArrowDataType::Utf8,
            };
            std::sync::Arc::new(Field::new(&v.name, dt, true))
        }).collect();
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, out_columns).map_err(|e| e.to_string())?;

    let mut out = Dataset::new();
    out.batches.push(batch);
    for var in out_variables {
        out.add_variable(var);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;

    fn create_test_dataset() -> Dataset {
        let schema = Arc::new(Schema::new(vec![
            Field::new("ID", ArrowDataType::Float64, true),
            Field::new("GROUP", ArrowDataType::Float64, true),
            Field::new("SCORE", ArrowDataType::Float64, true),
        ]));

        let id_arr = Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)])) as Arc<dyn arrow::array::Array>;
        let grp_arr = Arc::new(Float64Array::from(vec![Some(1.0), Some(1.0), Some(2.0), Some(2.0)])) as Arc<dyn arrow::array::Array>;
        let score_arr = Arc::new(Float64Array::from(vec![Some(80.0), Some(90.0), Some(70.0), None])) as Arc<dyn arrow::array::Array>;

        let batch = RecordBatch::try_new(schema, vec![id_arr, grp_arr, score_arr]).unwrap();

        let mut dataset = Dataset::new();
        dataset.batches.push(batch);
        dataset.add_variable(Variable::numeric("ID"));
        dataset.add_variable(Variable::numeric("GROUP"));
        dataset.add_variable(Variable::numeric("SCORE"));

        dataset
    }

    #[test]
    fn test_select_if_transform() {
        let dataset = create_test_dataset();
        let expr = oxstat_expr::parse("SCORE >= 80").unwrap();
        let filtered = select_if(&dataset, &expr).unwrap();
        assert_eq!(filtered.n_cases(), 2);
        assert_eq!(get_cell_value(&filtered, 0, 0, 0), Value::Numeric(1.0));
        assert_eq!(get_cell_value(&filtered, 0, 1, 0), Value::Numeric(2.0));
    }

    #[test]
    fn test_sort_cases_transform() {
        let dataset = create_test_dataset();
        let keys = vec![SortKey {
            var_name: "SCORE".to_string(),
            order: SortOrder::Descending,
        }];
        let sorted = sort_cases(&dataset, &keys).unwrap();
        assert_eq!(get_cell_value(&sorted, 0, 0, 2), Value::Numeric(90.0));
        assert_eq!(get_cell_value(&sorted, 0, 1, 2), Value::Numeric(80.0));
        assert_eq!(get_cell_value(&sorted, 0, 2, 2), Value::Numeric(70.0));
        assert_eq!(get_cell_value(&sorted, 0, 3, 2), Value::SystemMissing);
    }

    #[test]
    fn test_aggregate_transform() {
        let dataset = create_test_dataset();
        let specs = vec![
            AggSpec {
                target_var: "AVG_SCORE".to_string(),
                function: AggFunction::Mean("SCORE".to_string()),
            },
            AggSpec {
                target_var: "N_CASES".to_string(),
                function: AggFunction::Count,
            },
        ];
        let agg = aggregate(&dataset, &vec!["GROUP".to_string()], &specs).unwrap();
        assert_eq!(agg.n_cases(), 2);

        let row0_grp = get_cell_value(&agg, 0, 0, 0);
        let row1_grp = get_cell_value(&agg, 0, 1, 0);

        if row0_grp == Value::Numeric(1.0) {
            assert_eq!(get_cell_value(&agg, 0, 0, 1), Value::Numeric(85.0));
            assert_eq!(get_cell_value(&agg, 0, 0, 2), Value::Numeric(2.0));
            assert_eq!(row1_grp, Value::Numeric(2.0));
            assert_eq!(get_cell_value(&agg, 0, 1, 1), Value::Numeric(70.0));
            assert_eq!(get_cell_value(&agg, 0, 1, 2), Value::Numeric(2.0));
        } else {
            assert_eq!(row0_grp, Value::Numeric(2.0));
            assert_eq!(get_cell_value(&agg, 0, 0, 1), Value::Numeric(70.0));
            assert_eq!(get_cell_value(&agg, 0, 0, 2), Value::Numeric(2.0));
            assert_eq!(row1_grp, Value::Numeric(1.0));
            assert_eq!(get_cell_value(&agg, 0, 1, 1), Value::Numeric(85.0));
            assert_eq!(get_cell_value(&agg, 0, 1, 2), Value::Numeric(2.0));
        }
    }

    #[test]
    fn test_add_files_transform() {
        let ds1 = create_test_dataset();
        let ds2 = create_test_dataset();
        let appended = add_files(&[&ds1, &ds2]).unwrap();
        assert_eq!(appended.n_cases(), 8);
        assert_eq!(appended.n_variables(), 3);
    }

    #[test]
    fn test_match_files_one_to_one_transform() {
        let ds1 = create_test_dataset();
        let mut ds2 = Dataset::new();
        let schema = Arc::new(Schema::new(vec![
            Field::new("EXTRA", ArrowDataType::Float64, true),
        ]));
        let extra_arr = Arc::new(Float64Array::from(vec![Some(100.0), Some(200.0)])) as Arc<dyn arrow::array::Array>;
        let batch = RecordBatch::try_new(schema, vec![extra_arr]).unwrap();
        ds2.batches.push(batch);
        ds2.add_variable(Variable::numeric("EXTRA"));

        let matched = match_files_one_to_one(&[&ds1, &ds2]).unwrap();
        assert_eq!(matched.n_cases(), 4);
        assert_eq!(matched.n_variables(), 4);
        assert_eq!(get_cell_value(&matched, 0, 0, 3), Value::Numeric(100.0));
        assert_eq!(get_cell_value(&matched, 0, 2, 3), Value::SystemMissing);
    }

    #[test]
    fn test_match_files_by_key_transform() {
        let primary = create_test_dataset();
        let mut lookup = Dataset::new();
        let schema = Arc::new(Schema::new(vec![
            Field::new("GROUP", ArrowDataType::Float64, true),
            Field::new("DESC", ArrowDataType::Utf8, true),
        ]));
        let grp_arr = Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0)])) as Arc<dyn arrow::array::Array>;
        let desc_arr = Arc::new(arrow::array::StringArray::from(vec![Some("Alpha"), Some("Beta")])) as Arc<dyn arrow::array::Array>;
        let batch = RecordBatch::try_new(schema, vec![grp_arr, desc_arr]).unwrap();
        lookup.batches.push(batch);
        lookup.add_variable(Variable::numeric("GROUP"));
        lookup.add_variable(Variable::string("DESC", 10));

        let joined = match_files_by_key(&primary, &lookup, &vec!["GROUP".to_string()]).unwrap();
        assert_eq!(joined.n_cases(), 4);
        assert_eq!(joined.n_variables(), 4);
        assert_eq!(get_cell_value(&joined, 0, 0, 3), Value::String("Alpha".to_string()));
        assert_eq!(get_cell_value(&joined, 0, 2, 3), Value::String("Beta".to_string()));
    }

    #[test]
    fn test_vars_to_cases_transform() {
        let mut dataset = Dataset::new();
        let schema = Arc::new(Schema::new(vec![
            Field::new("ID", ArrowDataType::Float64, true),
            Field::new("INC1", ArrowDataType::Float64, true),
            Field::new("INC2", ArrowDataType::Float64, true),
        ]));
        let id = Arc::new(Float64Array::from(vec![Some(10.0)])) as Arc<dyn arrow::array::Array>;
        let inc1 = Arc::new(Float64Array::from(vec![Some(100.0)])) as Arc<dyn arrow::array::Array>;
        let inc2 = Arc::new(Float64Array::from(vec![Some(200.0)])) as Arc<dyn arrow::array::Array>;
        let batch = RecordBatch::try_new(schema, vec![id, inc1, inc2]).unwrap();
        dataset.batches.push(batch);
        dataset.add_variable(Variable::numeric("ID"));
        dataset.add_variable(Variable::numeric("INC1"));
        dataset.add_variable(Variable::numeric("INC2"));

        let specs = vec![TransposeSpec {
            target_var: "INC".to_string(),
            source_vars: vec!["INC1".to_string(), "INC2".to_string()],
        }];
        let reshaped = vars_to_cases(&dataset, &specs, Some("YEAR"), &vec!["ID".to_string()]).unwrap();
        assert_eq!(reshaped.n_cases(), 2);
        assert_eq!(reshaped.n_variables(), 3);
        assert_eq!(get_cell_value(&reshaped, 0, 0, 1), Value::Numeric(100.0));
        assert_eq!(get_cell_value(&reshaped, 0, 0, 2), Value::Numeric(1.0));
        assert_eq!(get_cell_value(&reshaped, 0, 1, 1), Value::Numeric(200.0));
        assert_eq!(get_cell_value(&reshaped, 0, 1, 2), Value::Numeric(2.0));
    }

    #[test]
    fn test_cases_to_vars_transform() {
        let mut dataset = Dataset::new();
        let schema = Arc::new(Schema::new(vec![
            Field::new("ID", ArrowDataType::Float64, true),
            Field::new("YEAR", ArrowDataType::Float64, true),
            Field::new("INC", ArrowDataType::Float64, true),
        ]));
        let id = Arc::new(Float64Array::from(vec![Some(10.0), Some(10.0)])) as Arc<dyn arrow::array::Array>;
        let year = Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0)])) as Arc<dyn arrow::array::Array>;
        let inc = Arc::new(Float64Array::from(vec![Some(100.0), Some(200.0)])) as Arc<dyn arrow::array::Array>;
        let batch = RecordBatch::try_new(schema, vec![id, year, inc]).unwrap();
        dataset.batches.push(batch);
        dataset.add_variable(Variable::numeric("ID"));
        dataset.add_variable(Variable::numeric("YEAR"));
        dataset.add_variable(Variable::numeric("INC"));

        let reshaped = cases_to_vars(&dataset, &vec!["ID".to_string()], "YEAR", &vec!["INC".to_string()]).unwrap();
        assert_eq!(reshaped.n_cases(), 1);
        assert_eq!(reshaped.n_variables(), 3);
        assert_eq!(get_cell_value(&reshaped, 0, 0, 1), Value::Numeric(100.0));
        assert_eq!(get_cell_value(&reshaped, 0, 0, 2), Value::Numeric(200.0));
    }
}
