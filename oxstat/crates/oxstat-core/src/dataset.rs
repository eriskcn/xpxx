use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::{Field, Schema};

use crate::variable::Variable;

/// A statistical dataset: columnar data + variable metadata.
pub struct Dataset {
    /// Variable definitions, ordered by position.
    pub variables: Vec<Variable>,
    /// Variable name → index lookup.
    name_index: HashMap<String, usize>,
    /// Row data stored as Arrow RecordBatches.
    pub batches: Vec<RecordBatch>,
}

impl Dataset {
    /// Create an empty dataset.
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            name_index: HashMap::new(),
            batches: Vec::new(),
        }
    }

    /// Number of variables (columns).
    pub fn n_variables(&self) -> usize {
        self.variables.len()
    }

    /// Number of cases (rows) across all batches.
    pub fn n_cases(&self) -> usize {
        self.batches.iter().map(|b| b.num_rows()).sum()
    }

    /// Look up a variable by name (case-insensitive).
    pub fn variable(&self, name: &str) -> Option<&Variable> {
        let key = name.to_uppercase();
        self.name_index.get(&key).map(|&i| &self.variables[i])
    }

    /// Look up a variable's index by name (case-insensitive).
    pub fn variable_index(&self, name: &str) -> Option<usize> {
        let key = name.to_uppercase();
        self.name_index.get(&key).copied()
    }

    /// Add a variable definition.
    pub fn add_variable(&mut self, var: Variable) {
        let key = var.name.to_uppercase();
        let idx = self.variables.len();
        self.name_index.insert(key, idx);
        self.variables.push(var);
    }

    /// Insert or replace a column in the dataset.
    pub fn insert_or_replace_column(
        &mut self,
        name: &str,
        var: Variable,
        arrays: Vec<ArrayRef>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let key = name.to_uppercase();
        if let Some(&var_idx) = self.name_index.get(&key) {
            self.variables[var_idx] = var;
            let mut new_batches = Vec::with_capacity(self.batches.len());
            for (i, batch) in self.batches.iter().enumerate() {
                let schema = batch.schema();
                let mut fields = schema.fields().to_vec();
                let data_type = arrays[i].data_type().clone();
                fields[var_idx] = Arc::new(Field::new(name, data_type, true));
                let new_schema = Arc::new(Schema::new(fields));
                let mut columns = batch.columns().to_vec();
                columns[var_idx] = arrays[i].clone();
                let new_batch = RecordBatch::try_new(new_schema, columns)?;
                new_batches.push(new_batch);
            }
            self.batches = new_batches;
        } else {
            let var_idx = self.variables.len();
            self.name_index.insert(key, var_idx);
            self.variables.push(var);
            let mut new_batches = Vec::with_capacity(self.batches.len());
            for (i, batch) in self.batches.iter().enumerate() {
                let schema = batch.schema();
                let mut fields = schema.fields().to_vec();
                let data_type = arrays[i].data_type().clone();
                fields.push(Arc::new(Field::new(name, data_type, true)));
                let new_schema = Arc::new(Schema::new(fields));
                let mut columns = batch.columns().to_vec();
                columns.push(arrays[i].clone());
                let new_batch = RecordBatch::try_new(new_schema, columns)?;
                new_batches.push(new_batch);
            }
            self.batches = new_batches;
        }
        Ok(())
    }
}

impl Default for Dataset {
    fn default() -> Self {
        Self::new()
    }
}
