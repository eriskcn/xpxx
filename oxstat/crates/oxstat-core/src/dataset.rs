use std::collections::HashMap;

use arrow::array::RecordBatch;

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
}

impl Default for Dataset {
    fn default() -> Self {
        Self::new()
    }
}
