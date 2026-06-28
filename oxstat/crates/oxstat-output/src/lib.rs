//! oxstat-output: Structured output model and renderers.
//!
//! Tables, charts, text blocks → plain text, HTML, PDF.

use serde::{Deserialize, Serialize};

/// A single output element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputItem {
    /// Section heading.
    Title(String),
    /// A statistical table.
    Table(Table),
    /// Plain text note or warning.
    Text(String),
    /// Chart reference (rendered separately).
    Chart { chart_type: String, data: String },
}

/// A table with headers and rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    pub title: String,
    pub column_headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub footnotes: Vec<String>,
}

/// Render output items to plain text.
pub fn render_text(items: &[OutputItem]) -> String {
    let mut out = String::new();
    for item in items {
        match item {
            OutputItem::Title(t) => {
                out.push_str(&format!("\n{t}\n"));
                out.push_str(&"=".repeat(t.len()));
                out.push('\n');
            }
            OutputItem::Table(table) => {
                out.push_str(&format!("\n{}\n", table.title));
                // Simple column rendering
                let header = table.column_headers.join("\t");
                out.push_str(&header);
                out.push('\n');
                out.push_str(&"-".repeat(header.len()));
                out.push('\n');
                for row in &table.rows {
                    out.push_str(&row.join("\t"));
                    out.push('\n');
                }
                for fn_ in &table.footnotes {
                    out.push_str(&format!("  * {fn_}\n"));
                }
            }
            OutputItem::Text(t) => {
                out.push_str(t);
                out.push('\n');
            }
            OutputItem::Chart { chart_type, .. } => {
                out.push_str(&format!("[Chart: {chart_type}]\n"));
            }
        }
    }
    out
}
