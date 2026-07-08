# OxStat

OxStat is an open-source, high-performance statistical analysis suite designed as a modern alternative to SPSS, written in Rust.

It provides a command-line interface (CLI) to load datasets, execute SPSS syntax, perform statistical computations, and render outputs in structured formats.

## Key Features

- **Columnar Data Engine**: Powered by Apache Arrow for zero-copy, highly efficient memory representation and vectorized operations.
- **Robust File I/O**:
  - **SPSS `.sav`**: Custom pure-Rust parser and writer supporting standard headers, variable metadata, discrete/range user-missing values, value labels, and bytecode compression.
  - **CSV**: High-performance CSV import/export.
  - **Excel**: Spreadsheet importing backed by `calamine`.
  - **Parquet**: Import/export preserving SPSS variable metadata serialized as JSON in Arrow Field metadata.
- **Descriptive Statistics**: Vectorized computation of Mean, Standard Deviation, Variance, Skewness, Excess Kurtosis, Sum, Min, Max, and listwise missing case deletion.
- **Expression Engine**: COMPUTE parser and evaluator supporting mathematical operations, standard/statistical built-in functions, logical operators with Kleene three-valued logic, and Gregorian date conversions.
- **Data Transformations**:
  - `SELECT IF / FILTER`: Select cases based on expressions.
  - `SORT CASES`: Multi-key sorting.
  - `AGGREGATE`: Group-by summaries.
  - `MATCH FILES / ADD FILES`: Horizontal join and vertical concatenation.
  - `VARSTOCASES / CASESTOVARS`: Wide-to-long and long-to-wide restructuring.
- **Structured Output**: Structured tables and a plain text renderer.

## Workspace Layout

```text
oxstat/
├── crates/
│   ├── oxstat-core/        # In-memory Dataset, Variable metadata, Value, and MissingValues
│   ├── oxstat-io/          # File import/export (.sav, CSV, Excel, Parquet)
│   ├── oxstat-expr/        # COMPUTE/IF/RECODE expression AST, winnow parser, and evaluator
│   ├── oxstat-transform/   # Data transformations (SELECT, SORT, AGGREGATE, MERGE, RESHAPE)
│   ├── oxstat-stats/       # DESCRIPTIVES statistical procedure
│   ├── oxstat-output/      # Output structures (Table) and plain text renderer
│   ├── oxstat-syntax/      # SPSS syntax parser and dispatcher stub
│   ├── oxstat-chart/       # Chart generation stub
│   └── oxstat-cli/         # Command-line interface binary
```

## Getting Started

### Prerequisites

Make sure you have Rust and Cargo installed (version 1.75 or later).

### Build

To compile the workspace:

```bash
cargo build --manifest-path oxstat/Cargo.toml
```

### Run Tests

To execute all unit and integration tests across the workspace:

```bash
cargo test --manifest-path oxstat/Cargo.toml
```

### Run CLI

To run the command-line interface with a data file:

```bash
cargo run --manifest-path oxstat/Cargo.toml -- --data data.sav
```
