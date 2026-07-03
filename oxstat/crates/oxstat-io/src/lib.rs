//! oxstat-io: File I/O for OxStat.
//!
//! Read/write SPSS .sav, CSV, Excel, and Parquet files.

pub mod csv_io;
pub mod excel_io;
pub mod sav;

pub use csv_io::{read_csv, write_csv};
pub use excel_io::read_excel;
