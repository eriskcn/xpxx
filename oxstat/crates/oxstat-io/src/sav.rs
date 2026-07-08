//! SPSS .sav file format reader/writer.
//!
//! Reference: GNU PSPP source + SPSS file format documentation.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Builder, StringBuilder, RecordBatch, ArrayBuilder};
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use oxstat_core::{Dataset, MeasureLevel, MissingValues, Value, Variable, VariableType};

// Byte reading helpers
fn read_i32<R: Read>(reader: &mut R) -> std::io::Result<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f64<R: Read>(reader: &mut R) -> std::io::Result<f64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_string<R: Read>(reader: &mut R, len: usize) -> std::io::Result<String> {
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).trim_end().to_string())
}

// Flat internal representation of a variable record
struct SavVarRecord {
    name: String,
    var_type: VariableType,
    measure: MeasureLevel,
    label: Option<String>,
    value_labels: HashMap<i64, String>,
    missing: MissingValues,
    print_width: u8,
    print_decimals: u8,
    is_continuation: bool,
    original_type_code: i32,
}

struct CompressedReader {
    commands: [u8; 8],
    cmd_idx: usize,
}

impl CompressedReader {
    fn new() -> Self {
        Self {
            commands: [0; 8],
            cmd_idx: 8,
        }
    }

    fn read_next_value<R: Read>(&mut self, reader: &mut R, bias: f64) -> std::io::Result<Value> {
        if self.cmd_idx >= 8 {
            reader.read_exact(&mut self.commands)?;
            self.cmd_idx = 0;
        }
        let cmd = self.commands[self.cmd_idx];
        self.cmd_idx += 1;

        match cmd {
            0 => self.read_next_value(reader, bias),
            252 => Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "SPSS EOF")),
            253 => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                Ok(Value::Numeric(f64::from_le_bytes(buf)))
            }
            254 => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                Ok(Value::String(String::from_utf8_lossy(&buf).to_string()))
            }
            255 => Ok(Value::SystemMissing),
            code => {
                let val = code as f64 - bias;
                Ok(Value::Numeric(val))
            }
        }
    }
}

/// Read SPSS .sav file.
pub fn read_sav<P: AsRef<Path>>(path: P) -> Result<Dataset, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;

    // 1. Read Header (176 bytes)
    let rec_type_header = read_string(&mut file, 4)?;
    if rec_type_header != "$FL2" && rec_type_header != "$FL3" {
        return Err("Invalid SPSS header record type".into());
    }

    let _prod_name = read_string(&mut file, 60)?;
    let layout_code = read_i32(&mut file)?;
    if layout_code != 2 {
        return Err(format!("Invalid layout code: {}", layout_code).into());
    }

    let nominal_case_size = read_i32(&mut file)? as usize;
    let compression = read_i32(&mut file)?;
    let _weight_var = read_i32(&mut file)?;
    let ncases_header = read_i32(&mut file)?;
    let bias = read_f64(&mut file)?;
    let _creation_date = read_string(&mut file, 9)?;
    let _creation_time = read_string(&mut file, 8)?;
    let _file_label = read_string(&mut file, 64)?;
    let mut padding = [0u8; 3];
    file.read_exact(&mut padding)?;

    // 2. Read dictionary records
    let mut records = Vec::new();
    let mut long_names_map: HashMap<String, String> = HashMap::new();

    loop {
        let rec_type = read_i32(&mut file)?;
        if rec_type == 999 {
            // End of dictionary. Skip filler.
            let _filler = read_i32(&mut file)?;
            break;
        }

        match rec_type {
            2 => {
                // Variable record
                let type_code = read_i32(&mut file)?;
                let has_label = read_i32(&mut file)? == 1;
                let n_missing = read_i32(&mut file)?;
                let print_format = read_i32(&mut file)?;
                let write_format = read_i32(&mut file)?;
                let short_name = read_string(&mut file, 8)?;

                let is_continuation = type_code == -1;
                let var_type = if type_code == 0 {
                    VariableType::Numeric
                } else if type_code > 0 {
                    VariableType::String(type_code as u32)
                } else {
                    VariableType::String(8) // continuation dummy type
                };

                let label = if has_label {
                    let len = read_i32(&mut file)? as usize;
                    let padded_len = (len + 3) & !3; // padded to multiple of 4
                    let lbl = read_string(&mut file, len)?;
                    if padded_len > len {
                        let mut junk = vec![0u8; padded_len - len];
                        file.read_exact(&mut junk)?;
                    }
                    Some(lbl)
                } else {
                    None
                };

                let mut missing = MissingValues::default();
                if n_missing != 0 {
                    let count = n_missing.abs() as usize;
                    let mut vals = Vec::with_capacity(count);
                    for _ in 0..count {
                        vals.push(read_f64(&mut file)?);
                    }
                    if n_missing > 0 {
                        missing.discrete = vals;
                    } else if n_missing == -2 {
                        missing.range = Some((vals[0], vals[1]));
                    } else if n_missing == -3 {
                        missing.range = Some((vals[0], vals[1]));
                        missing.discrete = vec![vals[2]];
                    }
                }

                // Decode print/write decimals and width
                // print_format format: decimal places (byte 0), width (byte 1), type (byte 2)
                let print_decimals = (print_format & 0xFF) as u8;
                let print_width = ((print_format >> 8) & 0xFF) as u8;

                records.push(SavVarRecord {
                    name: short_name,
                    var_type,
                    measure: MeasureLevel::Scale,
                    label,
                    value_labels: HashMap::new(),
                    missing,
                    print_width,
                    print_decimals,
                    is_continuation,
                    original_type_code: type_code,
                });
            }
            3 => {
                // Value label
                let count = read_i32(&mut file)? as usize;
                let mut labels = Vec::with_capacity(count);
                for _ in 0..count {
                    // SPSS stores double for numeric, or 8-byte char array for string
                    let mut val_buf = [0u8; 8];
                    file.read_exact(&mut val_buf)?;
                    let value = f64::from_le_bytes(val_buf);

                    let label_len = {
                        let mut len_buf = [0u8; 1];
                        file.read_exact(&mut len_buf)?;
                        len_buf[0] as usize
                    };
                    let padded_len = (label_len + 1 + 7) & !7; // padded to multiple of 8
                    let label_text = read_string(&mut file, label_len)?;
                    if padded_len > label_len + 1 {
                        let mut junk = vec![0u8; padded_len - (label_len + 1)];
                        file.read_exact(&mut junk)?;
                    }
                    labels.push((value, label_text));
                }

                // Read type 4 record (maps value labels to variables)
                let type_4_rec = read_i32(&mut file)?;
                if type_4_rec != 4 {
                    return Err("Expected Record Type 4 following Record Type 3".into());
                }
                let var_count = read_i32(&mut file)? as usize;
                for _ in 0..var_count {
                    let var_idx = (read_i32(&mut file)? - 1) as usize;
                    if var_idx < records.len() {
                        for &(val, ref txt) in &labels {
                            records[var_idx].value_labels.insert(val as i64, txt.clone());
                        }
                    }
                }
            }
            7 => {
                // Extension records
                let subtype = read_i32(&mut file)?;
                let size = read_i32(&mut file)? as usize;
                let count = read_i32(&mut file)? as usize;
                let total_size = size * count;

                match subtype {
                    11 => {
                        // Long variable names map
                        let map_str = read_string(&mut file, total_size)?;
                        // Parse "SHORTNAME=LONGNAME\tSHORTNAME=LONGNAME\t"
                        for entry in map_str.split('\t') {
                            let parts: Vec<&str> = entry.split('=').collect();
                            if parts.len() == 2 {
                                long_names_map.insert(parts[0].trim().to_uppercase(), parts[1].trim().to_string());
                            }
                        }
                    }
                    _ => {
                        // Skip other subtype extension data
                        file.seek(SeekFrom::Current(total_size as i64))?;
                    }
                }
            }
            _ => {
                // Unsupported record type, skip it
                return Err(format!("Unsupported dictionary record type: {}", rec_type).into());
            }
        }
    }

    // Apply long names mapping to records
    for rec in &mut records {
        let key = rec.name.to_uppercase();
        if let Some(long_name) = long_names_map.get(&key) {
            rec.name = long_name.clone();
        }
    }

    // 3. Filter variables (ignoring continuation variables for the output dataset variables list)
    let mut dataset_variables = Vec::new();
    for rec in &records {
        if !rec.is_continuation {
            let var = Variable {
                name: rec.name.clone(),
                var_type: rec.var_type,
                measure: rec.measure,
                label: rec.label.clone(),
                value_labels: rec.value_labels.clone(),
                missing: rec.missing.clone(),
                print_width: rec.print_width,
                print_decimals: rec.print_decimals,
            };
            dataset_variables.push(var);
        }
    }

    // 4. Parse Cases (rows)
    let ncases = if ncases_header >= 0 { ncases_header as usize } else { 0 };

    // Prepare arrays builder
    let mut builders: Vec<Box<dyn ArrayBuilder>> = Vec::new();
    for var in &dataset_variables {
        match var.var_type {
            VariableType::Numeric => builders.push(Box::new(Float64Builder::new())),
            VariableType::String(_) => builders.push(Box::new(StringBuilder::new())),
        }
    }

    let mut compressed_reader = CompressedReader::new();
    let mut current_case = 0;

    loop {
        if ncases_header >= 0 && current_case >= ncases {
            break;
        }

        // Read one row (case)
        let mut row_values = Vec::new();
        let mut err_eof = false;

        let mut primary_idx = 0;
        let mut current_string = String::new();
        let mut in_string = false;
        let mut expected_continuation_blocks = 0;

        for rec in &records {
            if rec.is_continuation {
                if in_string {
                    // Read continuation string block (8 bytes)
                    let val = if compression == 1 {
                        match compressed_reader.read_next_value(&mut file, bias) {
                            Ok(v) => v,
                            Err(_) => {
                                err_eof = true;
                                break;
                            }
                        }
                    } else {
                        // Uncompressed
                        let mut buf = [0u8; 8];
                        if file.read_exact(&mut buf).is_err() {
                            err_eof = true;
                            break;
                        }
                        Value::String(String::from_utf8_lossy(&buf).to_string())
                    };

                    if let Value::String(s) = val {
                        current_string.push_str(&s);
                    }
                    expected_continuation_blocks -= 1;
                    if expected_continuation_blocks == 0 {
                        // Trim and collect string value
                        row_values.push(Value::String(current_string.trim_end().to_string()));
                        in_string = false;
                        current_string.clear();
                    }
                }
                continue;
            }

            // Primary record
            let val = if compression == 1 {
                match compressed_reader.read_next_value(&mut file, bias) {
                    Ok(v) => v,
                    Err(_) => {
                        err_eof = true;
                        break;
                    }
                }
            } else {
                let mut buf = [0u8; 8];
                if file.read_exact(&mut buf).is_err() {
                    err_eof = true;
                    break;
                }
                match rec.var_type {
                    VariableType::Numeric => {
                        let d = f64::from_le_bytes(buf);
                        // Check for standard missing flag (SPSS SYSMIS)
                        if d.is_nan() {
                            Value::SystemMissing
                        } else {
                            Value::Numeric(d)
                        }
                    }
                    VariableType::String(_) => {
                        Value::String(String::from_utf8_lossy(&buf).to_string())
                    }
                }
            };

            match rec.var_type {
                VariableType::Numeric => {
                    row_values.push(val);
                }
                VariableType::String(width) => {
                    let total_blocks = (width as usize + 7) / 8;
                    if total_blocks > 1 {
                        in_string = true;
                        expected_continuation_blocks = total_blocks - 1;
                        if let Value::String(s) = val {
                            current_string = s;
                        }
                    } else {
                        // Fits in single 8-byte block
                        if let Value::String(s) = val {
                            row_values.push(Value::String(s.trim_end().to_string()));
                        } else {
                            row_values.push(Value::SystemMissing);
                        }
                    }
                }
            }
        }

        if err_eof {
            break;
        }

        // Append row values to builders
        for (i, val) in row_values.into_iter().enumerate() {
            match val {
                Value::Numeric(n) => {
                    let b = builders[i].as_any_mut().downcast_mut::<Float64Builder>().unwrap();
                    b.append_value(n);
                }
                Value::String(s) => {
                    let b = builders[i].as_any_mut().downcast_mut::<StringBuilder>().unwrap();
                    b.append_value(s);
                }
                Value::SystemMissing => {
                    // Match type
                    match dataset_variables[i].var_type {
                        VariableType::Numeric => {
                            let b = builders[i].as_any_mut().downcast_mut::<Float64Builder>().unwrap();
                            b.append_null();
                        }
                        VariableType::String(_) => {
                            let b = builders[i].as_any_mut().downcast_mut::<StringBuilder>().unwrap();
                            b.append_null();
                        }
                    }
                }
            }
        }
        current_case += 1;
    }

    // Construct record batch
    let mut fields = Vec::with_capacity(dataset_variables.len());
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(dataset_variables.len());

    for (i, var) in dataset_variables.iter().enumerate() {
        let (dt, array) = match var.var_type {
            VariableType::Numeric => {
                let mut b = builders[i].as_any_mut().downcast_mut::<Float64Builder>().unwrap();
                (ArrowDataType::Float64, Arc::new(b.finish()) as ArrayRef)
            }
            VariableType::String(_) => {
                let mut b = builders[i].as_any_mut().downcast_mut::<StringBuilder>().unwrap();
                (ArrowDataType::Utf8, Arc::new(b.finish()) as ArrayRef)
            }
        };
        fields.push(Field::new(&var.name, dt, true));
        columns.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, columns)?;

    let mut dataset = Dataset::new();
    dataset.batches.push(batch);
    for var in dataset_variables {
        dataset.add_variable(var);
    }

    Ok(dataset)
}

struct CompressedWriter<W> {
    inner: W,
    commands: [u8; 8],
    data_blocks: Vec<Vec<u8>>,
    cmd_idx: usize,
}

impl<W: Write> CompressedWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            commands: [0; 8],
            data_blocks: Vec::new(),
            cmd_idx: 0,
        }
    }

    fn write_value(&mut self, val: &Value) -> std::io::Result<()> {
        match val {
            Value::SystemMissing => {
                self.commands[self.cmd_idx] = 255;
                self.cmd_idx += 1;
            }
            Value::Numeric(n) => {
                let code = (n + 100.0).round();
                if code >= 1.0 && code <= 251.0 && (n - (code - 100.0)).abs() < 1e-9 {
                    self.commands[self.cmd_idx] = code as u8;
                    self.cmd_idx += 1;
                } else {
                    self.commands[self.cmd_idx] = 253;
                    self.cmd_idx += 1;
                    self.data_blocks.push(n.to_le_bytes().to_vec());
                }
            }
            Value::String(s) => {
                let mut bytes = [b' '; 8];
                let s_bytes = s.as_bytes();
                let len = s_bytes.len().min(8);
                bytes[..len].copy_from_slice(&s_bytes[..len]);

                self.commands[self.cmd_idx] = 254;
                self.cmd_idx += 1;
                self.data_blocks.push(bytes.to_vec());
            }
        }

        if self.cmd_idx >= 8 {
            self.flush_block()?;
        }
        Ok(())
    }

    fn flush_block(&mut self) -> std::io::Result<()> {
        if self.cmd_idx == 0 {
            return Ok(());
        }
        while self.cmd_idx < 8 {
            self.commands[self.cmd_idx] = 0;
            self.cmd_idx += 1;
        }
        self.inner.write_all(&self.commands)?;
        for block in &self.data_blocks {
            self.inner.write_all(block)?;
        }
        self.commands = [0; 8];
        self.data_blocks.clear();
        self.cmd_idx = 0;
        Ok(())
    }

    fn close(&mut self) -> std::io::Result<()> {
        if self.cmd_idx > 0 {
            self.commands[self.cmd_idx] = 252; // EOF
            self.cmd_idx += 1;
            self.flush_block()?;
        }
        Ok(())
    }
}

/// Write Dataset to SPSS .sav file.
pub fn write_sav<P: AsRef<Path>>(dataset: &Dataset, path: P) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);

    // 1. Determine short names & continuation mappings
    let mut flat_vars = Vec::new();
    let mut long_names_entries = Vec::new();

    for (idx, var) in dataset.variables.iter().enumerate() {
        let short_name = if var.name.len() <= 8 && var.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            var.name.to_uppercase()
        } else {
            // Generate unique short name
            let generated = format!("VAR{:05}", idx + 1);
            long_names_entries.push(format!("{}={}", generated, var.name));
            generated
        };

        match var.var_type {
            VariableType::Numeric => {
                flat_vars.push((short_name, var.var_type, var.label.clone(), var.missing.clone(), var.print_width, var.print_decimals, false));
            }
            VariableType::String(width) => {
                let blocks = (width as usize + 7) / 8;
                flat_vars.push((short_name.clone(), var.var_type, var.label.clone(), var.missing.clone(), var.print_width, var.print_decimals, false));
                for _ in 1..blocks {
                    flat_vars.push(("".to_string(), VariableType::String(8), None, MissingValues::default(), 0, 0, true));
                }
            }
        }
    }

    let nominal_case_size = flat_vars.len();

    // 2. Write Header (176 bytes)
    writer.write_all(b"$FL2")?;

    let mut prod_name = [b' '; 60];
    let prod_bytes = b"OxStat";
    prod_name[..prod_bytes.len()].copy_from_slice(prod_bytes);
    writer.write_all(&prod_name)?;

    writer.write_all(&2i32.to_le_bytes())?; // layout_code
    writer.write_all(&(nominal_case_size as i32).to_le_bytes())?;
    writer.write_all(&1i32.to_le_bytes())?; // compression = 1
    writer.write_all(&0i32.to_le_bytes())?; // weight_var = 0

    let ncases = dataset.n_cases() as i32;
    writer.write_all(&ncases.to_le_bytes())?;

    writer.write_all(&100.0f64.to_le_bytes())?; // bias

    writer.write_all(b"04 Jul 26")?; // creation_date
    writer.write_all(b"12:00:00")?; // creation_time

    let file_label = [b' '; 64];
    writer.write_all(&file_label)?;

    writer.write_all(&[0u8; 3])?; // padding

    // 3. Write Variable Records (Type 2)
    for &(ref sname, ref vtype, ref label, ref missing, pwidth, pdecimals, is_cont) in &flat_vars {
        writer.write_all(&2i32.to_le_bytes())?; // rec_type = 2

        let type_code = if is_cont {
            -1
        } else {
            match *vtype {
                VariableType::Numeric => 0,
                VariableType::String(w) => w as i32,
            }
        };
        writer.write_all(&type_code.to_le_bytes())?;

        let has_label = label.is_some() as i32;
        writer.write_all(&has_label.to_le_bytes())?;

        // Missing values definitions
        let n_missing = if is_cont {
            0
        } else if missing.range.is_some() {
            if !missing.discrete.is_empty() { -3 } else { -2 }
        } else {
            missing.discrete.len().min(3) as i32
        };
        writer.write_all(&n_missing.to_le_bytes())?;

        // formats (print & write)
        // print_format = decimal places (byte 0), width (byte 1), type (byte 2)
        // For numbers, print format is F8.2 (type 5). For strings, it is A8 (type 1).
        let fmt_type = match *vtype {
            VariableType::Numeric => 5,
            VariableType::String(_) => 1,
        };
        let packed_fmt = (pdecimals as i32) | ((pwidth as i32) << 8) | (fmt_type << 16);
        writer.write_all(&packed_fmt.to_le_bytes())?; // print_format
        writer.write_all(&packed_fmt.to_le_bytes())?; // write_format

        let mut short_buf = [b' '; 8];
        let name_bytes = sname.as_bytes();
        let name_len = name_bytes.len().min(8);
        short_buf[..name_len].copy_from_slice(&name_bytes[..name_len]);
        writer.write_all(&short_buf)?;

        if let Some(ref lbl) = *label {
            let bytes = lbl.as_bytes();
            let len = bytes.len();
            writer.write_all(&(len as i32).to_le_bytes())?;
            writer.write_all(bytes)?;
            let padded_len = (len + 3) & !3;
            if padded_len > len {
                writer.write_all(&vec![0u8; padded_len - len])?;
            }
        }

        if n_missing != 0 {
            if let Some((lo, hi)) = missing.range {
                writer.write_all(&lo.to_le_bytes())?;
                writer.write_all(&hi.to_le_bytes())?;
                if n_missing == -3 {
                    writer.write_all(&missing.discrete[0].to_le_bytes())?;
                }
            } else {
                for &val in &missing.discrete {
                    writer.write_all(&val.to_le_bytes())?;
                }
            }
        }
    }

    // 4. Write Value Labels Records (Type 3 & 4)
    let mut flat_idx = 0;
    for (i, var) in dataset.variables.iter().enumerate() {
        if !var.value_labels.is_empty() {
            writer.write_all(&3i32.to_le_bytes())?; // rec_type = 3

            let label_count = var.value_labels.len() as i32;
            writer.write_all(&label_count.to_le_bytes())?;

            for (&val, lbl) in &var.value_labels {
                writer.write_all(&(val as f64).to_le_bytes())?;
                let bytes = lbl.as_bytes();
                let len = bytes.len().min(255);
                writer.write_all(&[len as u8])?;
                writer.write_all(&bytes[..len])?;
                let padded_len = (len + 1 + 7) & !7;
                if padded_len > len + 1 {
                    writer.write_all(&vec![0u8; padded_len - (len + 1)])?;
                }
            }

            // Follow with Type 4 record
            writer.write_all(&4i32.to_le_bytes())?; // rec_type = 4
            writer.write_all(&1i32.to_le_bytes())?; // var_count = 1
            writer.write_all(&((flat_idx + 1) as i32).to_le_bytes())?; // 1-based variable index
        }

        // Skip continuation variables for value label index lookup
        flat_idx += match var.var_type {
            VariableType::Numeric => 1,
            VariableType::String(w) => (w as usize + 7) / 8,
        };
    }

    // 5. Write Long Names Extension Record (Type 7 Subtype 11)
    if !long_names_entries.is_empty() {
        writer.write_all(&7i32.to_le_bytes())?; // rec_type = 7
        writer.write_all(&11i32.to_le_bytes())?; // subtype = 11
        writer.write_all(&1i32.to_le_bytes())?; // size = 1

        let map_str = long_names_entries.join("\t") + "\t";
        let bytes = map_str.as_bytes();
        let count = bytes.len() as i32;
        writer.write_all(&count.to_le_bytes())?;
        writer.write_all(bytes)?;
    }

    // 6. Write End of Dictionary (Type 999)
    writer.write_all(&999i32.to_le_bytes())?;
    writer.write_all(&0i32.to_le_bytes())?; // filler = 0

    // 7. Write Case Data (Compressed)
    let mut comp_writer = CompressedWriter::new(&mut writer);
    let n_cases = dataset.n_cases();

    for row_idx in 0..n_cases {
        let mut batch_idx = 0;
        let mut row_offset = row_idx;
        while row_offset >= dataset.batches[batch_idx].num_rows() {
            row_offset -= dataset.batches[batch_idx].num_rows();
            batch_idx += 1;
        }

        let batch = &dataset.batches[batch_idx];

        for (i, var) in dataset.variables.iter().enumerate() {
            let col = batch.column(i);

            match var.var_type {
                VariableType::Numeric => {
                    let val = if col.is_null(row_offset) {
                        Value::SystemMissing
                    } else {
                        let array = col.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap();
                        Value::Numeric(array.value(row_offset))
                    };
                    comp_writer.write_value(&val)?;
                }
                VariableType::String(width) => {
                    let val_str = if col.is_null(row_offset) {
                        "".to_string()
                    } else {
                        let array = col.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
                        array.value(row_offset).to_string()
                    };

                    // Pad string to width
                    let mut s = val_str;
                    if s.len() < width as usize {
                        s.push_str(&" ".repeat(width as usize - s.len()));
                    } else {
                        s.truncate(width as usize);
                    }

                    // Slice into 8-byte blocks
                    let blocks = (width as usize + 7) / 8;
                    for b in 0..blocks {
                        let start = b * 8;
                        let end = (start + 8).min(s.len());
                        let mut chunk = s[start..end].to_string();
                        if chunk.len() < 8 {
                            chunk.push_str(&" ".repeat(8 - chunk.len()));
                        }
                        comp_writer.write_value(&Value::String(chunk))?;
                    }
                }
            }
        }
    }

    comp_writer.close()?;
    writer.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;

    #[test]
    fn test_sav_read_write() {
        let temp_dir = std::env::temp_dir();
        let sav_path = temp_dir.join("test_data.sav");

        let schema = Arc::new(Schema::new(vec![
            Field::new("score", ArrowDataType::Float64, true),
        ]));
        let score_array = Arc::new(Float64Array::from(vec![Some(85.0), Some(92.0)])) as Arc<dyn arrow::array::Array>;
        let batch = RecordBatch::try_new(schema, vec![score_array]).unwrap();

        let mut dataset = Dataset::new();
        dataset.batches.push(batch);

        let mut var = Variable::numeric("score");
        var.label = Some("Math Score".to_string());
        dataset.add_variable(var);

        // Write to SAV
        write_sav(&dataset, &sav_path).unwrap();

        // Read from SAV
        let loaded = read_sav(&sav_path).unwrap();
        assert_eq!(loaded.n_variables(), 1);
        assert_eq!(loaded.n_cases(), 2);

        let var_loaded = loaded.variable("score").unwrap();
        assert_eq!(var_loaded.label, Some("Math Score".to_string()));

        let _ = std::fs::remove_file(sav_path);
    }
}
