use std::path::Path;

use csv::{ReaderBuilder, Trim};
use qsv_sniffer::Sniffer;
use qsv_sniffer::metadata::Header;

/// Input file format hints.
#[derive(Clone, Copy, Debug)]
pub enum FileFormat {
    Csv,
    Arff,
}

/// Options for reading tabular data from disk.
#[derive(Clone, Debug)]
pub struct ReadOptions {
    pub format: Option<FileFormat>,
    pub delimiter: Option<u8>,
    pub header: Option<bool>,
    pub comment_char: Option<u8>,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self { format: None, delimiter: None, header: None, comment_char: Some(b'#') }
    }
}

/// Read a numeric data table from a CSV or ARFF file.
///
/// The result is a row-major vector of numeric records.
pub fn read_numeric_table<P: AsRef<Path>>(
    path: P, options: ReadOptions,
) -> Result<Vec<Vec<f64>>, String> {
    let path = path.as_ref();
    let format = options.format.unwrap_or_else(|| infer_format(path));

    match format {
        FileFormat::Arff => read_arff_table(path),
        FileFormat::Csv => read_csv_table(path, options),
    }
}

fn infer_format(path: &Path) -> FileFormat {
    match path.extension().and_then(|ext| ext.to_str()).map(|s| s.to_ascii_lowercase()) {
        Some(ext) if ext == "arff" => FileFormat::Arff,
        _ => FileFormat::Csv,
    }
}

fn read_arff_table(path: &Path) -> Result<Vec<Vec<f64>>, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read ARFF file {}: {}", path.display(), e))?;

    let rows: Vec<Vec<f64>> = arff::from_str(&contents)
        .map_err(|e| format!("Failed to parse ARFF file {}: {}", path.display(), e))?;

    if rows.is_empty() {
        return Ok(Vec::new());
    }

    let width = rows[0].len();
    if !rows.iter().all(|row| row.len() == width) {
        return Err(format!("ARFF data has inconsistent row lengths in {}", path.display()));
    }

    Ok(rows)
}

fn read_csv_table(path: &Path, options: ReadOptions) -> Result<Vec<Vec<f64>>, String> {
    let comment_char = options.comment_char.unwrap_or(b'#');
    let mut delimiter = options.delimiter;
    let mut header = options.header;

    if delimiter.is_none() || header.is_none() {
        let sniffed = sniff_csv(path, delimiter, header)?;
        delimiter = Some(delimiter.unwrap_or(sniffed.0));
        header = Some(header.unwrap_or(sniffed.1));
    }

    let delimiter = delimiter.unwrap_or(b',');
    let has_header = header.unwrap_or(false);

    if delimiter == b' ' {
        read_whitespace_table(path, has_header, comment_char)
    } else {
        read_csv_with_csv_crate(path, delimiter, has_header, Some(comment_char))
    }
}

fn sniff_csv(
    path: &Path, delimiter: Option<u8>, header: Option<bool>,
) -> Result<(u8, bool), String> {
    let mut sniffer = Sniffer::new();
    if let Some(del) = delimiter {
        sniffer.delimiter(del);
    }
    if let Some(has_header) = header {
        let header = Header { has_header_row: has_header, num_preamble_rows: 0 };
        sniffer.header(&header);
    }

    let metadata = sniffer
        .sniff_path(path)
        .map_err(|e| format!("Failed to sniff CSV file {}: {}", path.display(), e))?;

    let delimiter = metadata.dialect.delimiter;
    let has_header = metadata.dialect.header.has_header_row;
    Ok((delimiter, has_header))
}

fn read_csv_with_csv_crate(
    path: &Path, delimiter: u8, has_header: bool, comment_char: Option<u8>,
) -> Result<Vec<Vec<f64>>, String> {
    let mut reader = ReaderBuilder::new()
        .has_headers(has_header)
        .delimiter(delimiter)
        .trim(Trim::All)
        .comment(comment_char)
        .from_path(path)
        .map_err(|e| format!("Failed to open CSV file {}: {}", path.display(), e))?;

    let mut rows = Vec::new();
    for result in reader.records() {
        let record = result
            .map_err(|e| format!("Failed to parse CSV record in {}: {}", path.display(), e))?;
        let row = record
            .iter()
            .map(|field| {
                field.parse::<f64>().map_err(|e| {
                    format!("Failed to parse '{}' as float in {}: {}", field, path.display(), e)
                })
            })
            .collect::<Result<Vec<f64>, String>>()?;
        rows.push(row);
    }

    if rows.is_empty() {
        return Ok(Vec::new());
    }

    let width = rows[0].len();
    if !rows.iter().all(|row| row.len() == width) {
        return Err(format!("Inconsistent row lengths in {}", path.display()));
    }

    Ok(rows)
}

fn read_whitespace_table(
    path: &Path, has_header: bool, comment_char: u8,
) -> Result<Vec<Vec<f64>>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        format!("Failed to read whitespace-delimited file {}: {}", path.display(), e)
    })?;
    let mut rows = Vec::new();
    let mut first_data_row = true;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.as_bytes()[0] == comment_char {
            continue;
        }
        if first_data_row && has_header {
            first_data_row = false;
            continue;
        }
        first_data_row = false;
        let values: Result<Vec<f64>, _> = line
            .split_whitespace()
            .map(|field| field.trim_matches(|c| c == '"' || c == '\'').parse::<f64>())
            .collect();
        let row = values
            .map_err(|e| format!("Failed to parse numeric value in {}: {}", path.display(), e))?;
        rows.push(row);
    }
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let width = rows[0].len();
    if !rows.iter().all(|row| row.len() == width) {
        return Err(format!("Inconsistent row lengths in {}", path.display()));
    }
    Ok(rows)
}
