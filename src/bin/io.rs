use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::path::Path;

use csv::{ReaderBuilder, Trim};
use flate2::read::GzDecoder;
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use qsv_sniffer::Sniffer;
use qsv_sniffer::metadata::Header;

/// Input file format hints.
#[derive(Clone, Copy, Debug)]
pub enum FileFormat {
    Csv,
    Npy,
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

/// Read a numeric data table from disk.
///
/// Supported formats: CSV, NPY. Gzipped CSV and NPY files are supported when the file name
/// ends with `.gz`.
pub fn read_numeric_table<P: AsRef<Path>>(
    path: P, options: ReadOptions,
) -> Result<Vec<Vec<f64>>, String> {
    let path = path.as_ref();
    match options.format.unwrap_or_else(|| infer_format(path)) {
        FileFormat::Csv => read_csv_table(path, options),
        FileFormat::Npy => read_npy_table(path),
    }
}

fn infer_format(path: &Path) -> FileFormat {
    match path_extension(path) {
        Some(ext) if ext.eq_ignore_ascii_case("npy") => FileFormat::Npy,
        _ => FileFormat::Csv,
    }
}

fn path_extension(path: &Path) -> Option<&str> {
    if is_gzip_path(path) {
        path.file_stem()
            .and_then(OsStr::to_str)
            .and_then(|stem| Path::new(stem).extension())
            .and_then(OsStr::to_str)
    } else {
        path.extension().and_then(OsStr::to_str)
    }
}

fn read_npy_table(path: &Path) -> Result<Vec<Vec<f64>>, String> {
    let data_reader: Box<dyn Read> = if is_gzip_path(path) {
        let data = open_input_bytes(path)?;
        Box::new(Cursor::new(data))
    } else {
        let file = File::open(path)
            .map_err(|e| format!("Failed to open NPY file {}: {}", path.display(), e))?;
        Box::new(BufReader::new(file))
    };

    let array = Array2::<f64>::read_npy(data_reader)
        .map_err(|e| format!("Failed to read NPY file {}: {}", path.display(), e))?;

    Ok(array.rows().into_iter().map(|row| row.to_vec()).collect())
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

    let metadata = if is_gzip_path(path) {
        let data = open_input_bytes(path)?;
        let mut cursor = Cursor::new(data);
        sniffer
            .sniff_reader(&mut cursor)
            .map_err(|e| format!("Failed to sniff CSV file {}: {}", path.display(), e))?
    } else {
        sniffer
            .sniff_path(path)
            .map_err(|e| format!("Failed to sniff CSV file {}: {}", path.display(), e))?
    };

    let delimiter = metadata.dialect.delimiter;
    let has_header = metadata.dialect.header.has_header_row;
    Ok((delimiter, has_header))
}

fn is_gzip_path(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .map(|ext| ext.eq_ignore_ascii_case("gz"))
        .unwrap_or(false)
}

fn open_input_bytes(path: &Path) -> Result<Vec<u8>, String> {
    let file =
        File::open(path).map_err(|e| format!("Failed to open file {}: {}", path.display(), e))?;
    if is_gzip_path(path) {
        let mut decoder = GzDecoder::new(file);
        let mut buffer = Vec::new();
        decoder
            .read_to_end(&mut buffer)
            .map_err(|e| format!("Failed to decompress gzip file {}: {}", path.display(), e))?;
        Ok(buffer)
    } else {
        let mut reader = BufReader::new(file);
        let mut buffer = Vec::new();
        reader
            .read_to_end(&mut buffer)
            .map_err(|e| format!("Failed to read file {}: {}", path.display(), e))?;
        Ok(buffer)
    }
}

fn read_csv_with_csv_crate(
    path: &Path, delimiter: u8, has_header: bool, comment_char: Option<u8>,
) -> Result<Vec<Vec<f64>>, String> {
    let boxed_reader: Box<dyn Read> = if is_gzip_path(path) {
        let data = open_input_bytes(path)?;
        Box::new(Cursor::new(data))
    } else {
        let file = File::open(path)
            .map_err(|e| format!("Failed to open CSV file {}: {}", path.display(), e))?;
        Box::new(BufReader::new(file))
    };

    let mut reader = ReaderBuilder::new()
        .has_headers(has_header)
        .delimiter(delimiter)
        .trim(Trim::All)
        .comment(comment_char)
        .from_reader(boxed_reader);

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
    let content = if is_gzip_path(path) {
        let bytes = open_input_bytes(path)?;
        String::from_utf8(bytes)
            .map_err(|e| format!("Failed to decode UTF-8 from {}: {}", path.display(), e))?
    } else {
        std::fs::read_to_string(path).map_err(|e| {
            format!("Failed to read whitespace-delimited file {}: {}", path.display(), e)
        })?
    };
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
