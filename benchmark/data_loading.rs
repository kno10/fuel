use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use csv::ReaderBuilder;

#[allow(dead_code)]
pub fn read_numeric_data(path: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    read_numeric_data_with_limit(path, None)
}

pub fn read_numeric_data_with_limit(
    path: &str,
    limit: Option<usize>,
) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    if !Path::new(path).exists() {
        return Err(format!("CSV file not found: {path}").into());
    }

    let first_non_empty = first_non_empty_line(path)?;
    if first_non_empty
        .as_deref()
        .is_some_and(|line| !line.contains(','))
    {
        return read_whitespace_separated(path, limit);
    }

    read_comma_separated(path, limit)
}

fn first_non_empty_line(path: &str) -> Result<Option<String>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            return Ok(Some(trimmed.to_string()));
        }
    }

    Ok(None)
}

fn read_comma_separated(path: &str, limit: Option<usize>) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().has_headers(false).from_path(path)?;

    let mut rows = Vec::new();
    let mut expected_dims: Option<usize> = None;

    for (line_no, record_result) in reader.records().enumerate() {
        if let Some(limit) = limit {
            if rows.len() >= limit {
                break;
            }
        }
        let record = record_result?;
        if record.is_empty() {
            continue;
        }

        let dims = record.len();
        if let Some(expected) = expected_dims {
            if expected != dims {
                return Err(format!(
                    "inconsistent dimensionality at row {}: expected {}, found {}",
                    line_no + 1,
                    expected,
                    dims
                )
                .into());
            }
        } else {
            expected_dims = Some(dims);
        }

        let mut row = Vec::with_capacity(dims);
        for value in &record {
            row.push(parse_value(value, line_no + 1)?);
        }

        rows.push(row);
    }

    if rows.is_empty() {
        return Err("CSV has no data rows".into());
    }

    Ok(rows)
}

fn read_whitespace_separated(
    path: &str,
    limit: Option<usize>,
) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut rows = Vec::new();
    let mut expected_dims: Option<usize> = None;

    for (line_no, line_result) in reader.lines().enumerate() {
        if let Some(limit) = limit {
            if rows.len() >= limit {
                break;
            }
        }
        let line = line_result?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        let dims = parts.len();

        if let Some(expected) = expected_dims {
            if expected != dims {
                return Err(format!(
                    "inconsistent dimensionality at row {}: expected {}, found {}",
                    line_no + 1,
                    expected,
                    dims
                )
                .into());
            }
        } else {
            expected_dims = Some(dims);
        }

        let mut row = Vec::with_capacity(dims);
        for value in parts {
            row.push(parse_value(value, line_no + 1)?);
        }
        rows.push(row);
    }

    if rows.is_empty() {
        return Err("CSV has no data rows".into());
    }

    Ok(rows)
}

fn parse_value(value: &str, line_no: usize) -> Result<f64, Box<dyn Error>> {
    value
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("failed to parse numeric value '{value}' at row {line_no}").into())
}
