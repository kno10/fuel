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
    path: &str, limit: Option<usize>,
) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    if !Path::new(path).exists() {
        return Err(format!("CSV file not found: {path}").into());
    }

    let first_non_empty = first_non_empty_line(path)?;
    if first_non_empty.as_deref().is_some_and(|line| !line.contains(',')) {
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
        if let Some(limit) = limit
            && rows.len() >= limit
        {
            break;
        }
        let record = record_result?;
        if record.is_empty() {
            continue;
        }

        // Detect and skip a header row if it appears to contain non-numeric data.
        // The problem statement guarantees that a header row begins with a non-number
        // character, so we can simply try parsing the first row and skip it if
        // any field fails to parse. This mirrors the logic used in
        // `benchmark/kd_vs_vp.rs`.
        if rows.is_empty() && record.iter().any(|v| v.trim().parse::<f64>().is_err()) {
            // skip header and continue to next record
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
    path: &str, limit: Option<usize>,
) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut rows = Vec::new();
    let mut expected_dims: Option<usize> = None;

    for (line_no, line_result) in reader.lines().enumerate() {
        if let Some(limit) = limit
            && rows.len() >= limit
        {
            break;
        }
        let line = line_result?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        let dims = parts.len();

        // skip header row for whitespace-separated data as well
        if rows.is_empty() && parts.iter().any(|v| v.trim().parse::<f64>().is_err()) {
            continue;
        }

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

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    // helper to get dataset path
    fn data_path(name: &str) -> String {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("data/hierarchical");
        path.push(name);
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn skip_header_csv() {
        let path = data_path("nested_clusters.csv");
        let rows = read_numeric_data(&path).unwrap();
        // first row is header, remaining rows start with numeric
        assert!(!rows.is_empty());
        // sanity check first data point
        assert_eq!(rows[0].len(), 4); // includes label column
        assert!(rows[0][0].is_finite());
    }

    #[test]
    fn error_on_all_header() {
        // construct a temporary file with just a header to ensure error path still works
        let mut tmp = std::fs::File::create("test_header_only.csv").unwrap();
        writeln!(tmp, "a,b,c").unwrap();
        let err = read_numeric_data("test_header_only.csv").unwrap_err();
        assert!(err.to_string().contains("no data rows"));
        std::fs::remove_file("test_header_only.csv").unwrap();
    }
}
