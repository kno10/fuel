use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::{env, fs};

use flate2::write::GzEncoder;
use flate2::Compression;

mod io;

use fuel::intrinsicdimensionality::{ABID, AggregatedHillID};
use fuel::kernel::polynomial::PolynomialKernel;
use fuel::outlier::cof::connectivity_outlier_factor;
use fuel::outlier::common::OutlierResult;
use fuel::outlier::fast_abod::fast_angle_based_outlier_detection;
use fuel::outlier::inflo::influence_outlier;
use fuel::outlier::isos::intrinsic_stochastic_outlier_selection;
use fuel::outlier::kdeos::kdeos;
use fuel::outlier::kernel::KernelDensityFunction;
use fuel::outlier::knn::k_nearest_neighbors_outlier;
use fuel::outlier::knndd::k_nearest_neighbors_distance_deviation;
use fuel::outlier::knnsos::k_nearest_neighbors_sos;
use fuel::outlier::ldf::local_density_factor;
use fuel::outlier::ldof::local_density_outlier_factor;
use fuel::outlier::lid::local_intrinsic_dimensionality;
use fuel::outlier::local_isolation_coefficient::local_isolation_coefficient;
use fuel::outlier::lof::local_outlier_factor;
use fuel::outlier::odin::outlier_detection_independence_neighbor;
use fuel::outlier::simple_kernel_density_lof::simple_kernel_density_lof;
use fuel::outlier::simplified_lof::simplified_lof;
use fuel::outlier::variance_of_volume::variance_of_volume;
use fuel::outlier::weighted_knn::weighted_knn;
use fuel::outlier::{intrinsic_dimensionality_outlier_score, local_outlier_probabilities};
use fuel::search::proxy::ProxyKnnSearcher;
use fuel::{Data, DistanceData, KnnSearch, RangeSearch, VectorData};
use io::{FileFormat, ReadOptions, read_numeric_table};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Options for batch computation of kNN outlier scores.
#[derive(Clone, Debug)]
pub struct ComputeKnnOutlierScoresOptions {
    /// Names or substrings of methods to disable.
    pub disabled: Vec<String>,
    /// Maximum k for O(k^2) algorithms.
    pub ksquare_max: usize,
    /// Per-method time limit.
    pub time_limit: Option<Duration>,
}

impl Default for ComputeKnnOutlierScoresOptions {
    fn default() -> Self { Self { disabled: Vec::new(), ksquare_max: 1000, time_limit: None } }
}

impl ComputeKnnOutlierScoresOptions {
    fn is_disabled(&self, method: &str) -> bool {
        self.disabled.iter().any(|pattern| method.contains(pattern))
    }
}

/// A single outlier score vector produced for a given algorithm and k value.
#[derive(Clone, Debug)]
pub struct BatchOutlierResult {
    /// The method prefix, e.g. `KNN`, `LOF`, `KNNW`.
    pub method: String,
    /// The k parameter used to compute the score.
    pub k: usize,
    /// A combined label for the result, e.g. `KNN-10`.
    pub label: String,
    /// The outlier score vector.
    pub result: OutlierResult<f64>,
}

fn write_outlier_result_line<W>(
    writer: &mut W, prefix: &str, k: usize, result: &OutlierResult<f64>,
    scaling: Option<&dyn Fn(f64) -> f64>,
) -> std::io::Result<()>
where
    W: Write,
{
    write!(writer, "{}-{}", prefix, k)?;
    for score in result.scores.iter() {
        let value = scaling.map_or(*score, |scale| scale(*score));
        write!(writer, " {}", format_value(value))?;
    }
    writeln!(writer)?;
    Ok(())
}

fn run_for_each_k<W, R>(
    writer: &mut W, options: &ComputeKnnOutlierScoresOptions, ks: &[usize], prefix: &'static str,
    min_k: usize, max_k: usize, runner: R,
) -> std::io::Result<()>
where
    W: Write,
    R: Fn(usize) -> OutlierResult<f64>,
{
    if options.is_disabled(prefix) {
        return Ok(());
    }

    let allowed_ks: Vec<usize> = ks.iter().copied().filter(|&k| k >= min_k && k <= max_k).collect();

    for k in allowed_ks {
        let start = Instant::now();
        let result = runner(k);
        let elapsed = start.elapsed();

        write_outlier_result_line(writer, prefix, k, &result, None)?;

        eprintln!("{} k={} runtime={:.3}s", prefix, k, elapsed.as_secs_f64());

        if let Some(limit) = options.time_limit && elapsed > limit {
            break;
        }
    }

    Ok(())
}

fn build_sorted_ks(ks: impl IntoIterator<Item = usize>) -> Vec<usize> {
    let mut ks: Vec<usize> = ks.into_iter().filter(|&k| k > 0).collect();
    ks.sort_unstable();
    ks.dedup();
    ks
}

#[derive(Debug, Clone)]
struct Config {
    input: PathBuf,
    output: Option<PathBuf>,
    ks: Vec<usize>,
    delimiter: Option<u8>,
    format: Option<FileFormat>,
    header: Option<bool>,
    disable: Vec<String>,
    ksquare_max: usize,
    time_limit: Option<Duration>,
}

fn parse_delimiter(value: &str) -> Result<Option<u8>, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(None),
        "," | "comma" => Ok(Some(b',')),
        "\t" | "tab" => Ok(Some(b'\t')),
        ";" | "semicolon" => Ok(Some(b';')),
        "space" | "whitespace" | "ws" => Ok(Some(b' ')),
        other if other.len() == 1 => Ok(Some(other.as_bytes()[0])),
        other => Err(format!("Unsupported delimiter: {}", other)),
    }
}

fn parse_format(value: &str) -> Result<Option<FileFormat>, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "csv" => Ok(Some(FileFormat::Csv)),
        "npy" => Ok(Some(FileFormat::Npy)),
        "auto" => Ok(None),
        other => Err(format!("Unsupported format: {}", other)),
    }
}

fn parse_ks(value: &str) -> Result<Vec<usize>, String> {
    fn parse_single(token: &str) -> Result<Vec<usize>, String> {
        let token = token.trim();
        if token.is_empty() {
            return Ok(Vec::new());
        }
        if token.contains("..") {
            let parts: Vec<_> = token.split("..").collect();
            if parts.len() != 2 {
                return Err(format!("Invalid k range: {}", token));
            }
            let start: usize = parts[0]
                .trim()
                .parse()
                .map_err(|e| format!("Invalid k value: {} ({})", parts[0].trim(), e))?;
            let end: usize = parts[1]
                .trim()
                .parse()
                .map_err(|e| format!("Invalid k value: {} ({})", parts[1].trim(), e))?;
            if start == 0 || end == 0 {
                return Err("k values must be positive".to_string());
            }
            if start > end {
                return Err(format!("Invalid k range: {} (start > end)", token));
            }
            return Ok((start..=end).collect());
        }
        let separators = ['-', ':'];
        for sep in separators {
            if token.contains(sep) {
                let parts: Vec<_> = token.split(sep).collect();
                if parts.len() != 2 {
                    return Err(format!("Invalid k range: {}", token));
                }
                let start: usize = parts[0]
                    .trim()
                    .parse()
                    .map_err(|e| format!("Invalid k value: {} ({})", parts[0].trim(), e))?;
                let end: usize = parts[1]
                    .trim()
                    .parse()
                    .map_err(|e| format!("Invalid k value: {} ({})", parts[1].trim(), e))?;
                if start == 0 || end == 0 {
                    return Err("k values must be positive".to_string());
                }
                if start > end {
                    return Err(format!("Invalid k range: {} (start > end)", token));
                }
                return Ok((start..=end).collect());
            }
        }
        let k: usize = token.parse().map_err(|e| format!("Invalid k value: {} ({})", token, e))?;
        if k == 0 {
            return Err("k must be greater than zero".to_string());
        }
        Ok(vec![k])
    }

    let mut result = Vec::new();
    for token in value.split(|c: char| c == ',' || c == ';' || c.is_whitespace()) {
        let mut parsed = parse_single(token)?;
        result.append(&mut parsed);
    }
    Ok(result)
}

fn open_writer(output: Option<PathBuf>) -> Result<Box<dyn Write>, String> {
    if let Some(path) = output {
        let file = fs::File::create(&path)
            .map_err(|e| format!("Failed to open output file {}: {}", path.display(), e))?;
        if path.extension().and_then(|ext| ext.to_str()).map_or(false, |ext| ext.eq_ignore_ascii_case("gz")) {
            let encoder = GzEncoder::new(BufWriter::new(file), Compression::default());
            Ok(Box::new(encoder))
        } else {
            Ok(Box::new(BufWriter::new(file)))
        }
    } else {
        Ok(Box::new(BufWriter::new(std::io::stdout())))
    }
}

fn print_usage(program_name: &str) {
    eprintln!("Usage: {} [OPTIONS] <input> [output]", program_name);
    eprintln!();
    eprintln!("Read a CSV or whitespace-delimited dataset and compute kNN-based outlier scores.");
    eprintln!("Options:");
    eprintln!("  -h, --help                Show this help message");
    eprintln!("  -i, --input <path>        Input file path (or supply as positional argument)");
    eprintln!("  -o, --output <path>       Output file path (default: stdout)");
    eprintln!("  -k, --ks <list>           k values (e.g. 1,2,5 or 1..10)");
    eprintln!("      --delimiter <val>     Delimiter: comma, tab, semicolon, whitespace, auto");
    eprintln!("      --format <val>        Input format: csv, npy, auto");
    eprintln!("      --header              Treat the first non-comment line as a header");
    eprintln!("      --no-header           Do not treat the first line as a header");
    eprintln!("      --disable <patterns>  Disable methods by substring or comma-separated list");
    eprintln!("      --ksquare-max <n>     Maximum k for quadratic-cost methods (default 1000)");
    eprintln!("      --time-limit <secs>   Per-method time limit in seconds");
}

fn parse_args() -> Result<Config, String> {
    let mut args = env::args().skip(1);
    let mut config = Config {
        input: PathBuf::new(),
        output: None,
        ks: Vec::new(),
        delimiter: None,
        format: None,
        header: None,
        disable: Vec::new(),
        ksquare_max: 1000,
        time_limit: None,
    };

    let mut positional = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage(
                    &env::args().next().unwrap_or_else(|| "compute_knn_outlier_scores".to_string()),
                );
                std::process::exit(0);
            }
            "-i" | "--input" => {
                config.input =
                    args.next().ok_or_else(|| "Missing value for --input".to_string())?.into();
            }
            "-o" | "--output" => {
                config.output = Some(
                    args.next().ok_or_else(|| "Missing value for --output".to_string())?.into(),
                );
            }
            "-k" | "--ks" => {
                let value = args.next().ok_or_else(|| "Missing value for --ks".to_string())?;
                config.ks = parse_ks(&value)?;
            }
            "--delimiter" => {
                let value =
                    args.next().ok_or_else(|| "Missing value for --delimiter".to_string())?;
                config.delimiter = parse_delimiter(&value)?;
            }
            "--format" => {
                let value = args.next().ok_or_else(|| "Missing value for --format".to_string())?;
                config.format = parse_format(&value)?;
            }
            "--header" => {
                config.header = Some(true);
            }
            "--no-header" => {
                config.header = Some(false);
            }
            "--disable" => {
                let value = args.next().ok_or_else(|| "Missing value for --disable".to_string())?;
                config.disable.extend(
                    value
                        .split(|c| [',', ';'].contains(&c))
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty()),
                );
            }
            "--ksquare-max" => {
                let value =
                    args.next().ok_or_else(|| "Missing value for --ksquare-max".to_string())?;
                config.ksquare_max =
                    value.parse().map_err(|e| format!("Invalid ksquare-max: {}", e))?;
            }
            "--time-limit" => {
                let value =
                    args.next().ok_or_else(|| "Missing value for --time-limit".to_string())?;
                let secs: f64 = value.parse().map_err(|e| format!("Invalid time-limit: {}", e))?;
                if secs < 0.0 {
                    return Err("time-limit must be non-negative".to_string());
                }
                config.time_limit = Some(Duration::from_secs_f64(secs));
            }
            _ if arg.starts_with('-') => {
                return Err(format!("Unknown option: {}", arg));
            }
            _ => positional.push(PathBuf::from(arg)),
        }
    }

    if config.input.as_os_str().is_empty() {
        if let Some(path) = positional.first() {
            config.input = path.clone();
        } else {
            return Err("Missing input file path".to_string());
        }
    }

    if config.output.is_none() && let Some(path) = positional.get(1) {
        config.output = Some(path.clone());
    }

    if config.ks.is_empty() {
        return Err("At least one k value is required".to_string());
    }

    config.ks.sort_unstable();
    config.ks.dedup();
    Ok(config)
}

/// Compute a batch of kNN-based outlier scores for any distance dataset and
/// write results as they are generated.
pub fn compute_knn_outlier_scores<'a, S, D, W>(
    writer: &mut W, tree: &S, data: &'a D, ks: impl IntoIterator<Item = usize>,
    options: ComputeKnnOutlierScoresOptions,
) -> std::io::Result<()>
where
    W: Write,
    D: DistanceData<f64> + VectorData<f64> + Sync + 'a,
    S: KnnSearch<f64, D::Query<'a>> + RangeSearch<f64, D::Query<'a>> + Sync,
{
    let ks = build_sorted_ks(ks);
    let size = data.len();
    let max_k = ks.iter().copied().max().unwrap_or(0).min(size.saturating_sub(1));
    let max_ksq = max_k.min(options.ksquare_max);

    run_for_each_k(writer, &options, &ks, "KNN", 1, max_k, |k| {
        k_nearest_neighbors_outlier(tree, data, k)
    })?;

    run_for_each_k(writer, &options, &ks, "KNNW", 1, max_k, |k| weighted_knn(tree, data, k))?;

    run_for_each_k(writer, &options, &ks, "LOF", 1, max_k, |k| {
        local_outlier_factor(tree, data, k)
    })?;

    run_for_each_k(writer, &options, &ks, "SimplifiedLOF", 1, max_k, |k| {
        simplified_lof(tree, data, k)
    })?;

    run_for_each_k(writer, &options, &ks, "LoOP", 1, max_k, |k| {
        local_outlier_probabilities(tree, data, k, 1.0)
    })?;

    run_for_each_k(writer, &options, &ks, "LDOF", 2, max_ksq, |k| {
        local_density_outlier_factor(tree, data, k)
    })?;

    run_for_each_k(writer, &options, &ks, "ODIN", 1, max_k, |k| {
        outlier_detection_independence_neighbor(tree, data, k)
    })?;

    run_for_each_k(writer, &options, &ks, "INFLO", 1, max_k, |k| {
        influence_outlier(tree, data, k, 1.0)
    })?;

    run_for_each_k(writer, &options, &ks, "COF", 2, max_ksq, |k| {
        connectivity_outlier_factor(tree, data, k)
    })?;

    run_for_each_k(writer, &options, &ks, "LID", 2, max_k, |k| {
        local_intrinsic_dimensionality::<_, _, _, AggregatedHillID>(tree, data, k)
    })?;

    run_for_each_k(writer, &options, &ks, "ABID", 2, max_k, |k| {
        local_intrinsic_dimensionality::<_, _, _, ABID>(tree, data, k)
    })?;

    run_for_each_k(writer, &options, &ks, "IDOS", 2, max_k, |k| {
        intrinsic_dimensionality_outlier_score::<_, _, _, AggregatedHillID>(tree, data, k, k)
    })?;

    run_for_each_k(writer, &options, &ks, "LIC", 1, max_k, |k| {
        local_isolation_coefficient(tree, data, k)
    })?;

    run_for_each_k(writer, &options, &ks, "KNNDD", 1, max_k, |k| {
        k_nearest_neighbors_distance_deviation(tree, data, k)
    })?;

    run_for_each_k(writer, &options, &ks, "KNNSOS", 1, max_k, |k| {
        k_nearest_neighbors_sos(tree, data, k)
    })?;

    run_for_each_k(writer, &options, &ks, "ISOS", 2, max_k, |k| {
        intrinsic_stochastic_outlier_selection::<_, _, _, AggregatedHillID>(tree, data, k)
    })?;

    run_for_each_k(writer, &options, &ks, "KDEOS", 2, max_k, |k| {
        kdeos(
            tree,
            data,
            k,
            k,
            KernelDensityFunction::Gaussian,
            0.0,
            0.5 * KernelDensityFunction::Gaussian.canonical_bandwidth(),
            Some(2),
        )
    })?;

    run_for_each_k(writer, &options, &ks, "LDF", 1, max_k, |k| {
        local_density_factor(tree, data, k, 1.0, 0.1, KernelDensityFunction::Gaussian)
    })?;

    run_for_each_k(writer, &options, &ks, "KDLOF", 2, max_k, |k| {
        simple_kernel_density_lof(tree, data, k, 0.0, KernelDensityFunction::Gaussian)
    })?;

    run_for_each_k(writer, &options, &ks, "VOV", 1, max_k, |k| variance_of_volume(tree, data, k))?;

    run_for_each_k(writer, &options, &ks, "FastABOD", 3, max_ksq, |k| {
        let kernel = PolynomialKernel::new(2, 1.0_f64, 0.0_f64);
        fast_angle_based_outlier_detection(tree, data, k, |x, y| kernel.similarity(x, y))
    })?;

    Ok(())
}

/// Write batch outlier results in the same row-oriented format used by the ELKI
/// `ComputeKNNOutlierScores` application.
pub fn write_batch_outlier_scores<W>(
    writer: &mut W, results: &[BatchOutlierResult], scaling: Option<&dyn Fn(f64) -> f64>,
) -> std::io::Result<()>
where
    W: std::io::Write,
{
    for result in results {
        write!(writer, "{}", result.label)?;
        for score in result.result.scores.iter() {
            let value = scaling.map_or(*score, |scale| scale(*score));
            write!(writer, " {}", format_value(value))?;
        }
        writeln!(writer)?;
    }
    Ok(())
}

fn format_value(value: f64) -> String {
    let mut s = format!("{:?}", value);
    if s.ends_with(".0") {
        s.truncate(s.len() - 2);
    }
    s
}

fn main() {
    let config = match parse_args() {
        Ok(config) => config,
        Err(err) => {
            eprintln!("Error: {}", err);
            print_usage(
                &env::args().next().unwrap_or_else(|| "compute_knn_outlier_scores".to_string()),
            );
            std::process::exit(1);
        }
    };

    let load_start = Instant::now();
    let points = match read_numeric_table(
        &config.input,
        ReadOptions {
            format: config.format,
            delimiter: config.delimiter,
            header: config.header,
            comment_char: Some(b'#'),
        },
    ) {
        Ok(points) => points,
        Err(err) => {
            eprintln!("Failed to parse input file {}: {}", config.input.display(), err);
            std::process::exit(1);
        }
    };
    let load_time = load_start.elapsed();
    eprintln!("Data loading time: {:.3}s", load_time.as_secs_f64());

    if points.is_empty() {
        eprintln!("No numeric points were loaded from {}", config.input.display());
        std::process::exit(1);
    }

    let data = fuel::TableWithDistance::with_distance(&points, fuel::distance::Euclidean);
    let mut rng = StdRng::seed_from_u64(42);
    let index_start = Instant::now();
    let tree = fuel::search::vptree::VPTree::new(&data, 2, &mut rng);
    let ks = build_sorted_ks(config.ks.clone());
    let max_k = ks.iter().copied().max().unwrap_or(0).min(data.len().saturating_sub(1));
    let proxy = ProxyKnnSearcher::new(&tree, &data, max_k, 2);
    let index_time = index_start.elapsed();
    eprintln!("Data indexing time: {:.3}s", index_time.as_secs_f64());

    let options = ComputeKnnOutlierScoresOptions {
        disabled: config.disable,
        ksquare_max: config.ksquare_max,
        time_limit: config.time_limit,
    };

    let mut writer = match open_writer(config.output) {
        Ok(writer) => writer,
        Err(err) => {
            eprintln!("Failed to open output: {}", err);
            std::process::exit(1);
        }
    };

    if let Err(err) = compute_knn_outlier_scores(&mut writer, &proxy, &data, ks, options) {
        eprintln!("Failed to compute or write outlier scores: {}", err);
        std::process::exit(1);
    }
}
