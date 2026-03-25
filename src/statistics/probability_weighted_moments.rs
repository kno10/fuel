/// Probability-Weighted Moment and L-moment utilities.
/// Compute sample L-moments from sorted data using the method of probability-weighted moments.
///
/// The returned vector contains B(0)=L1, B(1)=L2, then tau3... for nmom>=3.
pub fn sam_lmr(sorted: &[f64], nmom: usize) -> Vec<f64> {
    let n = sorted.len();
    let nmom = std::cmp::min(n, nmom);
    if nmom == 0 {
        return Vec::new();
    }
    let mut sum = vec![0.0; nmom];

    for (i, &val) in sorted.iter().enumerate() {
        if !val.is_finite() {
            continue;
        }
        let mut term = val;
        sum[0] += term;
        let mut z = i as f64;
        for j in 1..nmom {
            term *= z;
            sum[j] += term;
            z -= 1.0;
            if z < 0.0 {
                break;
            }
        }
    }

    if n == 0 {
        return sum;
    }

    sum[0] /= n as f64;
    let mut z = n as f64;
    for (j, value) in sum.iter_mut().enumerate().take(nmom).skip(1) {
        z *= (n - j) as f64;
        if z == 0.0 {
            *value = f64::NAN;
        } else {
            *value /= z;
        }
    }

    // normalize L-moments (lambda-to-tau conversion)
    for k in (1..nmom).rev() {
        let mut p = if (k % 2) == 0 { 1.0 } else { -1.0 };
        let mut temp = p * sum[0];
        for i in 0..k {
            let ai = (i + 1) as f64;
            p *= -((k as f64 + ai) * (k as f64 - i as f64)) / (ai * ai);
            temp += sum[i + 1] * p;
        }
        sum[k] = temp;
    }

    if nmom > 1 && sum[1] == 0.0 {
        sum[2..].fill(0.0);
        return sum;
    }

    if sum[1] != 0.0 {
        let denom = sum[1];
        for val in sum.iter_mut().skip(2) {
            *val /= denom;
        }
    }

    sum
}
