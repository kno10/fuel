#![allow(clippy::all)]

use super::contingency_table::ClusterContingencyTable;

/// Pair-counting statistics derived from a contingency table.
#[derive(Debug, Clone, Copy)]
pub struct PairCounting {
    pub in_both: u64,
    pub in_first: u64,
    pub in_second: u64,
    pub in_none: u64,
}

impl PairCounting {
    pub fn new(table: &ClusterContingencyTable) -> Self {
        let mut in_b = 0u64;
        let mut in1 = 0u64;
        let mut in2 = 0u64;

        for i in 0..table.size1 {
            let size = table.contingency[i][table.size2 + 1] as u64;
            if table.break_noise_clusters && table.noise1[i] {
                if table.self_pairing {
                    in1 += size;
                }
            } else {
                in1 += size
                    * if table.self_pairing {
                        size
                    } else {
                        size.saturating_sub(1)
                    };
            }
        }

        for j in 0..table.size2 {
            let size = table.contingency[table.size1 + 1][j] as u64;
            if table.break_noise_clusters && table.noise2[j] {
                if table.self_pairing {
                    in2 += size;
                }
            } else {
                in2 += size
                    * if table.self_pairing {
                        size
                    } else {
                        size.saturating_sub(1)
                    };
            }
        }

        for i in 0..table.size1 {
            for j in 0..table.size2 {
                let size = table.contingency[i][j] as u64;
                if table.break_noise_clusters && (table.noise1[i] || table.noise2[j]) {
                    if table.self_pairing {
                        in_b += size;
                    }
                } else {
                    in_b += size
                        * if table.self_pairing {
                            size
                        } else {
                            size.saturating_sub(1)
                        };
                }
            }
        }

        let tsize = table.contingency[table.size1][table.size2] as u64;
        let total = tsize
            * if table.self_pairing {
                tsize
            } else {
                tsize.saturating_sub(1)
            };
        let in_first = in1.saturating_sub(in_b);
        let in_second = in2.saturating_sub(in_b);
        let in_none = total.saturating_sub(in_b + in_first + in_second);

        Self {
            in_both: in_b,
            in_first,
            in_second,
            in_none,
        }
    }

    // the various summary measures can stay here or be in helpers, but we can
    // include them for convenience.
    pub fn f_measure(&self, beta: f64) -> f64 {
        let beta2 = beta * beta;
        let a = (1.0 + beta2) * self.in_both as f64;
        let b = a + beta2 * self.in_first as f64 + self.in_second as f64;
        if b == 0.0 { 0.0 } else { a / b }
    }

    pub fn f1_measure(&self) -> f64 {
        let a = 2.0 * self.in_both as f64;
        let b = a + self.in_first as f64 + self.in_second as f64;
        if b == 0.0 { 0.0 } else { a / b }
    }

    pub fn precision(&self) -> f64 {
        let d = (self.in_both + self.in_second) as f64;
        if d == 0.0 {
            0.0
        } else {
            self.in_both as f64 / d
        }
    }

    pub fn recall(&self) -> f64 {
        let d = (self.in_both + self.in_first) as f64;
        if d == 0.0 {
            0.0
        } else {
            self.in_both as f64 / d
        }
    }

    pub fn fowlkes_mallows(&self) -> f64 {
        (self.precision() * self.recall()).sqrt()
    }

    pub fn rand_index(&self) -> f64 {
        let d = (self.in_both + self.in_first + self.in_second + self.in_none) as f64;
        if d == 0.0 {
            0.0
        } else {
            (self.in_both + self.in_none) as f64 / d
        }
    }

    pub fn adjusted_rand_index(&self) -> f64 {
        let d = ((self.in_both + self.in_first + self.in_second + self.in_none) as f64).sqrt();
        if d == 0.0 {
            return 0.0;
        }
        let exp =
            (self.in_both + self.in_first) as f64 / d * (self.in_both + self.in_second) as f64 / d;
        let opt = self.in_both as f64 + 0.5 * (self.in_first + self.in_second) as f64;
        let denom = opt - exp;
        if denom == 0.0 {
            0.0
        } else {
            (self.in_both as f64 - exp) / denom
        }
    }

    pub fn jaccard(&self) -> f64 {
        let d = (self.in_both + self.in_first + self.in_second) as f64;
        if d == 0.0 {
            0.0
        } else {
            self.in_both as f64 / d
        }
    }

    pub fn mirkin(&self) -> u64 {
        self.in_first + self.in_second
    }
}
