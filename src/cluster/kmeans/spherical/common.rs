use num_traits::Float;

#[inline(always)]
pub(crate) fn clamp_one<N: Float>(x: N) -> N {
    if x > N::one() { N::one() } else { x }
}

#[inline(always)]
pub(crate) fn sqrt_half_sim<N: Float>(sim: N) -> N {
    if sim > -N::one() {
        ((sim + N::one()) * N::from(0.5).unwrap()).sqrt()
    } else {
        N::zero()
    }
}

#[inline(always)]
pub(crate) fn sim_lower_bound<N: Float>(s1: N, s2: N) -> N {
    let v1 = clamp_one(s1);
    let v2 = clamp_one(s2);
    let rad = (N::one() - v1 * v1) * (N::one() - v2 * v2);
    v1 * v2
        - if rad > N::zero() {
            rad.sqrt()
        } else {
            N::zero()
        }
}

#[inline(always)]
pub(crate) fn sim_upper_bound<N: Float>(s1: N, s2: N) -> N {
    let v1 = clamp_one(s1);
    let v2 = clamp_one(s2);
    let rad = (N::one() - v1 * v1) * (N::one() - v2 * v2);
    v1 * v2
        + if rad > N::zero() {
            rad.sqrt()
        } else {
            N::zero()
        }
}
