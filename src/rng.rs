//! Counter-based random numbers, shared between the CPU and GPU samplers.
//!
//! Uses the `pcg4d` hash from Jarzynski & Olano, "Hash Functions for GPU
//! Rendering" (JCGT 2020). Every random draw is a pure function of
//! `(seed, step, chain, node)`, so parallel updates need no shared state
//! and CPU results are reproducible for a fixed seed.
//!
//! The `step` coordinate counts block updates from 0 during sampling.
//! The values at the top of the range are reserved for other draws, so
//! that they never collide with the sampling stream:
//!
//! - [`STEP_RANDOM_INIT`] for [`State::random`](crate::State::random),
//! - [`STEP_HINTON_INIT`] for Hinton initialization.

/// The step reserved for uniformly random state initialization.
pub const STEP_RANDOM_INIT: u32 = !0;

/// The step reserved for bias-based (Hinton) state initialization.
pub const STEP_HINTON_INIT: u32 = !1;

/// The 4D hash underlying all random draws of the samplers.
pub fn pcg4d(mut v: [u32; 4]) -> [u32; 4] {
    for x in v.iter_mut() {
        *x = x.wrapping_mul(1664525).wrapping_add(1013904223);
    }
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[3]));
    v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
    v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    v[3] = v[3].wrapping_add(v[1].wrapping_mul(v[2]));
    for x in v.iter_mut() {
        *x ^= *x >> 16;
    }
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[3]));
    v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
    v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    v[3] = v[3].wrapping_add(v[1].wrapping_mul(v[2]));
    v
}

/// A uniform draw in `[0, 1)` for the given counter coordinates.
pub fn uniform(seed: u32, step: u32, chain: u32, node: u32) -> f32 {
    let hash = pcg4d([seed, step, chain, node]);
    (hash[0] >> 8) as f32 * (1.0 / (1 << 24) as f32)
}
