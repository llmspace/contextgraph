//! Constants and configuration for the Temporal-Positional embedding model (E4).

/// Native dimension for TemporalPositional model (E4).
pub const TEMPORAL_POSITIONAL_DIMENSION: usize = 512;

/// Default base frequency for sinusoidal encoding (transformer standard).
/// This value is from the original "Attention Is All You Need" paper.
pub const DEFAULT_BASE: f32 = 10000.0;

/// Minimum valid base frequency (must be > 1.0 for proper frequency scaling).
pub(crate) const MIN_BASE: f32 = 1.0;

/// Maximum valid base frequency (prevent numerical issues).
pub(crate) const MAX_BASE: f32 = 1e10;
