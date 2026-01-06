//! Constants for CUDA allocation requirements.
//!
//! Defines hardware requirements for RTX 5090 (Blackwell) target hardware.

/// Required compute capability major version for RTX 5090 (Blackwell).
///
/// RTX 5090 has compute capability 12.0. We require this as the minimum
/// to ensure Blackwell-specific optimizations are available.
pub const REQUIRED_COMPUTE_MAJOR: u32 = 12;

/// Required compute capability minor version for RTX 5090.
pub const REQUIRED_COMPUTE_MINOR: u32 = 0;

/// Minimum VRAM required in bytes (32GB for RTX 5090).
///
/// This is the total VRAM on an RTX 5090. We require the full amount
/// to ensure all 12 embedding models can be loaded.
pub const MINIMUM_VRAM_BYTES: usize = 32 * 1024 * 1024 * 1024;

/// One gigabyte in bytes.
pub const GB: usize = 1024 * 1024 * 1024;

/// Maximum number of allocation history entries to keep.
pub const MAX_ALLOCATION_HISTORY: usize = 100;

/// Fake allocation base address pattern.
///
/// If a CUDA allocation returns a pointer with this base address pattern,
/// it indicates a mock/fake allocation that is NOT real VRAM.
///
/// # Constitution Compliance
///
/// AP-007: Fake allocations MUST cause exit(109).
/// This pattern is used to detect mock implementations.
pub const FAKE_ALLOCATION_BASE_PATTERN: u64 = 0x7f80_0000_0000;

/// Mask for extracting base address from device pointer.
///
/// Uses upper 20 bits to identify the memory region.
pub const FAKE_ALLOCATION_BASE_MASK: u64 = 0xFFFF_F000_0000_0000;

/// Energy concentration threshold for sin wave detection (80%).
///
/// If more than 80% of FFT energy is concentrated in a single frequency band,
/// the output is considered a fake sin wave pattern.
pub const SIN_WAVE_ENERGY_THRESHOLD: f32 = 0.80;

/// Golden similarity threshold for inference validation.
///
/// Output embeddings must have > 0.99 cosine similarity to golden reference.
pub const GOLDEN_SIMILARITY_THRESHOLD: f32 = 0.99;
