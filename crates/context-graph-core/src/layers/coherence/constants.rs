//! Constants for Global Workspace Theory thresholds.
//!
//! # Deprecation Notice
//!
//! The threshold constants (`GW_THRESHOLD`, `HYPERSYNC_THRESHOLD`, `FRAGMENTATION_THRESHOLD`)
//! are deprecated in favor of domain-aware [`GwtThresholds`](super::GwtThresholds).
//! Use `GwtThresholds::from_atc()` or `GwtThresholds::default_general()` instead.

/// Global workspace ignition threshold from constitution (coherence_threshold: 0.8)
/// Using 0.7 as per task spec for GW_THRESHOLD
///
/// # Deprecation
///
/// Use [`GwtThresholds::default_general().gate`](super::GwtThresholds::default_general) or
/// [`GwtThresholds::from_atc()`](super::GwtThresholds::from_atc) for domain-aware thresholds.
#[deprecated(
    since = "0.5.0",
    note = "Use GwtThresholds::default_general().gate or GwtThresholds::from_atc() instead"
)]
pub const GW_THRESHOLD: f32 = 0.7;

/// Number of integration steps per process call
pub const INTEGRATION_STEPS: usize = 10;

/// Hypersync threshold (r > 0.95 is pathological)
///
/// # Deprecation
///
/// Use [`GwtThresholds::default_general().hypersync`](super::GwtThresholds::default_general) or
/// [`GwtThresholds::from_atc()`](super::GwtThresholds::from_atc) for domain-aware thresholds.
#[deprecated(
    since = "0.5.0",
    note = "Use GwtThresholds::default_general().hypersync or GwtThresholds::from_atc() instead"
)]
pub const HYPERSYNC_THRESHOLD: f32 = 0.95;

/// Fragmentation threshold (r < 0.5)
///
/// # Deprecation
///
/// Use [`GwtThresholds::default_general().fragmentation`](super::GwtThresholds::default_general) or
/// [`GwtThresholds::from_atc()`](super::GwtThresholds::from_atc) for domain-aware thresholds.
#[deprecated(
    since = "0.5.0",
    note = "Use GwtThresholds::default_general().fragmentation or GwtThresholds::from_atc() instead"
)]
pub const FRAGMENTATION_THRESHOLD: f32 = 0.5;
