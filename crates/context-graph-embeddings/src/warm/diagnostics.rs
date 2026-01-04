//! Diagnostic Dump Module for Warm Model Loading
//!
//! Provides comprehensive diagnostic reporting for the warm model loading system.
//! Supports JSON serialization for automated monitoring and human-readable output
//! for debugging.
//!
//! # Overview
//!
//! The [`WarmDiagnostics`] service generates detailed diagnostic reports that capture:
//! - System information (hostname, OS, uptime)
//! - GPU information (device, VRAM, compute capability)
//! - Memory pool status (model pool, working pool)
//! - Per-model loading state and VRAM allocations
//! - Any errors encountered during loading
//!
//! # Design Principles
//!
//! - **NO WORKAROUNDS OR FALLBACKS**: Diagnostics must be accurate
//! - **COMPREHENSIVE LOGGING**: Full context on any issue
//! - **SERIALIZABLE**: Support JSON output for automated monitoring
//!
//! # Example
//!
//! ```rust,ignore
//! use context_graph_embeddings::warm::{WarmLoader, WarmConfig};
//! use context_graph_embeddings::warm::diagnostics::WarmDiagnostics;
//!
//! let config = WarmConfig::default();
//! let loader = WarmLoader::new(config)?;
//!
//! // Generate and print diagnostic report
//! let report = WarmDiagnostics::generate_report(&loader);
//! println!("{}", WarmDiagnostics::to_json(&loader)?);
//!
//! // On fatal error, dump to stderr
//! WarmDiagnostics::dump_to_stderr(&loader);
//! ```

use std::fs::File;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::cuda_alloc::GpuInfo;
use super::error::{WarmError, WarmResult};
use super::loader::WarmLoader;
use super::state::WarmModelState;

// ============================================================================
// Diagnostic Report Structures
// ============================================================================

/// Complete diagnostic report for the warm model loading system.
///
/// Contains all information needed to diagnose loading issues, including
/// system info, GPU status, memory usage, and per-model state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmDiagnosticReport {
    /// ISO 8601 timestamp when report was generated.
    pub timestamp: String,
    /// System information.
    pub system: SystemInfo,
    /// GPU information (None if no GPU available).
    pub gpu: Option<GpuDiagnostics>,
    /// Memory pool status.
    pub memory: MemoryDiagnostics,
    /// Per-model diagnostic information.
    pub models: Vec<ModelDiagnostic>,
    /// Any errors encountered during loading.
    pub errors: Vec<ErrorDiagnostic>,
}

impl WarmDiagnosticReport {
    /// Create an empty diagnostic report with current timestamp.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            timestamp: Self::current_timestamp(),
            system: SystemInfo::default(),
            gpu: None,
            memory: MemoryDiagnostics::default(),
            models: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Get the current ISO 8601 timestamp.
    fn current_timestamp() -> String {
        use std::time::SystemTime;
        let now = SystemTime::now();
        let duration = now
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();
        let secs = duration.as_secs();
        let millis = duration.subsec_millis();

        // Format as ISO 8601: 2025-01-03T12:00:00.000Z
        // Note: This is a simplified implementation; production would use chrono
        let days_since_epoch = secs / 86400;
        let secs_today = secs % 86400;
        let hours = secs_today / 3600;
        let minutes = (secs_today % 3600) / 60;
        let seconds = secs_today % 60;

        // Approximate date calculation (not accounting for leap years perfectly)
        let mut year = 1970;
        let mut remaining_days = days_since_epoch;

        while remaining_days >= 365 {
            let days_in_year = if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                366
            } else {
                365
            };
            if remaining_days >= days_in_year {
                remaining_days -= days_in_year;
                year += 1;
            } else {
                break;
            }
        }

        let days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let mut month = 1;
        for &days_in_month in &days_in_months {
            let days = if month == 2
                && year % 4 == 0
                && (year % 100 != 0 || year % 400 == 0)
            {
                29
            } else {
                days_in_month
            };
            if remaining_days >= days {
                remaining_days -= days;
                month += 1;
            } else {
                break;
            }
        }
        let day = remaining_days + 1;

        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
            year, month, day, hours, minutes, seconds, millis
        )
    }

    /// Check if any errors were recorded.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Get the count of warm models.
    #[must_use]
    pub fn warm_count(&self) -> usize {
        self.models.iter().filter(|m| m.state == "Warm").count()
    }

    /// Get the count of failed models.
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.models.iter().filter(|m| m.state == "Failed").count()
    }
}

/// System information for diagnostic context.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemInfo {
    /// System hostname.
    pub hostname: String,
    /// Operating system description.
    pub os: String,
    /// System uptime in seconds.
    pub uptime_seconds: f64,
}

impl SystemInfo {
    /// Gather system information from the current environment.
    #[must_use]
    pub fn gather() -> Self {
        let hostname = std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("COMPUTERNAME"))
            .unwrap_or_else(|_| {
                // Fallback: try reading /etc/hostname on Unix
                std::fs::read_to_string("/etc/hostname")
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|_| "unknown".to_string())
            });

        let os = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);

        // Uptime is hard to get portably; use process uptime as proxy
        let uptime_seconds = 0.0; // Would need platform-specific code

        Self {
            hostname,
            os,
            uptime_seconds,
        }
    }
}

/// GPU diagnostic information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDiagnostics {
    /// CUDA device ID.
    pub device_id: u32,
    /// GPU model name.
    pub name: String,
    /// Compute capability as string (e.g., "12.0").
    pub compute_capability: String,
    /// Total VRAM in bytes.
    pub total_vram_bytes: usize,
    /// Available (free) VRAM in bytes.
    pub available_vram_bytes: usize,
    /// CUDA driver version.
    pub driver_version: String,
}

impl GpuDiagnostics {
    /// Create GPU diagnostics from a [`GpuInfo`] structure.
    #[must_use]
    pub fn from_gpu_info(info: &GpuInfo, available_bytes: usize) -> Self {
        Self {
            device_id: info.device_id,
            name: info.name.clone(),
            compute_capability: info.compute_capability_string(),
            total_vram_bytes: info.total_memory_bytes,
            available_vram_bytes: available_bytes,
            driver_version: info.driver_version.clone(),
        }
    }
}

/// Memory pool diagnostic information.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryDiagnostics {
    /// Model pool capacity in bytes.
    pub model_pool_capacity_bytes: usize,
    /// Model pool used bytes.
    pub model_pool_used_bytes: usize,
    /// Working pool capacity in bytes.
    pub working_pool_capacity_bytes: usize,
    /// Working pool used bytes.
    pub working_pool_used_bytes: usize,
    /// Total number of model allocations.
    pub total_allocations: usize,
}

/// Per-model diagnostic information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDiagnostic {
    /// Model identifier (e.g., "E1_Semantic").
    pub model_id: String,
    /// Current state as string (e.g., "Warm", "Loading", "Failed").
    pub state: String,
    /// Expected size in bytes.
    pub expected_bytes: usize,
    /// Allocated bytes (if loaded).
    pub allocated_bytes: Option<usize>,
    /// VRAM pointer as hex string (if loaded).
    pub vram_ptr: Option<String>,
    /// Weight checksum as hex string (if loaded).
    pub checksum: Option<String>,
    /// Error message (if failed).
    pub error_message: Option<String>,
}

impl ModelDiagnostic {
    /// Create a diagnostic entry for a model from registry state.
    #[must_use]
    pub fn from_state(
        model_id: &str,
        state: &WarmModelState,
        expected_bytes: usize,
        handle_info: Option<(u64, usize, u64)>, // (vram_ptr, size, checksum)
    ) -> Self {
        let state_str = match state {
            WarmModelState::Pending => "Pending".to_string(),
            WarmModelState::Loading {
                progress_percent, ..
            } => format!("Loading ({}%)", progress_percent),
            WarmModelState::Validating => "Validating".to_string(),
            WarmModelState::Warm => "Warm".to_string(),
            WarmModelState::Failed { .. } => "Failed".to_string(),
        };

        let error_message = if let WarmModelState::Failed { error_message, .. } = state {
            Some(error_message.clone())
        } else {
            None
        };

        let (allocated_bytes, vram_ptr, checksum) = if let Some((ptr, size, chk)) = handle_info {
            (
                Some(size),
                Some(format!("0x{:016x}", ptr)),
                Some(format!("0x{:016X}", chk)),
            )
        } else {
            (None, None, None)
        };

        Self {
            model_id: model_id.to_string(),
            state: state_str,
            expected_bytes,
            allocated_bytes,
            vram_ptr,
            checksum,
            error_message,
        }
    }
}

/// Error diagnostic information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDiagnostic {
    /// Structured error code (e.g., "ERR-WARM-CUDA-INIT").
    pub error_code: String,
    /// Error category (e.g., "CUDA", "MODEL", "VRAM").
    pub category: String,
    /// Human-readable error message.
    pub message: String,
    /// Process exit code for this error.
    pub exit_code: i32,
}

impl ErrorDiagnostic {
    /// Create an error diagnostic from a [`WarmError`].
    #[must_use]
    pub fn from_error(error: &WarmError) -> Self {
        Self {
            error_code: error.error_code().to_string(),
            category: error.category().to_string(),
            message: error.to_string(),
            exit_code: error.exit_code(),
        }
    }
}

// ============================================================================
// Diagnostic Service
// ============================================================================

/// Diagnostic service for warm model loading.
///
/// Provides methods to generate diagnostic reports in various formats,
/// suitable for both human consumption and automated monitoring.
///
/// # Design
///
/// This is a stateless service with only static methods. All diagnostic
/// information is gathered from the [`WarmLoader`] instance.
pub struct WarmDiagnostics;

impl WarmDiagnostics {
    /// Generate a complete diagnostic report from the loader state.
    ///
    /// Captures:
    /// - System information
    /// - GPU information (if available)
    /// - Memory pool status
    /// - Per-model state and allocations
    /// - Any errors from failed models
    ///
    /// # Arguments
    ///
    /// * `loader` - Reference to the warm loader
    ///
    /// # Returns
    ///
    /// A [`WarmDiagnosticReport`] containing all diagnostic information.
    #[must_use]
    pub fn generate_report(loader: &WarmLoader) -> WarmDiagnosticReport {
        let mut report = WarmDiagnosticReport::empty();

        // Gather system information
        report.system = SystemInfo::gather();

        // Gather GPU information
        if let Some(gpu_info) = loader.gpu_info() {
            let available = loader
                .memory_pools()
                .model_pool_capacity()
                .saturating_sub(loader.memory_pools().list_model_allocations().iter().map(|a| a.size_bytes).sum::<usize>());
            report.gpu = Some(GpuDiagnostics::from_gpu_info(gpu_info, available));
        }

        // Gather memory pool information
        let pools = loader.memory_pools();
        let allocations = pools.list_model_allocations();
        let model_pool_used: usize = allocations.iter().map(|a| a.size_bytes).sum();

        report.memory = MemoryDiagnostics {
            model_pool_capacity_bytes: pools.model_pool_capacity(),
            model_pool_used_bytes: model_pool_used,
            working_pool_capacity_bytes: pools.working_pool_capacity(),
            working_pool_used_bytes: pools.working_pool_capacity()
                .saturating_sub(pools.available_working_bytes()),
            total_allocations: allocations.len(),
        };

        // Gather per-model information
        let registry = match loader.registry().read() {
            Ok(r) => r,
            Err(_) => return report, // Lock poisoned, return partial report
        };

        for model_id in super::registry::EMBEDDING_MODEL_IDS
            .iter()
            .chain(std::iter::once(&super::registry::FUSEMOE_MODEL_ID))
        {
            if let Some(entry) = registry.get_entry(model_id) {
                let handle_info = entry.handle.as_ref().map(|h| {
                    (h.vram_address(), h.allocation_bytes(), h.weight_checksum())
                });

                let diagnostic =
                    ModelDiagnostic::from_state(model_id, &entry.state, entry.expected_bytes, handle_info);

                // Collect errors from failed models
                if let WarmModelState::Failed {
                    error_code,
                    error_message,
                } = &entry.state
                {
                    report.errors.push(ErrorDiagnostic {
                        error_code: format!("ERR-WARM-MODEL-{}", error_code),
                        category: "MODEL".to_string(),
                        message: error_message.clone(),
                        exit_code: *error_code as i32,
                    });
                }

                report.models.push(diagnostic);
            }
        }

        report
    }

    /// Generate a JSON string representation of the diagnostic report.
    ///
    /// # Arguments
    ///
    /// * `loader` - Reference to the warm loader
    ///
    /// # Returns
    ///
    /// `Ok(String)` containing pretty-printed JSON, or `Err` on serialization failure.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let json = WarmDiagnostics::to_json(&loader)?;
    /// println!("{}", json);
    /// ```
    pub fn to_json(loader: &WarmLoader) -> WarmResult<String> {
        let report = Self::generate_report(loader);
        serde_json::to_string_pretty(&report).map_err(|e| WarmError::DiagnosticDumpFailed {
            reason: format!("JSON serialization failed: {}", e),
        })
    }

    /// Write the diagnostic report to a file.
    ///
    /// # Arguments
    ///
    /// * `loader` - Reference to the warm loader
    /// * `path` - Path to write the report to
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err` on file I/O failure.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// WarmDiagnostics::write_to_file(&loader, Path::new("/tmp/warm_diagnostics.json"))?;
    /// ```
    pub fn write_to_file(loader: &WarmLoader, path: &Path) -> WarmResult<()> {
        let json = Self::to_json(loader)?;

        let mut file = File::create(path).map_err(|e| WarmError::DiagnosticDumpFailed {
            reason: format!("Failed to create file {}: {}", path.display(), e),
        })?;

        file.write_all(json.as_bytes())
            .map_err(|e| WarmError::DiagnosticDumpFailed {
                reason: format!("Failed to write to file {}: {}", path.display(), e),
            })?;

        tracing::info!("Diagnostic report written to {}", path.display());
        Ok(())
    }

    /// Dump diagnostic report to stderr.
    ///
    /// Used for fatal errors when the system is about to exit.
    /// This is a best-effort operation that should never fail.
    ///
    /// # Arguments
    ///
    /// * `loader` - Reference to the warm loader
    pub fn dump_to_stderr(loader: &WarmLoader) {
        let report = Self::generate_report(loader);

        eprintln!("\n=== WARM MODEL LOADING DIAGNOSTIC DUMP ===");
        eprintln!("Timestamp: {}", report.timestamp);
        eprintln!();

        // System info
        eprintln!("SYSTEM:");
        eprintln!("  Hostname: {}", report.system.hostname);
        eprintln!("  OS: {}", report.system.os);
        eprintln!();

        // GPU info
        if let Some(gpu) = &report.gpu {
            eprintln!("GPU:");
            eprintln!("  Device: {} (ID: {})", gpu.name, gpu.device_id);
            eprintln!("  Compute Capability: {}", gpu.compute_capability);
            eprintln!("  Total VRAM: {}", format_bytes(gpu.total_vram_bytes));
            eprintln!("  Available VRAM: {}", format_bytes(gpu.available_vram_bytes));
            eprintln!("  Driver Version: {}", gpu.driver_version);
        } else {
            eprintln!("GPU: Not available");
        }
        eprintln!();

        // Memory info
        eprintln!("MEMORY:");
        eprintln!(
            "  Model Pool: {} / {} ({:.1}%)",
            format_bytes(report.memory.model_pool_used_bytes),
            format_bytes(report.memory.model_pool_capacity_bytes),
            if report.memory.model_pool_capacity_bytes > 0 {
                (report.memory.model_pool_used_bytes as f64
                    / report.memory.model_pool_capacity_bytes as f64)
                    * 100.0
            } else {
                0.0
            }
        );
        eprintln!(
            "  Working Pool: {} / {} ({:.1}%)",
            format_bytes(report.memory.working_pool_used_bytes),
            format_bytes(report.memory.working_pool_capacity_bytes),
            if report.memory.working_pool_capacity_bytes > 0 {
                (report.memory.working_pool_used_bytes as f64
                    / report.memory.working_pool_capacity_bytes as f64)
                    * 100.0
            } else {
                0.0
            }
        );
        eprintln!("  Total Allocations: {}", report.memory.total_allocations);
        eprintln!();

        // Model status
        eprintln!("MODELS ({} total):", report.models.len());
        for model in &report.models {
            let status_icon = match model.state.as_str() {
                "Warm" => "[OK]",
                "Failed" => "[FAIL]",
                s if s.starts_with("Loading") => "[...]",
                "Validating" => "[VAL]",
                _ => "[---]",
            };

            eprintln!(
                "  {} {} - {} (expected: {})",
                status_icon,
                model.model_id,
                model.state,
                format_bytes(model.expected_bytes)
            );

            if let Some(ptr) = &model.vram_ptr {
                eprintln!("      VRAM: {} ({} allocated)", ptr,
                    model.allocated_bytes.map(format_bytes).unwrap_or_else(|| "N/A".to_string()));
            }

            if let Some(err) = &model.error_message {
                eprintln!("      ERROR: {}", err);
            }
        }
        eprintln!();

        // Errors
        if !report.errors.is_empty() {
            eprintln!("ERRORS ({}):", report.errors.len());
            for error in &report.errors {
                eprintln!(
                    "  [{}] {} (exit code {})",
                    error.category, error.error_code, error.exit_code
                );
                eprintln!("      {}", error.message);
            }
            eprintln!();
        }

        eprintln!("=== END DIAGNOSTIC DUMP ===\n");
    }

    /// Generate a minimal status line for quick monitoring.
    ///
    /// Format: `WARM: 13/13 models | 24.0GB/24.0GB VRAM | OK`
    /// Or on error: `WARM: 12/13 models | 23.5GB/24.0GB VRAM | ERRORS: 1`
    ///
    /// # Arguments
    ///
    /// * `loader` - Reference to the warm loader
    ///
    /// # Returns
    ///
    /// A concise status string suitable for logs or monitoring dashboards.
    #[must_use]
    pub fn status_line(loader: &WarmLoader) -> String {
        let summary = loader.loading_summary();
        let pools = loader.memory_pools();

        let model_pool_used: usize = pools
            .list_model_allocations()
            .iter()
            .map(|a| a.size_bytes)
            .sum();

        let status = if summary.models_failed > 0 {
            format!("ERRORS: {}", summary.models_failed)
        } else if summary.models_warm == summary.total_models && summary.total_models > 0 {
            "OK".to_string()
        } else {
            format!(
                "LOADING: {}/{}",
                summary.models_warm, summary.total_models
            )
        };

        format!(
            "WARM: {}/{} models | {}/{} VRAM | {}",
            summary.models_warm,
            summary.total_models,
            format_bytes(model_pool_used),
            format_bytes(pools.model_pool_capacity()),
            status
        )
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Format bytes as a human-readable string.
fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2}GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2}MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2}KB", bytes as f64 / KB as f64)
    } else {
        format!("{}B", bytes)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::warm::config::WarmConfig;
    use crate::warm::handle::ModelHandle;
    use crate::warm::registry::{EMBEDDING_MODEL_IDS, FUSEMOE_MODEL_ID, TOTAL_MODEL_COUNT};

    /// Create a test config that doesn't require real files.
    fn test_config() -> WarmConfig {
        let mut config = WarmConfig::default();
        config.enable_test_inference = true;
        config
    }

    // ========================================================================
    // Test 1: Diagnostic Report Structure
    // ========================================================================

    #[test]
    fn test_diagnostic_report_structure() {
        let config = test_config();
        let loader = WarmLoader::new(config).expect("Failed to create loader");

        let report = WarmDiagnostics::generate_report(&loader);

        // Verify all top-level fields are populated
        assert!(!report.timestamp.is_empty());
        assert!(!report.system.hostname.is_empty() || report.system.hostname == "unknown");
        assert!(!report.system.os.is_empty());
        assert!(!report.models.is_empty());
        assert_eq!(report.models.len(), TOTAL_MODEL_COUNT);
    }

    // ========================================================================
    // Test 2: System Info Populated
    // ========================================================================

    #[test]
    fn test_system_info_populated() {
        let system_info = SystemInfo::gather();

        // Hostname should be non-empty (or "unknown" fallback)
        assert!(!system_info.hostname.is_empty());

        // OS should contain the current OS
        assert!(system_info.os.contains(std::env::consts::OS));
        assert!(system_info.os.contains(std::env::consts::ARCH));
    }

    // ========================================================================
    // Test 3: GPU Diagnostics From Info
    // ========================================================================

    #[test]
    fn test_gpu_diagnostics_from_info() {
        let gpu_info = GpuInfo::new(
            0,
            "NVIDIA GeForce RTX 5090".to_string(),
            (12, 0),
            32 * 1024 * 1024 * 1024, // 32GB
            "13.1.0".to_string(),
        );

        let diagnostics = GpuDiagnostics::from_gpu_info(&gpu_info, 8 * 1024 * 1024 * 1024);

        assert_eq!(diagnostics.device_id, 0);
        assert_eq!(diagnostics.name, "NVIDIA GeForce RTX 5090");
        assert_eq!(diagnostics.compute_capability, "12.0");
        assert_eq!(diagnostics.total_vram_bytes, 32 * 1024 * 1024 * 1024);
        assert_eq!(diagnostics.available_vram_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(diagnostics.driver_version, "13.1.0");
    }

    // ========================================================================
    // Test 4: Memory Diagnostics From Pools
    // ========================================================================

    #[test]
    fn test_memory_diagnostics_from_pools() {
        let config = test_config();
        let loader = WarmLoader::new(config).expect("Failed to create loader");

        // The loader starts with no allocations
        let report = WarmDiagnostics::generate_report(&loader);

        assert_eq!(
            report.memory.model_pool_capacity_bytes,
            24 * 1024 * 1024 * 1024
        );
        assert_eq!(report.memory.model_pool_used_bytes, 0);
        assert_eq!(
            report.memory.working_pool_capacity_bytes,
            8 * 1024 * 1024 * 1024
        );
        assert_eq!(report.memory.total_allocations, 0);
    }

    // ========================================================================
    // Test 5: Model Diagnostic Warm
    // ========================================================================

    #[test]
    fn test_model_diagnostic_warm() {
        let state = WarmModelState::Warm;
        let handle_info = Some((0x7f8000000000u64, 629145600usize, 0xDEADBEEFu64));

        let diagnostic = ModelDiagnostic::from_state("E1_Semantic", &state, 629145600, handle_info);

        assert_eq!(diagnostic.model_id, "E1_Semantic");
        assert_eq!(diagnostic.state, "Warm");
        assert_eq!(diagnostic.expected_bytes, 629145600);
        assert_eq!(diagnostic.allocated_bytes, Some(629145600));
        assert_eq!(diagnostic.vram_ptr, Some("0x00007f8000000000".to_string()));
        assert_eq!(diagnostic.checksum, Some("0x00000000DEADBEEF".to_string()));
        assert!(diagnostic.error_message.is_none());
    }

    // ========================================================================
    // Test 6: Model Diagnostic Failed
    // ========================================================================

    #[test]
    fn test_model_diagnostic_failed() {
        let state = WarmModelState::Failed {
            error_code: 102,
            error_message: "CUDA allocation failed".to_string(),
        };

        let diagnostic = ModelDiagnostic::from_state("E1_Semantic", &state, 629145600, None);

        assert_eq!(diagnostic.model_id, "E1_Semantic");
        assert_eq!(diagnostic.state, "Failed");
        assert_eq!(diagnostic.expected_bytes, 629145600);
        assert!(diagnostic.allocated_bytes.is_none());
        assert!(diagnostic.vram_ptr.is_none());
        assert!(diagnostic.checksum.is_none());
        assert_eq!(
            diagnostic.error_message,
            Some("CUDA allocation failed".to_string())
        );
    }

    // ========================================================================
    // Test 7: JSON Serialization
    // ========================================================================

    #[test]
    fn test_json_serialization() {
        let config = test_config();
        let loader = WarmLoader::new(config).expect("Failed to create loader");

        let json = WarmDiagnostics::to_json(&loader).expect("Failed to serialize to JSON");

        // Verify it's valid JSON by parsing it
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("Failed to parse JSON");

        // Verify expected fields exist
        assert!(parsed.get("timestamp").is_some());
        assert!(parsed.get("system").is_some());
        assert!(parsed.get("memory").is_some());
        assert!(parsed.get("models").is_some());
        assert!(parsed.get("errors").is_some());

        // Verify models is an array with correct count
        let models = parsed.get("models").unwrap().as_array().unwrap();
        assert_eq!(models.len(), TOTAL_MODEL_COUNT);
    }

    // ========================================================================
    // Test 8: Status Line Format
    // ========================================================================

    #[test]
    fn test_status_line_format() {
        let config = test_config();
        let loader = WarmLoader::new(config).expect("Failed to create loader");

        let status = WarmDiagnostics::status_line(&loader);

        // Should contain expected elements
        assert!(status.contains("WARM:"));
        assert!(status.contains("models"));
        assert!(status.contains("VRAM"));

        // Initial state should show LOADING since nothing is warm yet
        assert!(status.contains("LOADING: 0/13"));
    }

    // ========================================================================
    // Additional Tests
    // ========================================================================

    #[test]
    fn test_error_diagnostic_from_error() {
        let error = WarmError::CudaInitFailed {
            cuda_error: "Driver not found".to_string(),
            driver_version: String::new(),
            gpu_name: String::new(),
        };

        let diagnostic = ErrorDiagnostic::from_error(&error);

        assert_eq!(diagnostic.error_code, "ERR-WARM-CUDA-INIT");
        assert_eq!(diagnostic.category, "CUDA");
        assert_eq!(diagnostic.exit_code, 106);
        assert!(diagnostic.message.contains("Driver not found"));
    }

    #[test]
    fn test_report_warm_and_failed_counts() {
        let mut report = WarmDiagnosticReport::empty();

        report.models.push(ModelDiagnostic {
            model_id: "model1".to_string(),
            state: "Warm".to_string(),
            expected_bytes: 1000,
            allocated_bytes: Some(1000),
            vram_ptr: None,
            checksum: None,
            error_message: None,
        });

        report.models.push(ModelDiagnostic {
            model_id: "model2".to_string(),
            state: "Failed".to_string(),
            expected_bytes: 1000,
            allocated_bytes: None,
            vram_ptr: None,
            checksum: None,
            error_message: Some("Error".to_string()),
        });

        report.models.push(ModelDiagnostic {
            model_id: "model3".to_string(),
            state: "Warm".to_string(),
            expected_bytes: 1000,
            allocated_bytes: Some(1000),
            vram_ptr: None,
            checksum: None,
            error_message: None,
        });

        assert_eq!(report.warm_count(), 2);
        assert_eq!(report.failed_count(), 1);
    }

    #[test]
    fn test_report_has_errors() {
        let mut report = WarmDiagnosticReport::empty();
        assert!(!report.has_errors());

        report.errors.push(ErrorDiagnostic {
            error_code: "ERR-TEST".to_string(),
            category: "TEST".to_string(),
            message: "Test error".to_string(),
            exit_code: 1,
        });

        assert!(report.has_errors());
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0B");
        assert_eq!(format_bytes(512), "512B");
        assert_eq!(format_bytes(1024), "1.00KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00GB");
        assert_eq!(format_bytes(32 * 1024 * 1024 * 1024), "32.00GB");
    }

    #[test]
    fn test_model_diagnostic_loading_state() {
        let state = WarmModelState::Loading {
            progress_percent: 75,
            bytes_loaded: 500_000_000,
        };

        let diagnostic = ModelDiagnostic::from_state("E1_Semantic", &state, 629145600, None);

        assert_eq!(diagnostic.state, "Loading (75%)");
    }

    #[test]
    fn test_model_diagnostic_validating_state() {
        let state = WarmModelState::Validating;

        let diagnostic = ModelDiagnostic::from_state("E1_Semantic", &state, 629145600, None);

        assert_eq!(diagnostic.state, "Validating");
    }

    #[test]
    fn test_full_report_with_warm_models() {
        let config = test_config();
        let loader = WarmLoader::new(config).expect("Failed to create loader");

        // Manually transition all models to Warm for testing
        {
            let mut registry = loader.registry().write().unwrap();
            for model_id in EMBEDDING_MODEL_IDS.iter().chain(std::iter::once(&FUSEMOE_MODEL_ID)) {
                registry.start_loading(model_id).unwrap();
                registry.mark_validating(model_id).unwrap();
                let handle = ModelHandle::new(0x1000, 1024, 0, 0xDEAD);
                registry.mark_warm(model_id, handle).unwrap();
            }
        }

        let report = WarmDiagnostics::generate_report(&loader);

        assert_eq!(report.warm_count(), TOTAL_MODEL_COUNT);
        assert_eq!(report.failed_count(), 0);
        assert!(!report.has_errors());

        // Verify status line shows OK
        let status = WarmDiagnostics::status_line(&loader);
        assert!(status.contains("OK"));
        assert!(status.contains("13/13"));
    }

    #[test]
    fn test_timestamp_format() {
        let report = WarmDiagnosticReport::empty();

        // Should be in ISO 8601 format: YYYY-MM-DDTHH:MM:SS.mmmZ
        assert!(report.timestamp.contains('T'));
        assert!(report.timestamp.ends_with('Z'));
        assert_eq!(report.timestamp.len(), 24); // "2025-01-03T12:00:00.000Z"
    }
}
