//! GPU device utility functions.
//!
//! Internal utilities for GPU information queries and formatting.

use candle_core::Device;

use crate::gpu::GpuInfo;

/// Query GPU information from the device.
///
/// # Implementation Note
///
/// Candle doesn't expose detailed GPU info via cuDeviceGetAttribute or similar.
/// For RTX 5090 (Blackwell GB202), we use the known specifications.
/// Future versions may query actual hardware via cuda-sys bindings.
pub(crate) fn query_gpu_info(_device: &Device) -> GpuInfo {
    // TODO: When cuda-sys is available, query actual device properties:
    // - cuDeviceGetName
    // - cuDeviceTotalMem
    // - cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR)
    //
    // For now, use RTX 5090 target specifications
    GpuInfo {
        name: "NVIDIA GeForce RTX 5090".to_string(),
        total_vram: 32 * 1024 * 1024 * 1024,    // 32GB GDDR7
        compute_capability: "12.0".to_string(), // Blackwell SM_120
        available: true,
    }
}

/// Format bytes as human-readable string.
pub(crate) fn format_bytes(bytes: usize) -> String {
    const GB: usize = 1024 * 1024 * 1024;
    const MB: usize = 1024 * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes_zero() {
        assert_eq!(format_bytes(0), "0 bytes");
    }

    #[test]
    fn test_format_bytes_megabytes() {
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(512 * 1024 * 1024), "512.0 MB");
    }

    #[test]
    fn test_format_bytes_gigabytes() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_bytes(32 * 1024 * 1024 * 1024), "32.0 GB");
    }

    #[test]
    fn test_format_bytes_small() {
        assert_eq!(format_bytes(100), "100 bytes");
        assert_eq!(format_bytes(1023), "1023 bytes");
    }
}
