//! Full State Verification Tests
//!
//! # CRITICAL: Physical Database Verification
//!
//! This test module performs FULL STATE VERIFICATION by:
//! 1. Executing storage operations (writes, updates, deletes)
//! 2. Immediately performing SEPARATE read operations on the Source of Truth (RocksDB)
//! 3. Verifying the exact bytes stored match expectations
//! 4. Testing boundary and edge cases with physical inspection
//!
//! # Evidence of Success
//!
//! Each test provides:
//! - Hexdump of actual data in RocksDB
//! - Comparison between expected and actual values
//! - Physical key inspection
//! - Column family state verification
//!
//! # NO MOCKS - NO FALLBACKS
//!
//! All operations use REAL RocksDB databases in temp directories.
//! Failure is fatal and provides detailed diagnostics.

mod column_family_tests;
mod edge_case_tests;
mod helpers;
mod persistence_tests;
mod write_read_tests;

// Re-export for test discovery
pub use helpers::*;
