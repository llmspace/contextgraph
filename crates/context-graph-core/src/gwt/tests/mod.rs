//! Tests for GwtSystem and related functionality
//!
//! This module contains all integration tests for the GWT system including:
//! - System creation and basic functionality
//! - Kuramoto network integration
//! - Self-awareness loop activation
//! - Event wiring integration
//! - Edge cases and concurrent access

mod common;
mod system_tests;
mod kuramoto_tests;
mod awareness_tests;
mod awareness_edge_tests;
mod event_tests;
