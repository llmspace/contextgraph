//! Tests for k-means clustering.
//!
//! Comprehensive test suite for clustering functionality.
//!
//! # Test Organization
//!
//! - `helpers` - Common test utilities and data generators
//! - `config_tests` - KMeansConfig validation tests
//! - `cluster_tests` - PurposeCluster and ClusteringResult tests
//! - `clustering_tests` - Main StandardKMeans clustering tests
//! - `edge_cases` - Edge case and boundary condition tests

mod helpers;

mod cluster_tests;
mod clustering_tests;
mod config_tests;
mod edge_cases;
