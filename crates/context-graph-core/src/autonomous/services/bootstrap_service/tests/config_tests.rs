//! Configuration tests for BootstrapService

use crate::autonomous::services::bootstrap_service::*;
use crate::autonomous::BootstrapConfig;
use std::path::PathBuf;

#[test]
fn test_bootstrap_service_config_default() {
    let config = BootstrapServiceConfig::default();

    assert_eq!(config.doc_dir, PathBuf::from("."));
    assert_eq!(config.max_docs, 100);
    assert!(!config.file_extensions.is_empty());
    assert!(config.file_extensions.contains(&"md".to_string()));

    println!("[PASS] test_bootstrap_service_config_default");
}

#[test]
fn test_bootstrap_service_config_custom() {
    let config = BootstrapServiceConfig {
        doc_dir: PathBuf::from("/custom/path"),
        file_extensions: vec!["rst".into(), "adoc".into()],
        max_docs: 50,
        bootstrap_config: BootstrapConfig::default(),
    };

    assert_eq!(config.doc_dir, PathBuf::from("/custom/path"));
    assert_eq!(config.max_docs, 50);
    assert_eq!(config.file_extensions.len(), 2);

    println!("[PASS] test_bootstrap_service_config_custom");
}

#[test]
fn test_config_accessor() {
    let service = BootstrapService::new();
    let config = service.config();

    assert_eq!(config.max_docs, 100);
    assert!(!config.file_extensions.is_empty());

    println!("[PASS] test_config_accessor");
}

#[test]
fn test_results_cache_accessor() {
    let service = BootstrapService::new();
    let cache = service.results_cache();

    assert!(cache.is_empty());

    println!("[PASS] test_results_cache_accessor");
}
