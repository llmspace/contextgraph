//! Construction tests for BootstrapService

use crate::autonomous::services::bootstrap_service::*;
use crate::autonomous::BootstrapConfig;
use std::path::PathBuf;

#[test]
fn test_bootstrap_service_new() {
    let service = BootstrapService::new();

    assert_eq!(service.config().max_docs, 100);
    assert!(service.results_cache().is_empty());

    println!("[PASS] test_bootstrap_service_new");
}

#[test]
fn test_bootstrap_service_with_valid_config() {
    let config = BootstrapServiceConfig {
        doc_dir: PathBuf::from("."),
        file_extensions: vec!["md".into()],
        max_docs: 10,
        bootstrap_config: BootstrapConfig::default(),
    };

    let service = BootstrapService::with_config(config);
    assert_eq!(service.config().max_docs, 10);

    println!("[PASS] test_bootstrap_service_with_valid_config");
}

#[test]
#[should_panic(expected = "max_docs must be greater than 0")]
fn test_bootstrap_service_fails_zero_max_docs() {
    let config = BootstrapServiceConfig {
        doc_dir: PathBuf::from("."),
        file_extensions: vec!["md".into()],
        max_docs: 0,
        bootstrap_config: BootstrapConfig::default(),
    };

    BootstrapService::with_config(config);
}

#[test]
#[should_panic(expected = "file_extensions cannot be empty")]
fn test_bootstrap_service_fails_empty_extensions() {
    let config = BootstrapServiceConfig {
        doc_dir: PathBuf::from("."),
        file_extensions: vec![],
        max_docs: 10,
        bootstrap_config: BootstrapConfig::default(),
    };

    BootstrapService::with_config(config);
}

#[test]
fn test_default_trait_implementation() {
    let service = BootstrapService::default();

    assert_eq!(service.config().max_docs, 100);
    assert!(service.results_cache().is_empty());

    println!("[PASS] test_default_trait_implementation");
}
