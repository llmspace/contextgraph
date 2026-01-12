//! Integration tests for BootstrapService

use crate::autonomous::services::bootstrap_service::*;
use crate::autonomous::BootstrapConfig;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_bootstrap_from_documents_with_real_files() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    // Create test documents
    let readme_content = r#"
# Project Overview

The goal of this project is to build an intelligent context graph system.

## Features

This system provides memory management and learning capabilities.

## Mission

Our mission is to enable machines to understand and remember context effectively.
"#;

    let constitution_content = r#"
name: Context Graph
version: 1.0.0
purpose: Create a unified memory system for AI applications
objectives:
  - Implement efficient vector storage
  - Enable semantic search
  - Provide context-aware retrieval
"#;

    // Write files
    let readme_path = temp_dir.path().join("README.md");
    let mut readme_file = File::create(&readme_path).expect("Failed to create README");
    readme_file
        .write_all(readme_content.as_bytes())
        .expect("Failed to write README");

    let const_path = temp_dir.path().join("constitution.yaml");
    let mut const_file = File::create(&const_path).expect("Failed to create constitution");
    const_file
        .write_all(constitution_content.as_bytes())
        .expect("Failed to write constitution");

    // Run bootstrap
    let mut service = BootstrapService::new();
    let results = service.bootstrap_from_documents(temp_dir.path());

    assert!(!results.is_empty(), "Should find at least one goal");

    let best = &results[0];
    assert!(!best.goal_text.is_empty());
    assert!(best.confidence > 0.0);
    assert!(!best.extracted_from.is_empty());

    println!("[PASS] test_bootstrap_from_documents_with_real_files");
}

#[test]
fn test_bootstrap_from_documents_empty_directory() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    let mut service = BootstrapService::new();
    let results = service.bootstrap_from_documents(temp_dir.path());

    assert!(
        results.is_empty(),
        "Empty directory should yield no results"
    );

    println!("[PASS] test_bootstrap_from_documents_empty_directory");
}

#[test]
fn test_bootstrap_from_documents_respects_max_docs() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    // Create more files than max_docs
    for i in 0..5 {
        let path = temp_dir.path().join(format!("doc{}.md", i));
        let mut file = File::create(&path).expect("Failed to create file");
        file.write_all(format!("The goal of document {} is to test limits.", i).as_bytes())
            .expect("Failed to write");
    }

    let config = BootstrapServiceConfig {
        doc_dir: temp_dir.path().to_path_buf(),
        file_extensions: vec!["md".into()],
        max_docs: 2, // Limit to 2 docs
        bootstrap_config: BootstrapConfig::default(),
    };

    let mut service = BootstrapService::with_config(config);
    let _results = service.bootstrap_from_documents(temp_dir.path());

    // Cache should only have entries for max_docs files
    assert!(
        service.results_cache().len() <= 2,
        "Should respect max_docs limit"
    );

    println!("[PASS] test_bootstrap_from_documents_respects_max_docs");
}

#[test]
fn test_bootstrap_from_documents_filters_by_extension() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    // Create files with different extensions
    let md_path = temp_dir.path().join("doc.md");
    let mut md_file = File::create(&md_path).expect("Failed to create md");
    md_file
        .write_all(b"The goal is in markdown.")
        .expect("Failed to write md");

    let rs_path = temp_dir.path().join("doc.rs");
    let mut rs_file = File::create(&rs_path).expect("Failed to create rs");
    rs_file
        .write_all(b"// The goal is in rust.")
        .expect("Failed to write rs");

    let config = BootstrapServiceConfig {
        doc_dir: temp_dir.path().to_path_buf(),
        file_extensions: vec!["md".into()], // Only .md files
        max_docs: 10,
        bootstrap_config: BootstrapConfig::default(),
    };

    let mut service = BootstrapService::with_config(config);
    let _results = service.bootstrap_from_documents(temp_dir.path());

    // Should only process .md files
    assert!(
        service
            .results_cache()
            .keys()
            .all(|p| p.extension().map(|e| e == "md").unwrap_or(false)),
        "Should only process .md files"
    );

    println!("[PASS] test_bootstrap_from_documents_filters_by_extension");
}

#[test]
#[should_panic(expected = "Document directory does not exist")]
fn test_bootstrap_from_documents_fails_nonexistent_dir() {
    let mut service = BootstrapService::new();
    service.bootstrap_from_documents(std::path::Path::new("/nonexistent/path/12345"));
}

#[test]
fn test_bootstrap_caches_results() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    let doc_path = temp_dir.path().join("test.md");
    let mut file = File::create(&doc_path).expect("Failed to create file");
    file.write_all(b"The purpose is to test caching.")
        .expect("Failed to write");

    let mut service = BootstrapService::new();
    let _ = service.bootstrap_from_documents(temp_dir.path());

    assert!(
        !service.results_cache().is_empty(),
        "Results should be cached"
    );

    println!("[PASS] test_bootstrap_caches_results");
}

#[test]
fn test_recursive_directory_scanning() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let sub_dir = temp_dir.path().join("subdir");
    fs::create_dir(&sub_dir).expect("Failed to create subdir");

    // File in root
    let root_path = temp_dir.path().join("root.md");
    let mut root_file = File::create(&root_path).expect("Failed to create root file");
    root_file
        .write_all(b"The goal in root directory.")
        .expect("Failed to write");

    // File in subdirectory
    let sub_path = sub_dir.join("nested.md");
    let mut sub_file = File::create(&sub_path).expect("Failed to create sub file");
    sub_file
        .write_all(b"The purpose in nested directory.")
        .expect("Failed to write");

    let mut service = BootstrapService::new();
    let _ = service.bootstrap_from_documents(temp_dir.path());

    // Should have found files in both directories
    assert!(
        service.results_cache().len() >= 2,
        "Should scan subdirectories recursively"
    );

    println!("[PASS] test_recursive_directory_scanning");
}
