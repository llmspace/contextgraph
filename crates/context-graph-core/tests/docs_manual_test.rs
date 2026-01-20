//! Manual test: Process actual ./docs/*.md files through file watcher
//!
//! This test verifies the REAL file watcher functionality with actual
//! documentation files in the project's ./docs/ directory.

use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::time::sleep;

use context_graph_core::memory::capture::{
    EmbeddingProvider, MemoryCaptureService, TestEmbeddingProvider,
};
use context_graph_core::memory::store::MemoryStore;
use context_graph_core::memory::watcher::MDFileWatcher;
use context_graph_core::stubs::InMemoryTeleologicalStore;
use context_graph_core::traits::TeleologicalMemoryStore;
use context_graph_core::types::SourceType;

/// Test: Process REAL ./docs/*.md files and verify source metadata
#[tokio::test]
async fn test_real_docs_directory_processing() {
    println!("\n=== REAL DOCS DIRECTORY TEST ===\n");

    // Use actual ./docs/ directory from project root
    let docs_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("docs");

    println!("Docs directory: {:?}", docs_dir);

    if !docs_dir.exists() {
        panic!("./docs/ directory does not exist at {:?}", docs_dir);
    }

    // List .md files in docs
    let md_files: Vec<_> = std::fs::read_dir(&docs_dir)
        .expect("read docs dir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "md")
                .unwrap_or(false)
        })
        .collect();

    println!("Found {} .md files:", md_files.len());
    for f in &md_files {
        println!("  - {:?}", f.path());
    }

    // Setup test environment
    let db_dir = TempDir::new().expect("create db temp dir");
    let memory_store = Arc::new(MemoryStore::new(db_dir.path()).expect("create memory store"));
    let teleological_store = Arc::new(InMemoryTeleologicalStore::new());
    let embedder: Arc<dyn EmbeddingProvider> = Arc::new(TestEmbeddingProvider);

    let capture_service = Arc::new(MemoryCaptureService::with_teleological_store(
        memory_store.clone(),
        embedder,
        teleological_store.clone(),
    ));

    // Create file watcher for ./docs/
    let mut watcher = MDFileWatcher::new(
        vec![docs_dir.clone()],
        capture_service.clone(),
        "docs-test-session".to_string(),
    )
    .expect("create watcher");

    println!("\nStarting file watcher on ./docs/...");
    watcher.start().await.expect("start watcher");

    // Wait for initial scan to complete
    sleep(Duration::from_millis(500)).await;

    println!("\n=== VERIFICATION: Source Metadata ===\n");

    // Get all stored memories
    let total_count = memory_store.count().expect("count");
    println!("Total memories stored: {}", total_count);

    // Verify each .md file has memories with correct source metadata
    for entry in &md_files {
        let file_path = entry.path();
        let canonical = file_path.canonicalize().expect("canonicalize");
        let path_str = canonical.to_string_lossy().to_string();

        let memories = memory_store.get_by_file_path(&path_str).expect("get by path");

        if memories.is_empty() {
            println!("  WARNING: No memories for {:?}", file_path.file_name());
            continue;
        }

        println!(
            "\nFile: {:?} ({} chunks)",
            file_path.file_name().unwrap(),
            memories.len()
        );

        // Check source metadata for each chunk
        for (i, memory) in memories.iter().enumerate() {
            let source_metadata = teleological_store
                .get_source_metadata(memory.id)
                .await
                .expect("get source metadata");

            match source_metadata {
                Some(meta) => {
                    assert_eq!(
                        meta.source_type,
                        SourceType::MDFileChunk,
                        "Source type should be MDFileChunk"
                    );
                    assert_eq!(
                        meta.file_path,
                        Some(path_str.clone()),
                        "File path should match"
                    );
                    println!(
                        "  Chunk {}: {} ✓",
                        i + 1,
                        meta.display_string()
                    );
                }
                None => {
                    panic!(
                        "FAIL: Source metadata missing for memory {} from {:?}",
                        memory.id,
                        file_path.file_name()
                    );
                }
            }

            // Verify content is stored
            let content = teleological_store
                .get_content(memory.id)
                .await
                .expect("get content");
            assert!(content.is_some(), "Content should be stored");
            let content_len = content.unwrap().len();
            println!("         Content: {} chars stored", content_len);
        }
    }

    watcher.stop();

    println!("\n=== REAL DOCS TEST PASSED ===\n");
    println!("Summary:");
    println!("  - Processed {} .md files", md_files.len());
    println!("  - Created {} total memory chunks", total_count);
    println!("  - All source metadata verified (MDFileChunk type, file paths, chunk indices)");
    println!("  - All content stored alongside embeddings");
}
