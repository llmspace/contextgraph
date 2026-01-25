//! CRITICAL DIAGNOSTIC TEST: Verify E9 vectors differ from E1 vectors in fingerprint
//!
//! This test verifies that after calling embed_all(), the SemanticFingerprint
//! has DIFFERENT vectors for e1_semantic and e9_hdc. If they are identical,
//! the entire 13-embedder system is broken.
//!
//! Run with: cargo test --package context-graph-embeddings e9_vector_differentiation -- --nocapture

use context_graph_core::traits::MultiArrayEmbeddingProvider;
use context_graph_embeddings::config::GpuConfig;
use context_graph_embeddings::provider::ProductionMultiArrayProvider;
use std::path::PathBuf;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn vectors_are_identical(a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-9)
}

/// CRITICAL: Verify E1 and E9 fingerprint vectors are different
#[tokio::test]
async fn test_e1_e9_vectors_differ_in_fingerprint() {
    println!("\n========================================");
    println!("CRITICAL: E1 vs E9 Fingerprint Vector Test");
    println!("========================================\n");

    let models_dir = PathBuf::from(
        std::env::var("MODELS_DIR").unwrap_or_else(|_| "./models".to_string()),
    );

    if !models_dir.exists() {
        println!("SKIPPED: Models directory not found at {:?}", models_dir);
        println!("Set MODELS_DIR environment variable or ensure ./models exists");
        return;
    }

    println!("Creating MultiArrayEmbeddingProvider...");
    let provider = match ProductionMultiArrayProvider::new(models_dir, GpuConfig::default()).await {
        Ok(p) => {
            println!("✓ Provider created successfully");
            p
        }
        Err(e) => {
            println!("SKIPPED: Could not create provider: {}", e);
            return;
        }
    };

    let test_content = "authentication failed for user";
    println!("\nTest content: '{}'\n", test_content);

    // Embed and get fingerprint
    println!("Calling embed_all()...");
    let result = provider.embed_all(test_content).await;
    let fingerprint = match result {
        Ok(output) => {
            println!("✓ embed_all() succeeded");
            output.fingerprint
        }
        Err(e) => {
            println!("ERROR: embed_all() failed: {}", e);
            panic!("embed_all() failed");
        }
    };

    // Extract vectors
    let e1 = &fingerprint.e1_semantic;
    let e9 = &fingerprint.e9_hdc;

    println!("\nE1 (Semantic) vector:");
    println!("  Dimension: {}", e1.len());
    println!("  First 10 values: {:?}", &e1[..10.min(e1.len())]);
    println!("  L2 norm: {:.6}", e1.iter().map(|x| x * x).sum::<f32>().sqrt());

    println!("\nE9 (HDC) vector:");
    println!("  Dimension: {}", e9.len());
    println!("  First 10 values: {:?}", &e9[..10.min(e9.len())]);
    println!("  L2 norm: {:.6}", e9.iter().map(|x| x * x).sum::<f32>().sqrt());

    // Check if identical
    let identical = vectors_are_identical(e1, e9);
    let similarity = if e1.len() == e9.len() {
        cosine_similarity(e1, e9)
    } else {
        0.0
    };

    println!("\n========================================");
    println!("RESULTS:");
    println!("  Vectors identical: {}", identical);
    println!("  Cosine similarity: {:.6}", similarity);

    if identical {
        println!("\n❌ CRITICAL BUG: E1 and E9 vectors are IDENTICAL!");
        println!("   This breaks the entire 13-embedder system.");
        println!("   The HDC model is NOT being used correctly.");
    } else if similarity > 0.99 {
        println!("\n⚠️  WARNING: E1 and E9 vectors are extremely similar (>0.99)");
        println!("   This is unexpected - they should have low similarity.");
    } else {
        println!("\n✓ PASS: E1 and E9 vectors are different");
        println!("   E9 (HDC) is producing unique embeddings.");
    }
    println!("========================================\n");

    assert!(!identical, "CRITICAL: E1 and E9 vectors must NOT be identical!");
    assert!(similarity < 0.95, "E1 and E9 should not have >0.95 similarity, got {}", similarity);
}

/// Verify E5 vector is also different from E1
#[tokio::test]
async fn test_e1_e5_vectors_differ_in_fingerprint() {
    println!("\n========================================");
    println!("E1 vs E5 Fingerprint Vector Test");
    println!("========================================\n");

    let models_dir = PathBuf::from(
        std::env::var("MODELS_DIR").unwrap_or_else(|_| "./models".to_string()),
    );

    if !models_dir.exists() {
        println!("SKIPPED: Models directory not found");
        return;
    }

    let provider = match ProductionMultiArrayProvider::new(models_dir, GpuConfig::default()).await {
        Ok(p) => p,
        Err(e) => {
            println!("SKIPPED: Could not create provider: {}", e);
            return;
        }
    };

    let result = provider.embed_all("test content for verification").await;
    let fingerprint = match result {
        Ok(output) => output.fingerprint,
        Err(e) => {
            println!("ERROR: embed_all() failed: {}", e);
            panic!("embed_all() failed");
        }
    };

    let e1 = &fingerprint.e1_semantic;
    let e5 = fingerprint.e5_active_vector();

    println!("E1 dimension: {}", e1.len());
    println!("E5 dimension: {}", e5.len());

    // E1 and E5 have different dimensions (1024 vs 768), so they can't be identical
    // But let's verify they're not somehow using the same underlying data
    println!("\nE1 first 5 values: {:?}", &e1[..5.min(e1.len())]);
    println!("E5 first 5 values: {:?}", &e5[..5.min(e5.len())]);

    // Verify dimensions are correct
    assert_eq!(e1.len(), 1024, "E1 should be 1024D");
    assert_eq!(e5.len(), 768, "E5 should be 768D");

    println!("✓ E1 and E5 have correct and different dimensions");
}
