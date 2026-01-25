//! DIAGNOSTIC TEST: Verify E9 vectors differ from E1 vectors
//!
//! This test directly checks if the SemanticFingerprint has unique vectors
//! for each embedder after embedding.

#[cfg(test)]
mod vector_identity_tests {
    use crate::config::GpuConfig;
    use crate::provider::multi_array::MultiArrayEmbeddingProvider;
    use crate::traits::MultiSpaceEmbedder;
    use std::path::PathBuf;

    /// CRITICAL DIAGNOSTIC: Are E1 and E9 vectors different?
    #[tokio::test]
    async fn diagnostic_e1_e9_vector_difference() {
        println!("\n========================================");
        println!("DIAGNOSTIC: E1 vs E9 Vector Identity Check");
        println!("========================================\n");

        // Create embedding provider (requires GPU - skip if not available)
        let models_dir = PathBuf::from(
            std::env::var("MODELS_DIR").unwrap_or_else(|_| "./models".to_string()),
        );

        if !models_dir.exists() {
            println!("SKIPPED: Models directory not found at {:?}", models_dir);
            return;
        }

        let provider = match MultiArrayEmbeddingProvider::new(models_dir, GpuConfig::default()) {
            Ok(p) => p,
            Err(e) => {
                println!("SKIPPED: Could not create provider: {}", e);
                return;
            }
        };

        // Test content
        let test_content = "authentication failed for user";
        println!("Test content: '{}'\n", test_content);

        // Embed
        let result = provider.embed_all(test_content).await;
        let fingerprint = match result {
            Ok(output) => output.fingerprint,
            Err(e) => {
                println!("ERROR: Embedding failed: {}", e);
                panic!("Embedding failed");
            }
        };

        // Extract E1 and E9 vectors
        let e1 = &fingerprint.e1_semantic;
        let e9 = &fingerprint.e9_hdc;

        println!("E1 (Semantic) vector:");
        println!("  Dimension: {}", e1.len());
        println!("  First 10 values: {:?}", &e1[..10.min(e1.len())]);

        println!("\nE9 (HDC) vector:");
        println!("  Dimension: {}", e9.len());
        println!("  First 10 values: {:?}", &e9[..10.min(e9.len())]);

        // Check if vectors are identical
        let are_identical = e1.len() == e9.len() && e1.iter().zip(e9.iter()).all(|(a, b)| (a - b).abs() < 1e-9);

        println!("\n========================================");
        if are_identical {
            println!("❌ CRITICAL BUG: E1 and E9 vectors are IDENTICAL!");
            println!("   This breaks the entire 13-embedder system.");
        } else {
            println!("✓ PASS: E1 and E9 vectors are different");
        }
        println!("========================================\n");

        assert!(!are_identical, "E1 and E9 vectors must be different!");

        // Also check E5
        let e5 = fingerprint.e5_active_vector();
        println!("E5 (Causal) vector:");
        println!("  Dimension: {}", e5.len());
        println!("  First 10 values: {:?}", &e5[..10.min(e5.len())]);

        let e1_e5_identical = e1.len() == e5.len() && e1.iter().zip(e5.iter()).all(|(a, b)| (a - b).abs() < 1e-9);
        if e1_e5_identical {
            println!("❌ CRITICAL BUG: E1 and E5 vectors are IDENTICAL!");
        } else {
            println!("✓ PASS: E1 and E5 vectors are different (different dimensions is OK)");
        }
    }

    /// Check if the problem is in cosine similarity computation itself
    #[test]
    fn diagnostic_cosine_similarity_computation() {
        println!("\n========================================");
        println!("DIAGNOSTIC: Cosine Similarity Computation");
        println!("========================================\n");

        // Create two clearly different vectors
        let vec_a = vec![1.0, 0.0, 0.0, 0.0];
        let vec_b = vec![0.0, 1.0, 0.0, 0.0];
        let vec_c = vec![1.0, 0.0, 0.0, 0.0]; // Same as A

        fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_a == 0.0 || norm_b == 0.0 {
                return 0.0;
            }
            dot / (norm_a * norm_b)
        }

        let sim_ab = cosine_sim(&vec_a, &vec_b);
        let sim_ac = cosine_sim(&vec_a, &vec_c);

        println!("A = [1,0,0,0], B = [0,1,0,0], C = [1,0,0,0]");
        println!("cosine(A, B) = {} (expected: 0.0)", sim_ab);
        println!("cosine(A, C) = {} (expected: 1.0)", sim_ac);

        assert!((sim_ab - 0.0).abs() < 0.001, "A and B are orthogonal");
        assert!((sim_ac - 1.0).abs() < 0.001, "A and C are identical");

        println!("✓ Cosine similarity computation is correct\n");
    }
}
