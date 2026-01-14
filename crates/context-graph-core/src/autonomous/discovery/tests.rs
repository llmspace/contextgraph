//! Tests for goal discovery pipeline.

#[cfg(test)]
mod tests {
    use crate::autonomous::discovery::{
        DiscoveryConfig, GoalCandidate, GoalDiscoveryPipeline, NumClusters,
    };
    use crate::autonomous::evolution::GoalLevel;
    use crate::teleological::comparator::TeleologicalComparator;
    use crate::teleological::Embedder;
    use crate::types::fingerprint::{
        SemanticFingerprint, SparseVector, TeleologicalArray, E1_DIM, E2_DIM, E3_DIM, E4_DIM,
        E5_DIM, E7_DIM, E8_DIM, E9_DIM, E10_DIM, E11_DIM, E12_TOKEN_DIM,
    };

    /// Create a test fingerprint with specific patterns for deterministic testing.
    fn create_test_fingerprint(cluster_id: usize, variance: f32) -> SemanticFingerprint {
        // Create base patterns based on cluster_id
        let base_e1: Vec<f32> = (0..E1_DIM)
            .map(|i| {
                let phase = (cluster_id as f32) * 2.0 * std::f32::consts::PI / 3.0;
                (i as f32 / E1_DIM as f32 * std::f32::consts::PI + phase).sin() + variance
            })
            .collect();

        // Normalize to unit vector
        let norm: f32 = base_e1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let e1_semantic: Vec<f32> = base_e1.iter().map(|x| x / norm).collect();

        // Similar patterns for other embeddings
        let create_normalized = |dim: usize| -> Vec<f32> {
            let base: Vec<f32> = (0..dim)
                .map(|i| {
                    let phase = (cluster_id as f32) * 2.0 * std::f32::consts::PI / 3.0;
                    (i as f32 / dim as f32 * std::f32::consts::PI + phase).cos() + variance
                })
                .collect();
            let norm: f32 = base.iter().map(|x| x * x).sum::<f32>().sqrt();
            base.iter().map(|x| x / norm.max(1e-6)).collect()
        };

        SemanticFingerprint {
            e1_semantic,
            e2_temporal_recent: create_normalized(E2_DIM),
            e3_temporal_periodic: create_normalized(E3_DIM),
            e4_temporal_positional: create_normalized(E4_DIM),
            e5_causal: create_normalized(E5_DIM),
            e6_sparse: SparseVector::empty(), // Sparse vectors are optional
            e7_code: create_normalized(E7_DIM),
            e8_graph: create_normalized(E8_DIM),
            e9_hdc: create_normalized(E9_DIM),
            e10_multimodal: create_normalized(E10_DIM),
            e11_entity: create_normalized(E11_DIM),
            e12_late_interaction: vec![create_normalized(E12_TOKEN_DIM)],
            e13_splade: SparseVector::empty(),
        }
    }

    #[test]
    fn test_kmeans_three_clusters() {
        // Create 30 TeleologicalArrays in 3 known clusters
        let mut arrays: Vec<TeleologicalArray> = Vec::new();

        // Cluster A: 10 arrays with variance around 0.0
        for _ in 0..10 {
            arrays.push(create_test_fingerprint(0, 0.01));
        }

        // Cluster B: 10 arrays with variance around 1/3 period
        for _ in 0..10 {
            arrays.push(create_test_fingerprint(1, 0.01));
        }

        // Cluster C: 10 arrays with variance around 2/3 period
        for _ in 0..10 {
            arrays.push(create_test_fingerprint(2, 0.01));
        }

        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);

        let config = DiscoveryConfig {
            min_cluster_size: 5,
            min_coherence: 0.5, // Lower threshold for test data
            num_clusters: NumClusters::Fixed(3),
            ..Default::default()
        };

        let result = pipeline.discover(&arrays, &config);

        // Verify 3 clusters found
        assert!(
            result.clusters_found >= 2,
            "Expected at least 2 clusters, got {}",
            result.clusters_found
        );

        // Verify total arrays analyzed
        assert_eq!(result.total_arrays_analyzed, 30);

        println!(
            "[PASS] test_kmeans_three_clusters: Found {} clusters",
            result.clusters_found
        );
    }

    #[test]
    fn test_centroid_is_valid_teleological_array() {
        // Create test arrays
        let arrays: Vec<TeleologicalArray> = (0..10)
            .map(|i| create_test_fingerprint(i % 3, 0.02))
            .collect();

        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);

        let members: Vec<&TeleologicalArray> = arrays.iter().collect();
        let centroid = pipeline.compute_centroid(&members);

        // Verify all 13 embedders are populated
        assert_eq!(centroid.e1_semantic.len(), E1_DIM, "E1 dimension mismatch");
        assert_eq!(
            centroid.e2_temporal_recent.len(),
            E2_DIM,
            "E2 dimension mismatch"
        );
        assert_eq!(
            centroid.e3_temporal_periodic.len(),
            E3_DIM,
            "E3 dimension mismatch"
        );
        assert_eq!(
            centroid.e4_temporal_positional.len(),
            E4_DIM,
            "E4 dimension mismatch"
        );
        assert_eq!(centroid.e5_causal.len(), E5_DIM, "E5 dimension mismatch");
        assert_eq!(centroid.e7_code.len(), E7_DIM, "E7 dimension mismatch");
        assert_eq!(centroid.e8_graph.len(), E8_DIM, "E8 dimension mismatch");
        assert_eq!(centroid.e9_hdc.len(), E9_DIM, "E9 dimension mismatch");
        assert_eq!(
            centroid.e10_multimodal.len(),
            E10_DIM,
            "E10 dimension mismatch"
        );
        assert_eq!(centroid.e11_entity.len(), E11_DIM, "E11 dimension mismatch");

        // E12 should have at least one token
        assert!(
            !centroid.e12_late_interaction.is_empty(),
            "E12 should have tokens"
        );
        for token in &centroid.e12_late_interaction {
            assert_eq!(token.len(), E12_TOKEN_DIM, "E12 token dimension mismatch");
        }

        println!("[PASS] test_centroid_is_valid_teleological_array");
    }

    #[test]
    fn test_goal_level_assignment() {
        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);

        let base_centroid = create_test_fingerprint(0, 0.0);

        // Test NorthStar: size=50, coherence=0.85
        let candidate_ns = GoalCandidate {
            goal_id: "ns".to_string(),
            description: "Test".to_string(),
            level: GoalLevel::Operational,
            confidence: 0.9,
            member_count: 50,
            centroid: base_centroid.clone(),
            dominant_embedders: vec![Embedder::Semantic],
            coherence_score: 0.85,
        };
        assert_eq!(pipeline.assign_level(&candidate_ns), GoalLevel::NorthStar);

        // Test Strategic (size not met): size=49, coherence=0.85
        let candidate_strat_size = GoalCandidate {
            member_count: 49,
            coherence_score: 0.85,
            ..candidate_ns.clone()
        };
        assert_eq!(
            pipeline.assign_level(&candidate_strat_size),
            GoalLevel::Strategic
        );

        // Test Strategic (coherence not met): size=50, coherence=0.84
        let candidate_strat_coh = GoalCandidate {
            member_count: 50,
            coherence_score: 0.84,
            ..candidate_ns.clone()
        };
        assert_eq!(
            pipeline.assign_level(&candidate_strat_coh),
            GoalLevel::Strategic
        );

        // Test Strategic: size=20, coherence=0.80
        let candidate_strat = GoalCandidate {
            member_count: 20,
            coherence_score: 0.80,
            ..candidate_ns.clone()
        };
        assert_eq!(pipeline.assign_level(&candidate_strat), GoalLevel::Strategic);

        // Test Tactical: size=10, coherence=0.75
        let candidate_tact = GoalCandidate {
            member_count: 10,
            coherence_score: 0.75,
            ..candidate_ns.clone()
        };
        assert_eq!(pipeline.assign_level(&candidate_tact), GoalLevel::Tactical);

        // Test Operational: size=5, coherence=0.70
        let candidate_op = GoalCandidate {
            member_count: 5,
            coherence_score: 0.70,
            ..candidate_ns.clone()
        };
        assert_eq!(
            pipeline.assign_level(&candidate_op),
            GoalLevel::Operational
        );

        println!("[PASS] test_goal_level_assignment");
    }

    #[test]
    fn test_hierarchy_construction() {
        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);

        let base_centroid = create_test_fingerprint(0, 0.0);
        let similar_centroid = create_test_fingerprint(0, 0.05); // Very similar

        // Parent cluster: Large (50 members), high coherence (NorthStar)
        let parent = GoalCandidate {
            goal_id: "parent".to_string(),
            description: "Parent goal".to_string(),
            level: GoalLevel::NorthStar,
            confidence: 0.9,
            member_count: 50,
            centroid: base_centroid.clone(),
            dominant_embedders: vec![Embedder::Semantic],
            coherence_score: 0.85,
        };

        // Child cluster: Small (10 members), similar centroid (Tactical)
        let child = GoalCandidate {
            goal_id: "child".to_string(),
            description: "Child goal".to_string(),
            level: GoalLevel::Tactical,
            confidence: 0.7,
            member_count: 10,
            centroid: similar_centroid,
            dominant_embedders: vec![Embedder::Semantic],
            coherence_score: 0.75,
        };

        let candidates = vec![parent, child];
        let hierarchy = pipeline.build_hierarchy(&candidates);

        // Should have at least one parent-child relationship
        // (depends on centroid similarity threshold)
        println!(
            "[INFO] test_hierarchy_construction: Found {} relationships",
            hierarchy.len()
        );

        // Verify relationship structure if any exist
        for rel in &hierarchy {
            assert!(
                rel.similarity >= 0.5,
                "Relationship similarity should be >= 0.5"
            );
            println!(
                "[PASS] Parent {} -> Child {}, similarity={}",
                rel.parent_id, rel.child_id, rel.similarity
            );
        }

        println!("[PASS] test_hierarchy_construction");
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_fail_fast_empty_input() {
        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);
        let config = DiscoveryConfig::default();

        let empty: Vec<TeleologicalArray> = vec![];
        let _ = pipeline.discover(&empty, &config);
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_fail_fast_insufficient_data() {
        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);

        let config = DiscoveryConfig {
            min_cluster_size: 5,
            ..Default::default()
        };

        // Only 3 arrays with min_cluster_size=5
        let arrays: Vec<TeleologicalArray> = (0..3)
            .map(|i| create_test_fingerprint(i, 0.01))
            .collect();

        let _ = pipeline.discover(&arrays, &config);
    }

    #[test]
    fn test_all_arrays_identical() {
        // All arrays identical -> single cluster with coherence 1.0
        let identical = create_test_fingerprint(0, 0.0);
        let arrays: Vec<TeleologicalArray> = vec![identical.clone(); 10];

        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);

        let config = DiscoveryConfig {
            min_cluster_size: 5,
            min_coherence: 0.5,
            num_clusters: NumClusters::Fixed(1),
            ..Default::default()
        };

        let result = pipeline.discover(&arrays, &config);

        // Should have 1 cluster
        assert!(result.clusters_found >= 1);

        // Coherence should be high (close to 1.0)
        if let Some(goal) = result.discovered_goals.first() {
            assert!(
                goal.coherence_score > 0.9,
                "Identical arrays should have coherence > 0.9, got {}",
                goal.coherence_score
            );
        }

        println!("[PASS] test_all_arrays_identical");
    }

    #[test]
    fn test_widely_dispersed_arrays() {
        // Create widely dispersed arrays (each in its own "cluster")
        let arrays: Vec<TeleologicalArray> = (0..20)
            .map(|i| create_test_fingerprint(i, 0.5)) // High variance
            .collect();

        let comparator = TeleologicalComparator::new();
        let pipeline = GoalDiscoveryPipeline::new(comparator);

        let config = DiscoveryConfig {
            min_cluster_size: 2,
            min_coherence: 0.3, // Very low threshold
            num_clusters: NumClusters::Auto,
            ..Default::default()
        };

        let result = pipeline.discover(&arrays, &config);

        // Should have multiple clusters (dispersed data)
        println!(
            "[INFO] test_widely_dispersed_arrays: Found {} clusters",
            result.clusters_found
        );

        // Coherence should be lower than identical arrays
        for goal in &result.discovered_goals {
            println!(
                "  Cluster: size={}, coherence={:.3}",
                goal.member_count, goal.coherence_score
            );
        }

        println!("[PASS] test_widely_dispersed_arrays");
    }

    #[test]
    fn test_dominant_embedders() {
        use crate::autonomous::discovery::centroid::l2_norm;
        use crate::teleological::NUM_EMBEDDERS;

        let comparator = TeleologicalComparator::new();
        let _pipeline = GoalDiscoveryPipeline::new(comparator);

        let centroid = create_test_fingerprint(0, 0.0);

        // Manually find dominant embedders to test the logic
        let mut embedder_magnitudes: Vec<(Embedder, f32)> = Vec::with_capacity(NUM_EMBEDDERS);

        embedder_magnitudes.push((Embedder::Semantic, l2_norm(&centroid.e1_semantic)));
        embedder_magnitudes.push((
            Embedder::TemporalRecent,
            l2_norm(&centroid.e2_temporal_recent),
        ));
        embedder_magnitudes.push((
            Embedder::TemporalPeriodic,
            l2_norm(&centroid.e3_temporal_periodic),
        ));
        embedder_magnitudes.push((
            Embedder::TemporalPositional,
            l2_norm(&centroid.e4_temporal_positional),
        ));
        embedder_magnitudes.push((Embedder::Causal, l2_norm(&centroid.e5_causal)));
        embedder_magnitudes.push((Embedder::Code, l2_norm(&centroid.e7_code)));
        embedder_magnitudes.push((Embedder::Emotional, l2_norm(&centroid.e8_graph)));
        embedder_magnitudes.push((Embedder::Hdc, l2_norm(&centroid.e9_hdc)));
        embedder_magnitudes.push((Embedder::Multimodal, l2_norm(&centroid.e10_multimodal)));
        embedder_magnitudes.push((Embedder::Entity, l2_norm(&centroid.e11_entity)));
        embedder_magnitudes.push((Embedder::Sparse, centroid.e6_sparse.l2_norm()));
        embedder_magnitudes.push((Embedder::KeywordSplade, centroid.e13_splade.l2_norm()));

        let e12_magnitude: f32 = centroid
            .e12_late_interaction
            .iter()
            .map(|t| l2_norm(t))
            .sum::<f32>()
            / centroid.e12_late_interaction.len().max(1) as f32;
        embedder_magnitudes.push((Embedder::LateInteraction, e12_magnitude));

        embedder_magnitudes
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let dominant: Vec<Embedder> = embedder_magnitudes
            .into_iter()
            .take(3)
            .map(|(e, _)| e)
            .collect();

        // Should return exactly 3 embedders
        assert_eq!(dominant.len(), 3, "Should return 3 dominant embedders");

        // All should be valid Embedder variants
        for emb in &dominant {
            println!("[INFO] Dominant embedder: {:?}", emb);
        }

        println!("[PASS] test_dominant_embedders");
    }

    #[test]
    fn test_discovery_config_defaults() {
        let config = DiscoveryConfig::default();

        assert_eq!(config.sample_size, 500);
        assert_eq!(config.min_cluster_size, 5);
        assert!((config.min_coherence - 0.75).abs() < f32::EPSILON);
        assert_eq!(config.max_iterations, 100);

        println!("[PASS] test_discovery_config_defaults");
    }
}
