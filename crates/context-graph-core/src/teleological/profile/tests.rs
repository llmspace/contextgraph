//! Tests for teleological profile components.

#[cfg(test)]
mod tests {
    use crate::teleological::profile::{
        FusionStrategy, ProfileMetrics, TaskType, TeleologicalProfile,
    };
    use crate::teleological::types::NUM_EMBEDDERS;

    // ===== FusionStrategy Tests =====

    #[test]
    fn test_fusion_strategy_default() {
        assert_eq!(FusionStrategy::default(), FusionStrategy::WeightedAverage);

        println!("[PASS] FusionStrategy::default is WeightedAverage");
    }

    #[test]
    fn test_fusion_strategy_tucker_default() {
        let strategy = FusionStrategy::tucker_default();

        match strategy {
            FusionStrategy::TuckerDecomposition { ranks } => {
                assert_eq!(ranks, FusionStrategy::DEFAULT_TUCKER_RANKS);
            }
            _ => panic!("Expected TuckerDecomposition"),
        }

        println!("[PASS] tucker_default uses correct ranks");
    }

    #[test]
    fn test_fusion_strategy_attention_default() {
        let strategy = FusionStrategy::attention_default();

        match strategy {
            FusionStrategy::Attention { heads } => {
                assert_eq!(heads, FusionStrategy::DEFAULT_ATTENTION_HEADS);
            }
            _ => panic!("Expected Attention"),
        }

        println!("[PASS] attention_default uses correct heads");
    }

    #[test]
    fn test_fusion_strategy_cost() {
        assert!(FusionStrategy::PrimaryOnly.cost() < FusionStrategy::WeightedAverage.cost());
        assert!(FusionStrategy::WeightedAverage.cost() < FusionStrategy::tucker_default().cost());

        println!("[PASS] FusionStrategy costs are ordered correctly");
    }

    #[test]
    fn test_fusion_strategy_serialization() {
        let strategy = FusionStrategy::TuckerDecomposition { ranks: (2, 3, 64) };
        let json = serde_json::to_string(&strategy).unwrap();
        let deserialized: FusionStrategy = serde_json::from_str(&json).unwrap();

        assert_eq!(strategy, deserialized);

        println!("[PASS] FusionStrategy serialization works");
    }

    // ===== TaskType Tests =====

    #[test]
    fn test_task_type_default() {
        assert_eq!(TaskType::default(), TaskType::General);

        println!("[PASS] TaskType::default is General");
    }

    #[test]
    fn test_task_type_all() {
        assert_eq!(TaskType::ALL.len(), 8);

        println!("[PASS] TaskType::ALL contains 8 types");
    }

    #[test]
    fn test_task_type_primary_embedders() {
        let code = TaskType::CodeSearch.primary_embedders();
        assert!(code.contains(&5)); // E6
        assert!(code.contains(&6)); // E7

        let semantic = TaskType::SemanticSearch.primary_embedders();
        assert!(semantic.contains(&0)); // E1
        assert!(semantic.contains(&4)); // E5

        println!("[PASS] primary_embedders match teleoplan.md");
    }

    #[test]
    fn test_task_type_suggested_strategy() {
        assert_eq!(
            TaskType::CodeSearch.suggested_strategy(),
            FusionStrategy::PrimaryOnly
        );

        match TaskType::SocialSearch.suggested_strategy() {
            FusionStrategy::Attention { .. } => {}
            _ => panic!("Expected Attention for SocialSearch"),
        }

        println!("[PASS] suggested_strategy returns appropriate strategies");
    }

    #[test]
    fn test_task_type_display() {
        assert_eq!(format!("{}", TaskType::CodeSearch), "Code");
        assert_eq!(format!("{}", TaskType::General), "General");

        println!("[PASS] TaskType Display works");
    }

    // ===== ProfileMetrics Tests =====

    #[test]
    fn test_profile_metrics_default() {
        let metrics = ProfileMetrics::default();

        assert!((metrics.mrr - 0.0).abs() < f32::EPSILON);
        assert_eq!(metrics.retrieval_count, 0);

        println!("[PASS] ProfileMetrics::default creates zeros");
    }

    #[test]
    fn test_profile_metrics_quality_score() {
        let metrics = ProfileMetrics::new(0.8, 0.7, 0.9, 0.6, 0.85);

        // 0.3 * 0.8 + 0.3 * 0.9 + 0.4 * 0.85 = 0.24 + 0.27 + 0.34 = 0.85
        assert!((metrics.quality_score() - 0.85).abs() < 0.001);

        println!("[PASS] quality_score computes correctly");
    }

    #[test]
    fn test_profile_metrics_f1() {
        let metrics = ProfileMetrics::new(0.0, 0.0, 0.8, 0.0, 0.6);

        // F1 = 2 * 0.6 * 0.8 / (0.6 + 0.8) = 0.96 / 1.4 = 0.6857
        assert!((metrics.f1_at_10() - 0.6857).abs() < 0.01);

        // Zero case
        let zero_metrics = ProfileMetrics::default();
        assert!((zero_metrics.f1_at_10() - 0.0).abs() < f32::EPSILON);

        println!("[PASS] f1_at_10 computes correctly");
    }

    #[test]
    fn test_profile_metrics_update_ewma() {
        let mut metrics = ProfileMetrics::new(0.5, 0.5, 0.5, 0.5, 0.5);

        metrics.update_ewma(1.0, 1.0, 1.0, 1.0, 1.0, 50.0, 0.1);

        // After EWMA: 0.1 * 1.0 + 0.9 * 0.5 = 0.55
        assert!((metrics.mrr - 0.55).abs() < 0.001);
        assert_eq!(metrics.retrieval_count, 1);

        println!("[PASS] update_ewma applies EWMA correctly");
    }

    // ===== TeleologicalProfile Tests =====

    #[test]
    fn test_profile_new() {
        let profile = TeleologicalProfile::new("test", "Test Profile", TaskType::General);

        assert_eq!(profile.id.as_str(), "test");
        assert_eq!(profile.name, "Test Profile");
        assert_eq!(profile.task_type, TaskType::General);
        assert_eq!(profile.sample_count, 0);
        assert!(!profile.is_system);

        // Weights should be uniform
        let expected_weight = 1.0 / NUM_EMBEDDERS as f32;
        for &w in profile.embedding_weights.iter() {
            assert!((w - expected_weight).abs() < 0.001);
        }

        println!("[PASS] TeleologicalProfile::new creates valid profile");
    }

    #[test]
    fn test_profile_system() {
        let profile = TeleologicalProfile::system(TaskType::CodeSearch);

        assert!(profile.is_system);
        assert!(profile.id.as_str().contains("system"));

        // Primary embedders should have higher weights
        let primary = TaskType::CodeSearch.primary_embedders();
        for &idx in primary {
            assert!(
                profile.embedding_weights[idx] > 0.1,
                "Primary embedder {} should have high weight",
                idx
            );
        }

        println!("[PASS] TeleologicalProfile::system creates optimized profile");
    }

    #[test]
    fn test_profile_code_implementation() {
        let profile = TeleologicalProfile::code_implementation();

        // Check specific weights from teleoplan.md
        assert!((profile.embedding_weights[5] - 0.25).abs() < 0.001); // E6
        assert!((profile.embedding_weights[6] - 0.18).abs() < 0.001); // E7

        println!("[PASS] code_implementation matches teleoplan.md");
    }

    #[test]
    fn test_profile_conceptual_research() {
        let profile = TeleologicalProfile::conceptual_research();

        // Check specific weights from teleoplan.md
        assert!((profile.embedding_weights[10] - 0.20).abs() < 0.001); // E11

        match profile.fusion_strategy {
            FusionStrategy::Hierarchical => {}
            _ => panic!("Expected Hierarchical fusion"),
        }

        println!("[PASS] conceptual_research matches teleoplan.md");
    }

    #[test]
    fn test_profile_normalize_weights() {
        let mut profile = TeleologicalProfile::new("test", "Test", TaskType::General);

        // Set non-normalized weights
        profile.embedding_weights = [1.0; NUM_EMBEDDERS];
        profile.normalize_weights();

        let sum: f32 = profile.embedding_weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        println!("[PASS] normalize_weights sums to 1.0");
    }

    #[test]
    fn test_profile_get_set_weight() {
        let mut profile = TeleologicalProfile::new("test", "Test", TaskType::General);

        profile.set_weight(5, 0.5);
        assert!((profile.get_weight(5) - 0.5).abs() < f32::EPSILON);

        println!("[PASS] get/set weight work correctly");
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_profile_get_weight_out_of_bounds() {
        let profile = TeleologicalProfile::new("test", "Test", TaskType::General);
        let _ = profile.get_weight(13);
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_profile_set_negative_weight() {
        let mut profile = TeleologicalProfile::new("test", "Test", TaskType::General);
        profile.set_weight(0, -0.1);
    }

    #[test]
    fn test_profile_top_embedders() {
        let profile = TeleologicalProfile::code_implementation();
        let top3 = profile.top_embedders(3);

        assert_eq!(top3.len(), 3);
        assert!(top3.contains(&5)); // E6 has highest weight (0.25)
        assert!(top3.contains(&6)); // E7 has second highest (0.18)

        println!("[PASS] top_embedders returns highest weighted");
    }

    #[test]
    fn test_profile_similarity() {
        let p1 = TeleologicalProfile::code_implementation();
        let p2 = TeleologicalProfile::code_implementation();

        let sim = p1.similarity(&p2);
        assert!((sim - 1.0).abs() < 0.001);

        let p3 = TeleologicalProfile::conceptual_research();
        let sim2 = p1.similarity(&p3);
        assert!(sim2 < 0.95); // Should be different

        println!("[PASS] profile similarity works correctly");
    }

    #[test]
    fn test_profile_serialization() {
        let profile = TeleologicalProfile::code_implementation();

        let json = serde_json::to_string(&profile).unwrap();
        let deserialized: TeleologicalProfile = serde_json::from_str(&json).unwrap();

        assert_eq!(profile.id, deserialized.id);
        assert_eq!(profile.embedding_weights, deserialized.embedding_weights);

        println!("[PASS] Profile serialization roundtrip works");
    }

    #[test]
    fn test_profile_with_description() {
        let profile = TeleologicalProfile::new("test", "Test", TaskType::General)
            .with_description("A test profile");

        assert_eq!(profile.description, Some("A test profile".to_string()));

        println!("[PASS] with_description works");
    }
}
