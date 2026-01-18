//! Tests for goal hierarchy types.

#[cfg(test)]
mod tests {
    use crate::purpose::goals::{
        DiscoveryMethod, GoalDiscoveryMetadata, GoalHierarchy, GoalHierarchyError, GoalLevel,
        GoalNode, GoalNodeError,
    };
    use crate::types::fingerprint::{SemanticFingerprint, ValidationError};
    use chrono::Utc;
    use uuid::Uuid;

    // Helper function to create a valid zeroed fingerprint for testing
    fn test_fingerprint() -> SemanticFingerprint {
        SemanticFingerprint::zeroed()
    }

    // Helper function to create bootstrap discovery metadata
    fn test_discovery() -> GoalDiscoveryMetadata {
        GoalDiscoveryMetadata::bootstrap()
    }

    // Helper function to create clustering discovery metadata
    fn clustering_discovery(
        confidence: f32,
        cluster_size: usize,
        coherence: f32,
    ) -> Result<GoalDiscoveryMetadata, GoalNodeError> {
        GoalDiscoveryMetadata::new(
            DiscoveryMethod::Clustering,
            confidence,
            cluster_size,
            coherence,
        )
    }

    // TASK-P0-001: Updated for 3-level hierarchy (Strategic → Tactical → Immediate)
    #[test]
    fn test_goal_level_propagation_weights() {
        // Strategic is now top-level with weight 1.0
        assert_eq!(GoalLevel::Strategic.propagation_weight(), 1.0);
        assert_eq!(GoalLevel::Tactical.propagation_weight(), 0.6);
        assert_eq!(GoalLevel::Immediate.propagation_weight(), 0.3);
        println!("[VERIFIED] GoalLevel propagation weights match constitution.yaml");
    }

    // TASK-P0-001: Updated depths for 3-level hierarchy
    #[test]
    fn test_goal_level_depth() {
        assert_eq!(GoalLevel::Strategic.depth(), 0); // Top-level
        assert_eq!(GoalLevel::Tactical.depth(), 1);
        assert_eq!(GoalLevel::Immediate.depth(), 2);
        println!("[VERIFIED] GoalLevel depth values are correct");
    }

    #[test]
    fn test_discovery_metadata_valid() {
        let discovery = clustering_discovery(0.85, 42, 0.78).unwrap();
        assert_eq!(discovery.method, DiscoveryMethod::Clustering);
        assert_eq!(discovery.confidence, 0.85);
        assert_eq!(discovery.cluster_size, 42);
        assert_eq!(discovery.coherence, 0.78);
        println!("[VERIFIED] GoalDiscoveryMetadata::new creates valid metadata");
    }

    #[test]
    fn test_discovery_metadata_bootstrap() {
        let discovery = GoalDiscoveryMetadata::bootstrap();
        assert_eq!(discovery.method, DiscoveryMethod::Bootstrap);
        assert_eq!(discovery.confidence, 0.0);
        assert_eq!(discovery.cluster_size, 0);
        assert_eq!(discovery.coherence, 0.0);
        println!("[VERIFIED] GoalDiscoveryMetadata::bootstrap creates correct defaults");
    }

    #[test]
    fn test_discovery_metadata_invalid_confidence() {
        let result = clustering_discovery(1.5, 10, 0.8);
        assert!(matches!(result, Err(GoalNodeError::InvalidConfidence(1.5))));

        let result = clustering_discovery(-0.1, 10, 0.8);
        assert!(matches!(result, Err(GoalNodeError::InvalidConfidence(_))));
        println!("[VERIFIED] GoalDiscoveryMetadata rejects invalid confidence");
    }

    #[test]
    fn test_discovery_metadata_invalid_coherence() {
        let result = clustering_discovery(0.8, 10, 1.5);
        assert!(matches!(result, Err(GoalNodeError::InvalidCoherence(1.5))));

        let result = clustering_discovery(0.8, 10, -0.1);
        assert!(matches!(result, Err(GoalNodeError::InvalidCoherence(_))));
        println!("[VERIFIED] GoalDiscoveryMetadata rejects invalid coherence");
    }

    #[test]
    fn test_discovery_metadata_empty_cluster() {
        let result = clustering_discovery(0.8, 0, 0.7);
        assert!(matches!(result, Err(GoalNodeError::EmptyCluster)));
        println!("[VERIFIED] GoalDiscoveryMetadata rejects empty cluster for non-Bootstrap");
    }

    #[test]
    fn test_goal_node_autonomous_creation() {
        let fp = test_fingerprint();
        let discovery = test_discovery();

        let goal = GoalNode::autonomous_goal(
            "Test Strategic Goal".into(),
            GoalLevel::Strategic,
            fp,
            discovery,
        )
        .expect("Should create goal");

        assert_eq!(goal.level, GoalLevel::Strategic);
        assert!(goal.parent_id.is_none());
        assert!(goal.child_ids.is_empty());
        assert!(goal.is_top_level());
        println!("[VERIFIED] GoalNode::autonomous_goal creates correct structure");
    }

    // TASK-P0-001: Updated - child goals must be Tactical or Immediate, not Strategic
    #[test]
    fn test_goal_node_child_creation() {
        let fp = test_fingerprint();
        let discovery = test_discovery();
        let parent_id = Uuid::new_v4();

        // Child goals must be Tactical or Immediate (not Strategic)
        let child = GoalNode::child_goal(
            "Test Tactical Goal".into(),
            GoalLevel::Tactical,
            parent_id,
            fp,
            discovery,
        )
        .expect("Should create child goal");

        assert_eq!(child.level, GoalLevel::Tactical);
        assert_eq!(child.parent_id, Some(parent_id));
        assert!(!child.is_top_level());
        println!("[VERIFIED] GoalNode::child_goal creates correct structure");
    }

    // TASK-P0-001: Updated - Strategic goals cannot have a parent (they are top-level)
    #[test]
    #[should_panic(expected = "Strategic goals cannot have a parent")]
    fn test_child_goal_cannot_be_strategic() {
        let fp = test_fingerprint();
        let discovery = test_discovery();
        let parent_id = Uuid::new_v4();

        // Strategic goals cannot be created as children
        let _ = GoalNode::child_goal(
            "Bad goal".into(),
            GoalLevel::Strategic, // Should panic
            parent_id,
            fp,
            discovery,
        );
    }

    #[test]
    fn test_goal_node_invalid_fingerprint() {
        let mut fp = test_fingerprint();
        fp.e1_semantic = vec![]; // Invalid - empty

        let discovery = test_discovery();
        let result = GoalNode::autonomous_goal("Test".into(), GoalLevel::Strategic, fp, discovery);

        assert!(result.is_err());
        assert!(matches!(result, Err(GoalNodeError::InvalidArray(_))));
        println!("[VERIFIED] GoalNode rejects invalid teleological array");
    }

    #[test]
    fn test_goal_node_array_access() {
        let fp = test_fingerprint();
        let discovery = test_discovery();

        let goal =
            GoalNode::autonomous_goal("Test".into(), GoalLevel::Strategic, fp, discovery).unwrap();

        let array = goal.array();
        assert_eq!(array.e1_semantic.len(), 1024);
        println!("[VERIFIED] GoalNode::array() provides access to teleological array");
    }

    #[test]
    fn test_goal_node_child_management() {
        let fp = test_fingerprint();
        let discovery = test_discovery();

        let mut goal =
            GoalNode::autonomous_goal("Test".into(), GoalLevel::Strategic, fp, discovery).unwrap();

        let child_id = Uuid::new_v4();
        goal.add_child(child_id);
        assert!(goal.child_ids.contains(&child_id));
        assert_eq!(goal.child_ids.len(), 1);

        // Adding same child again should not duplicate
        goal.add_child(child_id);
        assert_eq!(goal.child_ids.len(), 1);

        goal.remove_child(child_id);
        assert!(!goal.child_ids.contains(&child_id));
        println!("[VERIFIED] GoalNode child management works correctly");
    }

    #[test]
    fn test_goal_hierarchy_multiple_strategic() {
        let mut hierarchy = GoalHierarchy::new();

        let s1 = GoalNode::autonomous_goal(
            "Strategic1".into(),
            GoalLevel::Strategic,
            test_fingerprint(),
            test_discovery(),
        )
        .unwrap();

        let s2 = GoalNode::autonomous_goal(
            "Strategic2".into(),
            GoalLevel::Strategic,
            test_fingerprint(),
            test_discovery(),
        )
        .unwrap();

        // TASK-P0-001/ARCH-03: Multiple Strategic goals are now allowed
        assert!(hierarchy.add_goal(s1).is_ok());
        let result = hierarchy.add_goal(s2);
        assert!(
            result.is_ok(),
            "Multiple Strategic goals should be allowed per ARCH-03"
        );
        assert_eq!(hierarchy.top_level_goals().len(), 2);
        println!("[VERIFIED] GoalHierarchy allows multiple Strategic goals (ARCH-03)");
    }

    // TASK-P0-001: Updated - use Tactical level for child goal test
    #[test]
    fn test_goal_hierarchy_parent_validation() {
        let mut hierarchy = GoalHierarchy::new();

        let fake_parent = Uuid::new_v4();
        // Use Tactical (not Strategic) since Strategic goals are top-level
        let child = GoalNode::child_goal(
            "Orphan".into(),
            GoalLevel::Tactical,
            fake_parent,
            test_fingerprint(),
            test_discovery(),
        )
        .unwrap();

        let result = hierarchy.add_goal(child);
        assert!(matches!(result, Err(GoalHierarchyError::ParentNotFound(_))));
        println!("[VERIFIED] GoalHierarchy validates parent existence");
    }

    // TASK-P0-001: Updated for 3-level hierarchy (Strategic → Tactical → Immediate)
    #[test]
    fn test_goal_hierarchy_full_tree() {
        let mut hierarchy = GoalHierarchy::new();

        // Add Strategic goal (top-level, no parent)
        let strategic = GoalNode::autonomous_goal(
            "Master ML".into(),
            GoalLevel::Strategic,
            test_fingerprint(),
            clustering_discovery(0.9, 100, 0.85).unwrap(),
        )
        .unwrap();
        let strategic_id = strategic.id;
        hierarchy.add_goal(strategic).unwrap();

        // Add Tactical child (of Strategic)
        let tactical = GoalNode::child_goal(
            "Learn PyTorch".into(),
            GoalLevel::Tactical,
            strategic_id,
            test_fingerprint(),
            clustering_discovery(0.8, 50, 0.75).unwrap(),
        )
        .unwrap();
        let tactical_id = tactical.id;
        hierarchy.add_goal(tactical).unwrap();

        // Add Immediate child (of Tactical)
        let immediate = GoalNode::child_goal(
            "Complete tutorial".into(),
            GoalLevel::Immediate,
            tactical_id,
            test_fingerprint(),
            clustering_discovery(0.7, 20, 0.65).unwrap(),
        )
        .unwrap();
        let immediate_id = immediate.id;
        hierarchy.add_goal(immediate).unwrap();

        assert_eq!(hierarchy.len(), 3);
        assert!(!hierarchy.is_empty());
        assert!(hierarchy.has_top_level_goals());
        assert!(hierarchy.top_level_goals().first().is_some());
        assert_eq!(hierarchy.at_level(GoalLevel::Strategic).len(), 1);
        assert_eq!(hierarchy.at_level(GoalLevel::Tactical).len(), 1);
        assert_eq!(hierarchy.at_level(GoalLevel::Immediate).len(), 1);
        assert_eq!(hierarchy.children(&strategic_id).len(), 1);
        assert!(hierarchy.validate().is_ok());

        let path = hierarchy.path_to_root(&immediate_id);
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], immediate_id);
        assert_eq!(path[1], tactical_id);
        assert_eq!(path[2], strategic_id);

        println!("[VERIFIED] GoalHierarchy full tree structure works correctly");
    }

    // TASK-P0-001/ARCH-03: Empty hierarchies are now valid - goals emerge autonomously
    #[test]
    fn test_goal_hierarchy_validate_empty_is_valid() {
        let hierarchy = GoalHierarchy::new();

        // ARCH-03: Empty hierarchies are valid - goals emerge autonomously
        let result = hierarchy.validate();
        assert!(
            result.is_ok(),
            "Empty hierarchy should be valid per ARCH-03"
        );
        assert!(!hierarchy.has_top_level_goals());
        println!("[VERIFIED] Empty hierarchy is valid (ARCH-03)");
    }

    #[test]
    fn test_goal_hierarchy_validate_with_strategic_goal() {
        let mut hierarchy = GoalHierarchy::new();

        // Add a Strategic goal (no parent - top level)
        let goal = GoalNode {
            id: Uuid::new_v4(),
            description: "Strategic Goal".into(),
            level: GoalLevel::Strategic,
            teleological_array: test_fingerprint(),
            parent_id: None,
            child_ids: vec![],
            discovery: test_discovery(),
            created_at: Utc::now(),
        };
        hierarchy.nodes.insert(goal.id, goal);

        let result = hierarchy.validate();
        assert!(
            result.is_ok(),
            "Hierarchy with Strategic goal should be valid"
        );
        assert!(hierarchy.has_top_level_goals());
        println!("[VERIFIED] Hierarchy with Strategic goal is valid");
    }

    // TASK-P0-001: Updated for 3-level hierarchy
    #[test]
    fn test_goal_hierarchy_iter() {
        let mut hierarchy = GoalHierarchy::new();

        let strategic = GoalNode::autonomous_goal(
            "Strategic".into(),
            GoalLevel::Strategic,
            test_fingerprint(),
            test_discovery(),
        )
        .unwrap();
        let strategic_id = strategic.id;
        hierarchy.add_goal(strategic).unwrap();

        // Use Tactical (not Strategic) as child
        let tactical = GoalNode::child_goal(
            "Tactical".into(),
            GoalLevel::Tactical,
            strategic_id,
            test_fingerprint(),
            test_discovery(),
        )
        .unwrap();
        hierarchy.add_goal(tactical).unwrap();

        let count = hierarchy.iter().count();
        assert_eq!(count, 2);
        println!("[VERIFIED] GoalHierarchy iter works correctly");
    }

    #[test]
    fn test_goal_serialization_roundtrip() {
        let fp = test_fingerprint();
        let discovery = clustering_discovery(0.85, 42, 0.78).unwrap();

        let goal =
            GoalNode::autonomous_goal("Test goal".into(), GoalLevel::Strategic, fp, discovery)
                .unwrap();

        // Serialize
        let json = serde_json::to_string(&goal).expect("Serialize");

        // Deserialize
        let restored: GoalNode = serde_json::from_str(&json).expect("Deserialize");

        // Verify
        assert_eq!(goal.id, restored.id);
        assert_eq!(goal.level, restored.level);
        assert_eq!(goal.description, restored.description);
        assert_eq!(
            goal.teleological_array.e1_semantic.len(),
            restored.teleological_array.e1_semantic.len()
        );
        println!("[VERIFIED] GoalNode survives JSON serialization roundtrip");
    }

    #[test]
    fn test_discovery_method_serialization() {
        let methods = vec![
            DiscoveryMethod::Clustering,
            DiscoveryMethod::PatternRecognition,
            DiscoveryMethod::Decomposition,
            DiscoveryMethod::Bootstrap,
        ];

        for method in methods {
            let json = serde_json::to_string(&method).expect("Serialize");
            let restored: DiscoveryMethod = serde_json::from_str(&json).expect("Deserialize");
            assert_eq!(method, restored);
        }
        println!("[VERIFIED] DiscoveryMethod serialization works correctly");
    }

    // Edge Case Tests from Task Spec

    #[test]
    fn test_edge_case_incomplete_fingerprint_rejected() {
        let mut fp = test_fingerprint();
        fp.e1_semantic = vec![]; // Invalid - empty

        let discovery = test_discovery();
        let result = GoalNode::autonomous_goal("Test".into(), GoalLevel::Strategic, fp, discovery);

        assert!(result.is_err());
        match result {
            Err(GoalNodeError::InvalidArray(ValidationError::DimensionMismatch { .. })) => {
                println!("[EDGE CASE 1 PASSED] Incomplete fingerprint rejected");
            }
            _ => panic!("Wrong error type: {:?}", result),
        }
    }

    #[test]
    fn test_edge_case_invalid_confidence_rejected() {
        let result = GoalDiscoveryMetadata::new(
            DiscoveryMethod::Clustering,
            1.5, // Invalid
            10,
            0.8,
        );

        assert!(matches!(result, Err(GoalNodeError::InvalidConfidence(1.5))));
        println!("[EDGE CASE 2 PASSED] Invalid confidence rejected");
    }

    // TASK-P0-001/ARCH-03: Multiple Strategic goals are now allowed
    #[test]
    fn test_edge_case_multiple_strategic_goals_allowed() {
        let mut hierarchy = GoalHierarchy::new();
        let fp = test_fingerprint();
        let discovery = test_discovery();

        let s1 = GoalNode::autonomous_goal(
            "Strategic1".into(),
            GoalLevel::Strategic,
            fp.clone(),
            discovery.clone(),
        )
        .unwrap();

        let s2 =
            GoalNode::autonomous_goal("Strategic2".into(), GoalLevel::Strategic, fp, discovery)
                .unwrap();

        hierarchy.add_goal(s1).unwrap();
        let result = hierarchy.add_goal(s2);

        // ARCH-03: Multiple Strategic goals are now allowed
        assert!(
            result.is_ok(),
            "Multiple Strategic goals should be allowed per ARCH-03"
        );
        assert_eq!(hierarchy.top_level_goals().len(), 2);
        println!("[EDGE CASE 3 PASSED] Multiple Strategic goals allowed (ARCH-03)");
    }

    // TASK-P0-001: Updated to use Tactical level
    #[test]
    fn test_edge_case_orphan_child_rejected() {
        let mut hierarchy = GoalHierarchy::new();
        let fp = test_fingerprint();
        let discovery = test_discovery();

        let fake_parent = Uuid::new_v4();
        // Use Tactical (not Strategic) since Strategic goals are top-level
        let child = GoalNode::child_goal(
            "Orphan".into(),
            GoalLevel::Tactical,
            fake_parent,
            fp,
            discovery,
        )
        .unwrap();

        let result = hierarchy.add_goal(child);
        assert!(matches!(result, Err(GoalHierarchyError::ParentNotFound(_))));
        println!("[EDGE CASE 4 PASSED] Orphan child rejected");
    }

    #[test]
    fn test_edge_case_empty_cluster_rejected() {
        let result = GoalDiscoveryMetadata::new(
            DiscoveryMethod::Clustering,
            0.8,
            0, // Invalid for Clustering
            0.7,
        );

        assert!(matches!(result, Err(GoalNodeError::EmptyCluster)));
        println!("[EDGE CASE 5 PASSED] Empty cluster rejected");
    }
}
