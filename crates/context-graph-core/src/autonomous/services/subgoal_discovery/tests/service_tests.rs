//! Tests for SubGoalDiscovery service.

use crate::autonomous::bootstrap::GoalId;
use crate::autonomous::evolution::{GoalEvolutionConfig, GoalLevel, SubGoalCandidate};

use super::super::cluster::MemoryCluster;
use super::super::config::DiscoveryConfig;
use super::super::service::SubGoalDiscovery;
use super::{make_cluster, make_labeled_cluster};

#[test]
fn test_subgoal_discovery_new() {
    let discovery = SubGoalDiscovery::new();

    assert_eq!(discovery.config().min_cluster_size, 10);

    println!("[PASS] test_subgoal_discovery_new");
}

#[test]
fn test_subgoal_discovery_with_config() {
    let config = DiscoveryConfig {
        min_cluster_size: 5,
        min_coherence: 0.5,
        emergence_threshold: 0.8,
        max_candidates: 10,
        min_confidence: 0.4,
    };

    let discovery = SubGoalDiscovery::with_config(config);

    assert_eq!(discovery.config().min_cluster_size, 5);
    assert!((discovery.config().min_coherence - 0.5).abs() < f32::EPSILON);

    println!("[PASS] test_subgoal_discovery_with_config");
}

#[test]
fn test_subgoal_discovery_from_evolution_config() {
    let evolution_config = GoalEvolutionConfig {
        min_cluster_size: 25,
        ..Default::default()
    };

    let discovery = SubGoalDiscovery::from_evolution_config(&evolution_config);

    assert_eq!(discovery.config().min_cluster_size, 25);

    println!("[PASS] test_subgoal_discovery_from_evolution_config");
}

#[test]
fn test_discover_from_clusters_empty() {
    let discovery = SubGoalDiscovery::new();
    let result = discovery.discover_from_clusters(&[]);

    assert!(!result.has_candidates());
    assert_eq!(result.cluster_count, 0);

    println!("[PASS] test_discover_from_clusters_empty");
}

#[test]
fn test_discover_from_clusters_filters_small() {
    let discovery = SubGoalDiscovery::new(); // min_cluster_size = 10

    let clusters = vec![
        make_cluster(5, 0.8, 0.7),  // Too small
        make_cluster(15, 0.8, 0.7), // Valid
    ];

    let result = discovery.discover_from_clusters(&clusters);

    assert_eq!(result.cluster_count, 2);
    assert_eq!(result.viable_clusters, 1);
    assert!(result.filtered_count >= 1);

    println!("[PASS] test_discover_from_clusters_filters_small");
}

#[test]
fn test_discover_from_clusters_filters_low_coherence() {
    let config = DiscoveryConfig {
        min_cluster_size: 5,
        min_coherence: 0.7,
        ..Default::default()
    };
    let discovery = SubGoalDiscovery::with_config(config);

    let clusters = vec![
        make_cluster(10, 0.5, 0.7), // Low coherence
        make_cluster(10, 0.8, 0.7), // Valid
    ];

    let result = discovery.discover_from_clusters(&clusters);

    assert_eq!(result.viable_clusters, 1);

    println!("[PASS] test_discover_from_clusters_filters_low_coherence");
}

#[test]
fn test_discover_from_clusters_valid() {
    let config = DiscoveryConfig {
        min_cluster_size: 5,
        min_coherence: 0.5,
        min_confidence: 0.3,
        ..Default::default()
    };
    let discovery = SubGoalDiscovery::with_config(config);

    let clusters = vec![
        make_labeled_cluster(10, 0.8, 0.7, "Goal A"),
        make_labeled_cluster(15, 0.9, 0.8, "Goal B"),
    ];

    let result = discovery.discover_from_clusters(&clusters);

    assert!(result.has_candidates());
    assert_eq!(result.candidates.len(), 2);

    println!("[PASS] test_discover_from_clusters_valid");
}

#[test]
fn test_discover_from_clusters_respects_max_candidates() {
    let config = DiscoveryConfig {
        min_cluster_size: 5,
        min_coherence: 0.5,
        min_confidence: 0.0,
        max_candidates: 2,
        ..Default::default()
    };
    let discovery = SubGoalDiscovery::with_config(config);

    let clusters: Vec<MemoryCluster> = (0..5)
        .map(|i| make_labeled_cluster(10, 0.8, 0.7, &format!("Goal {}", i)))
        .collect();

    let result = discovery.discover_from_clusters(&clusters);

    assert_eq!(result.candidates.len(), 2);
    assert!(result.filtered_count >= 3);

    println!("[PASS] test_discover_from_clusters_respects_max_candidates");
}

#[test]
fn test_extract_candidate_valid() {
    let config = DiscoveryConfig {
        min_cluster_size: 5,
        min_coherence: 0.5,
        ..Default::default()
    };
    let discovery = SubGoalDiscovery::with_config(config);

    let cluster = make_labeled_cluster(10, 0.8, 0.7, "Test Goal");

    let candidate = discovery.extract_candidate(&cluster);

    assert!(candidate.is_some());
    let c = candidate.unwrap();
    assert_eq!(c.suggested_description, "Test Goal");
    assert_eq!(c.cluster_size, 10);
    assert!((c.centroid_alignment - 0.7).abs() < f32::EPSILON);
    assert!(c.confidence > 0.0);

    println!("[PASS] test_extract_candidate_valid");
}

#[test]
fn test_extract_candidate_empty_cluster() {
    let discovery = SubGoalDiscovery::new();
    let cluster = MemoryCluster::new(vec![0.1], vec![], 0.8);

    let candidate = discovery.extract_candidate(&cluster);

    assert!(candidate.is_none());

    println!("[PASS] test_extract_candidate_empty_cluster");
}

#[test]
fn test_extract_candidate_too_small() {
    let discovery = SubGoalDiscovery::new(); // min_cluster_size = 10
    let cluster = make_cluster(5, 0.8, 0.7);

    let candidate = discovery.extract_candidate(&cluster);

    assert!(candidate.is_none());

    println!("[PASS] test_extract_candidate_too_small");
}

#[test]
fn test_extract_candidate_low_coherence() {
    let discovery = SubGoalDiscovery::new(); // min_coherence = 0.6
    let cluster = make_cluster(15, 0.4, 0.7);

    let candidate = discovery.extract_candidate(&cluster);

    assert!(candidate.is_none());

    println!("[PASS] test_extract_candidate_low_coherence");
}

#[test]
fn test_extract_candidate_generates_description() {
    let config = DiscoveryConfig {
        min_cluster_size: 5,
        min_coherence: 0.5,
        ..Default::default()
    };
    let discovery = SubGoalDiscovery::with_config(config);

    let cluster = make_cluster(10, 0.8, 0.7); // No label

    let candidate = discovery.extract_candidate(&cluster).unwrap();

    assert!(candidate.suggested_description.contains("10 memories"));

    println!("[PASS] test_extract_candidate_generates_description");
}

#[test]
fn test_compute_confidence_empty_cluster() {
    let discovery = SubGoalDiscovery::new();
    let cluster = MemoryCluster::new(vec![0.1], vec![], 0.8);

    let confidence = discovery.compute_confidence(&cluster);

    assert!((confidence - 0.0).abs() < f32::EPSILON);

    println!("[PASS] test_compute_confidence_empty_cluster");
}

#[test]
fn test_compute_confidence_components() {
    let discovery = SubGoalDiscovery::new();

    // High coherence cluster
    let high_coherence = make_cluster(10, 0.9, 0.5);
    let conf1 = discovery.compute_confidence(&high_coherence);

    // Low coherence cluster (same size and alignment)
    let low_coherence = make_cluster(10, 0.3, 0.5);
    let conf2 = discovery.compute_confidence(&low_coherence);

    assert!(
        conf1 > conf2,
        "Higher coherence should yield higher confidence"
    );

    // Larger cluster (same coherence and alignment)
    let larger = make_cluster(50, 0.7, 0.5);
    let smaller = make_cluster(10, 0.7, 0.5);
    let conf3 = discovery.compute_confidence(&larger);
    let conf4 = discovery.compute_confidence(&smaller);

    assert!(
        conf3 > conf4,
        "Larger cluster should yield higher confidence"
    );

    println!("[PASS] test_compute_confidence_components");
}

#[test]
fn test_compute_confidence_bounds() {
    let discovery = SubGoalDiscovery::new();

    // Maximum values
    let max_cluster = make_cluster(100, 1.0, 1.0);
    let max_conf = discovery.compute_confidence(&max_cluster);
    assert!(max_conf <= 1.0);
    assert!(max_conf >= 0.0);

    // Minimum non-empty values
    let min_cluster = make_cluster(1, 0.0, 0.0);
    let min_conf = discovery.compute_confidence(&min_cluster);
    assert!(min_conf >= 0.0);
    assert!(min_conf <= 1.0);

    println!("[PASS] test_compute_confidence_bounds");
}

#[test]
fn test_find_parent_goal_empty_goals() {
    let discovery = SubGoalDiscovery::new();
    let candidate = SubGoalCandidate {
        suggested_description: "Test".into(),
        level: GoalLevel::Tactical,
        parent_id: GoalId::new(),
        cluster_size: 10,
        centroid_alignment: 0.7,
        confidence: 0.8,
        supporting_memories: vec![],
    };

    let parent = discovery.find_parent_goal(&candidate, &[]);

    assert!(parent.is_none());

    println!("[PASS] test_find_parent_goal_empty_goals");
}

// TASK-P0-001: Tests Tactical level since Strategic goals are top-level and have no parent
#[test]
fn test_find_parent_goal_tactical() {
    let discovery = SubGoalDiscovery::new();
    let goals = vec![GoalId::new(), GoalId::new()];

    let candidate = SubGoalCandidate {
        suggested_description: "Test".into(),
        level: GoalLevel::Tactical,
        parent_id: GoalId::new(),
        cluster_size: 50,
        centroid_alignment: 0.9,
        confidence: 0.9,
        supporting_memories: vec![],
    };

    let parent = discovery.find_parent_goal(&candidate, &goals);

    assert!(parent.is_some(), "Tactical goals should have a parent");
    assert_eq!(parent.unwrap(), goals[0]);

    println!("[PASS] test_find_parent_goal_tactical");
}

#[test]
fn test_find_parent_goal_operational() {
    let discovery = SubGoalDiscovery::new();
    let goals = vec![GoalId::new(), GoalId::new()];

    let candidate = SubGoalCandidate {
        suggested_description: "Test".into(),
        level: GoalLevel::Operational,
        parent_id: GoalId::new(),
        cluster_size: 10,
        centroid_alignment: 0.5,
        confidence: 0.6,
        supporting_memories: vec![],
    };

    let parent = discovery.find_parent_goal(&candidate, &goals);

    assert!(parent.is_some());
    // Operational prefers second goal if available
    assert_eq!(parent.unwrap(), goals[1]);

    println!("[PASS] test_find_parent_goal_operational");
}

// TASK-P0-001: Strategic is now top-level
#[test]
fn test_find_parent_goal_strategic_top_level() {
    let discovery = SubGoalDiscovery::new();
    let goals = vec![GoalId::new()];

    let candidate = SubGoalCandidate {
        suggested_description: "Test".into(),
        level: GoalLevel::Strategic,
        parent_id: GoalId::new(),
        cluster_size: 100,
        centroid_alignment: 1.0,
        confidence: 1.0,
        supporting_memories: vec![],
    };

    let parent = discovery.find_parent_goal(&candidate, &goals);

    assert!(
        parent.is_none(),
        "Strategic goals are top-level and should have no parent"
    );

    println!("[PASS] test_find_parent_goal_strategic_top_level");
}

#[test]
fn test_determine_level_strategic() {
    let discovery = SubGoalDiscovery::new();

    let level = discovery.determine_level(0.9, 60);

    assert_eq!(level, GoalLevel::Strategic);

    println!("[PASS] test_determine_level_strategic");
}

#[test]
fn test_determine_level_tactical() {
    let discovery = SubGoalDiscovery::new();

    let level = discovery.determine_level(0.75, 30);

    assert_eq!(level, GoalLevel::Tactical);

    println!("[PASS] test_determine_level_tactical");
}

#[test]
fn test_determine_level_operational() {
    let discovery = SubGoalDiscovery::new();

    let level = discovery.determine_level(0.5, 10);

    assert_eq!(level, GoalLevel::Operational);

    println!("[PASS] test_determine_level_operational");
}

#[test]
fn test_determine_level_boundary_strategic() {
    let discovery = SubGoalDiscovery::new();

    // Just at strategic threshold
    let level = discovery.determine_level(0.85, 50);
    assert_eq!(level, GoalLevel::Strategic);

    // Just below strategic (high confidence but not enough evidence)
    let level = discovery.determine_level(0.85, 49);
    assert_eq!(level, GoalLevel::Tactical);

    println!("[PASS] test_determine_level_boundary_strategic");
}

#[test]
fn test_determine_level_boundary_tactical() {
    let discovery = SubGoalDiscovery::new();

    // Just at tactical threshold
    let level = discovery.determine_level(0.7, 20);
    assert_eq!(level, GoalLevel::Tactical);

    // Just below tactical
    let level = discovery.determine_level(0.69, 20);
    assert_eq!(level, GoalLevel::Operational);

    println!("[PASS] test_determine_level_boundary_tactical");
}

#[test]
fn test_should_promote_valid() {
    let discovery = SubGoalDiscovery::new(); // emergence_threshold = 0.7

    let candidate = SubGoalCandidate {
        suggested_description: "Test".into(),
        level: GoalLevel::Tactical,
        parent_id: GoalId::new(),
        cluster_size: 15,
        centroid_alignment: 0.6,
        confidence: 0.8,
        supporting_memories: vec![],
    };

    assert!(discovery.should_promote(&candidate));

    println!("[PASS] test_should_promote_valid");
}

#[test]
fn test_should_promote_low_confidence() {
    let discovery = SubGoalDiscovery::new(); // emergence_threshold = 0.7

    let candidate = SubGoalCandidate {
        suggested_description: "Test".into(),
        level: GoalLevel::Tactical,
        parent_id: GoalId::new(),
        cluster_size: 15,
        centroid_alignment: 0.6,
        confidence: 0.5, // Below threshold
        supporting_memories: vec![],
    };

    assert!(!discovery.should_promote(&candidate));

    println!("[PASS] test_should_promote_low_confidence");
}

#[test]
fn test_should_promote_small_cluster() {
    let discovery = SubGoalDiscovery::new(); // min_cluster_size = 10

    let candidate = SubGoalCandidate {
        suggested_description: "Test".into(),
        level: GoalLevel::Tactical,
        parent_id: GoalId::new(),
        cluster_size: 5, // Below minimum
        centroid_alignment: 0.6,
        confidence: 0.8,
        supporting_memories: vec![],
    };

    assert!(!discovery.should_promote(&candidate));

    println!("[PASS] test_should_promote_small_cluster");
}

#[test]
fn test_should_promote_low_alignment() {
    let discovery = SubGoalDiscovery::new();

    let candidate = SubGoalCandidate {
        suggested_description: "Test".into(),
        level: GoalLevel::Tactical,
        parent_id: GoalId::new(),
        cluster_size: 15,
        centroid_alignment: 0.2, // Below 0.3 threshold
        confidence: 0.8,
        supporting_memories: vec![],
    };

    assert!(!discovery.should_promote(&candidate));

    println!("[PASS] test_should_promote_low_alignment");
}

#[test]
fn test_rank_candidates_by_confidence() {
    let discovery = SubGoalDiscovery::new();

    let mut candidates = vec![
        SubGoalCandidate {
            suggested_description: "Low".into(),
            level: GoalLevel::Operational,
            parent_id: GoalId::new(),
            cluster_size: 10,
            centroid_alignment: 0.5,
            confidence: 0.5,
            supporting_memories: vec![],
        },
        SubGoalCandidate {
            suggested_description: "High".into(),
            level: GoalLevel::Strategic,
            parent_id: GoalId::new(),
            cluster_size: 10,
            centroid_alignment: 0.5,
            confidence: 0.9,
            supporting_memories: vec![],
        },
    ];

    discovery.rank_candidates(&mut candidates);

    assert_eq!(candidates[0].suggested_description, "High");
    assert_eq!(candidates[1].suggested_description, "Low");

    println!("[PASS] test_rank_candidates_by_confidence");
}

#[test]
fn test_rank_candidates_by_size_tiebreaker() {
    let discovery = SubGoalDiscovery::new();

    let mut candidates = vec![
        SubGoalCandidate {
            suggested_description: "Small".into(),
            level: GoalLevel::Tactical,
            parent_id: GoalId::new(),
            cluster_size: 10,
            centroid_alignment: 0.5,
            confidence: 0.8,
            supporting_memories: vec![],
        },
        SubGoalCandidate {
            suggested_description: "Large".into(),
            level: GoalLevel::Tactical,
            parent_id: GoalId::new(),
            cluster_size: 50,
            centroid_alignment: 0.5,
            confidence: 0.8, // Same confidence
            supporting_memories: vec![],
        },
    ];

    discovery.rank_candidates(&mut candidates);

    assert_eq!(candidates[0].suggested_description, "Large");
    assert_eq!(candidates[1].suggested_description, "Small");

    println!("[PASS] test_rank_candidates_by_size_tiebreaker");
}

#[test]
fn test_rank_candidates_by_alignment_tiebreaker() {
    let discovery = SubGoalDiscovery::new();

    let mut candidates = vec![
        SubGoalCandidate {
            suggested_description: "LowAlign".into(),
            level: GoalLevel::Tactical,
            parent_id: GoalId::new(),
            cluster_size: 20,
            centroid_alignment: 0.3,
            confidence: 0.8,
            supporting_memories: vec![],
        },
        SubGoalCandidate {
            suggested_description: "HighAlign".into(),
            level: GoalLevel::Tactical,
            parent_id: GoalId::new(),
            cluster_size: 20, // Same size
            centroid_alignment: 0.9,
            confidence: 0.8, // Same confidence
            supporting_memories: vec![],
        },
    ];

    discovery.rank_candidates(&mut candidates);

    assert_eq!(candidates[0].suggested_description, "HighAlign");
    assert_eq!(candidates[1].suggested_description, "LowAlign");

    println!("[PASS] test_rank_candidates_by_alignment_tiebreaker");
}

#[test]
fn test_rank_candidates_empty() {
    let discovery = SubGoalDiscovery::new();
    let mut candidates: Vec<SubGoalCandidate> = vec![];

    discovery.rank_candidates(&mut candidates);

    assert!(candidates.is_empty());

    println!("[PASS] test_rank_candidates_empty");
}

#[test]
fn test_rank_candidates_single() {
    let discovery = SubGoalDiscovery::new();
    let mut candidates = vec![SubGoalCandidate {
        suggested_description: "Only".into(),
        level: GoalLevel::Tactical,
        parent_id: GoalId::new(),
        cluster_size: 10,
        centroid_alignment: 0.5,
        confidence: 0.7,
        supporting_memories: vec![],
    }];

    discovery.rank_candidates(&mut candidates);

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].suggested_description, "Only");

    println!("[PASS] test_rank_candidates_single");
}

#[test]
fn test_subgoal_discovery_default() {
    let discovery = SubGoalDiscovery::default();

    assert_eq!(discovery.config().min_cluster_size, 10);

    println!("[PASS] test_subgoal_discovery_default");
}
