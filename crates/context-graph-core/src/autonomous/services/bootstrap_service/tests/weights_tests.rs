//! Section weights tests for BootstrapService

use crate::autonomous::services::bootstrap_service::*;
use crate::autonomous::{BootstrapConfig, SectionWeights};
use std::path::PathBuf;

#[test]
fn test_section_weights_affect_scoring() {
    let high_position_config = BootstrapServiceConfig {
        doc_dir: PathBuf::from("."),
        file_extensions: vec!["md".into()],
        max_docs: 10,
        bootstrap_config: BootstrapConfig {
            section_weights: SectionWeights {
                position_weight: 3.0, // High position weight
                density_weight: 1.0,
                apply_idf: false,
            },
            ..Default::default()
        },
    };

    let high_density_config = BootstrapServiceConfig {
        doc_dir: PathBuf::from("."),
        file_extensions: vec!["md".into()],
        max_docs: 10,
        bootstrap_config: BootstrapConfig {
            section_weights: SectionWeights {
                position_weight: 1.0,
                density_weight: 3.0, // High density weight
                apply_idf: false,
            },
            ..Default::default()
        },
    };

    let service_pos = BootstrapService::with_config(high_position_config);
    let service_den = BootstrapService::with_config(high_density_config);

    // Candidate at start with low density
    let start_low_density = GoalCandidate {
        text: "The goal is here in this architecture system.".to_string(),
        source: "test.md".to_string(),
        position: 0.0,
        density: 0.05,
        keyword_count: 2,
        line_number: 1,
    };

    // Candidate in middle with high density
    let middle_high_density = GoalCandidate {
        text: "The goal mission purpose objective vision is to build a system architecture."
            .to_string(),
        source: "test.md".to_string(),
        position: 0.5,
        density: 0.4,
        keyword_count: 5,
        line_number: 50,
    };

    let pos_score_start = service_pos.score_candidate(&start_low_density);
    let pos_score_middle = service_pos.score_candidate(&middle_high_density);

    let den_score_start = service_den.score_candidate(&start_low_density);
    let den_score_middle = service_den.score_candidate(&middle_high_density);

    // With high position weight, start should be favored more
    // With high density weight, high density should be favored more
    // The relative difference should change based on weights
    let pos_diff = pos_score_start - pos_score_middle;
    let den_diff = den_score_start - den_score_middle;

    // When position is weighted high, start should gain relative to middle
    // When density is weighted high, high-density should gain relative to low-density
    assert!(
        pos_diff > den_diff,
        "Position weighting should favor start position more"
    );

    println!("[PASS] test_section_weights_affect_scoring");
}

#[test]
fn test_idf_weighting_enabled() {
    let config_with_idf = BootstrapServiceConfig {
        bootstrap_config: BootstrapConfig {
            section_weights: SectionWeights {
                position_weight: 1.0,
                density_weight: 1.0,
                apply_idf: true,
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let config_without_idf = BootstrapServiceConfig {
        bootstrap_config: BootstrapConfig {
            section_weights: SectionWeights {
                position_weight: 1.0,
                density_weight: 1.0,
                apply_idf: false,
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let service_with = BootstrapService::with_config(config_with_idf);
    let service_without = BootstrapService::with_config(config_without_idf);

    let candidate = GoalCandidate {
        text: "The goal mission purpose objective is to build this system architecture.".to_string(),
        source: "test.md".to_string(),
        position: 0.1,
        density: 0.2,
        keyword_count: 5,
        line_number: 1,
    };

    let score_with = service_with.score_candidate(&candidate);
    let score_without = service_without.score_candidate(&candidate);

    // IDF should boost candidates with multiple unique keywords
    assert!(
        score_with >= score_without,
        "IDF should boost or maintain score for multi-keyword candidates"
    );

    println!("[PASS] test_idf_weighting_enabled");
}
