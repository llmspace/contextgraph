//! Normalization and similarity tests for FusedEmbedding.

#[cfg(test)]
mod tests {
    use crate::types::dimensions::{FUSED_OUTPUT, TOP_K_EXPERTS};
    use crate::types::fused::FusedEmbedding;

    fn make_valid_vector() -> Vec<f32> {
        vec![0.1; FUSED_OUTPUT]
    }

    fn make_valid_weights() -> [f32; 8] {
        [0.25, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25]
    }

    fn make_valid_selected() -> [u8; TOP_K_EXPERTS] {
        [0, 1, 4, 5]
    }

    fn make_valid_fused() -> FusedEmbedding {
        FusedEmbedding::new(
            make_valid_vector(),
            make_valid_weights(),
            make_valid_selected(),
            1000,
            12345,
        )
        .expect("Test helper should create valid embedding")
    }

    // ========== Normalization Tests ==========

    #[test]
    fn test_normalize_produces_unit_vector() {
        let mut emb = make_valid_fused();
        for (i, val) in emb.vector.iter_mut().enumerate() {
            *val = (i % 10) as f32;
        }

        emb.normalize();

        assert!(
            (emb.magnitude() - 1.0).abs() < 1e-5,
            "Magnitude should be ~1.0, got {}",
            emb.magnitude()
        );
    }

    #[test]
    fn test_normalize_handles_zero_vector() {
        let mut emb = make_valid_fused();
        for val in emb.vector.iter_mut() {
            *val = 0.0;
        }

        emb.normalize();

        assert!(emb.vector.iter().all(|&v| v == 0.0));
        assert!(!emb.vector.iter().any(|v| v.is_nan()));
    }

    #[test]
    fn test_is_normalized_returns_true_after_normalize() {
        let mut emb = make_valid_fused();
        for (i, val) in emb.vector.iter_mut().enumerate() {
            *val = i as f32 * 0.001;
        }

        assert!(!emb.is_normalized(), "Should not be normalized before");

        emb.normalize();

        assert!(emb.is_normalized(), "Should be normalized after");
    }

    #[test]
    fn test_is_normalized_returns_false_before_normalize() {
        let emb = make_valid_fused();

        let mag = emb.magnitude();
        let is_norm = emb.is_normalized();

        assert!(!is_norm);
        assert!(mag > 1.0); // Vector of all 0.1 has mag > 1
    }

    #[test]
    fn test_magnitude_computes_correct_l2_norm() {
        let mut emb = make_valid_fused();
        for val in emb.vector.iter_mut() {
            *val = 0.0;
        }
        emb.vector[0] = 3.0;
        emb.vector[1] = 4.0;

        let mag = emb.magnitude();

        assert!((mag - 5.0).abs() < 1e-6);
    }

    // ========== Similarity Tests ==========

    #[test]
    fn test_cosine_similarity_returns_1_for_identical_vectors() {
        let emb1 = make_valid_fused();
        let emb2 = emb1.clone();

        let sim = emb1.cosine_similarity(&emb2);

        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Similarity of identical vectors should be 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_returns_neg1_for_opposite_vectors() {
        let emb1 = make_valid_fused();
        let mut emb2 = emb1.clone();
        for val in emb2.vector.iter_mut() {
            *val = -*val;
        }

        let sim = emb1.cosine_similarity(&emb2);

        assert!(
            (sim - (-1.0)).abs() < 1e-6,
            "Similarity of opposite vectors should be -1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_returns_0_for_orthogonal_vectors() {
        let mut emb1 = make_valid_fused();
        let mut emb2 = make_valid_fused();

        for val in emb1.vector.iter_mut() {
            *val = 0.0;
        }
        emb1.vector[0] = 1.0;

        for val in emb2.vector.iter_mut() {
            *val = 0.0;
        }
        emb2.vector[1] = 1.0;

        let sim = emb1.cosine_similarity(&emb2);

        assert!(
            sim.abs() < 1e-6,
            "Similarity of orthogonal vectors should be 0.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_is_symmetric() {
        let emb1 = make_valid_fused();
        let mut emb2 = make_valid_fused();
        for (i, val) in emb2.vector.iter_mut().enumerate() {
            *val = (i as f32 * 0.01).sin();
        }

        let sim12 = emb1.cosine_similarity(&emb2);
        let sim21 = emb2.cosine_similarity(&emb1);

        assert!(
            (sim12 - sim21).abs() < 1e-6,
            "Similarity should be symmetric: {} vs {}",
            sim12,
            sim21
        );
    }

    #[test]
    fn test_cosine_similarity_range_is_minus1_to_1() {
        let emb1 = make_valid_fused();
        let mut emb2 = make_valid_fused();
        for (i, val) in emb2.vector.iter_mut().enumerate() {
            *val = (i as f32 * 0.1).cos() - 0.5;
        }

        let sim = emb1.cosine_similarity(&emb2);

        assert!(
            sim >= -1.0 && sim <= 1.0,
            "Similarity should be in [-1, 1], got {}",
            sim
        );
    }

    #[test]
    fn test_edge_case_zero_vector_normalization() {
        let mut emb = make_valid_fused();
        for val in emb.vector.iter_mut() {
            *val = 0.0;
        }

        emb.normalize();

        assert!(emb.vector.iter().all(|&v| v == 0.0));
        assert!(!emb.vector.iter().any(|v| v.is_nan()));
    }
}
