//! Core construction and validation tests for FusedEmbedding.

#[cfg(test)]
mod tests {
    use crate::error::EmbeddingError;
    use crate::types::dimensions::{COLBERT_V3_DIM, FUSED_OUTPUT, TOP_K_EXPERTS};
    use crate::types::fused::{AuxiliaryEmbeddingData, FusedEmbedding};
    use crate::types::ModelId;

    // ========== Helper Functions for Tests ==========

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

    fn make_token_vector() -> Vec<f32> {
        vec![0.5; COLBERT_V3_DIM]
    }

    // ========== Construction Tests ==========

    #[test]
    fn test_new_with_valid_1536d_vector_succeeds() {
        let vector = make_valid_vector();
        let weights = make_valid_weights();
        let selected = make_valid_selected();

        let result = FusedEmbedding::new(vector.clone(), weights, selected, 1000, 12345);

        assert!(result.is_ok());
        let emb = result.unwrap();
        assert_eq!(emb.vector.len(), FUSED_OUTPUT);
    }

    #[test]
    fn test_new_with_wrong_dimension_fails() {
        let vector = vec![0.1; 512];
        let weights = make_valid_weights();
        let selected = make_valid_selected();

        let result = FusedEmbedding::new(vector, weights, selected, 1000, 12345);

        assert!(result.is_err());
        match result.unwrap_err() {
            EmbeddingError::InvalidDimension { expected, actual } => {
                assert_eq!(expected, FUSED_OUTPUT);
                assert_eq!(actual, 512);
            }
            e => panic!("Expected InvalidDimension, got {:?}", e),
        }
    }

    #[test]
    fn test_new_with_invalid_expert_indices_fails() {
        let vector = make_valid_vector();
        let weights = make_valid_weights();
        let selected = [8, 0, 1, 2];

        let result = FusedEmbedding::new(vector, weights, selected, 1000, 12345);

        assert!(result.is_err());
        match result.unwrap_err() {
            EmbeddingError::FusionError { message } => {
                assert!(message.contains("8"));
                assert!(message.contains("Invalid expert index"));
            }
            e => panic!("Expected FusionError, got {:?}", e),
        }
    }

    #[test]
    fn test_new_with_valid_expert_weights_succeeds() {
        let vector = make_valid_vector();
        let weights = [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1];
        let selected = make_valid_selected();

        let result = FusedEmbedding::new(vector, weights, selected, 1000, 12345);

        assert!(result.is_ok());
        let emb = result.unwrap();
        assert_eq!(emb.expert_weights, weights);
    }

    #[test]
    fn test_new_computes_proper_defaults_for_latency_and_hash() {
        let vector = make_valid_vector();
        let weights = make_valid_weights();
        let selected = make_valid_selected();
        let latency = 5000u64;
        let hash = 0xDEADBEEF_u64;

        let result = FusedEmbedding::new(vector, weights, selected, latency, hash);

        assert!(result.is_ok());
        let emb = result.unwrap();
        assert_eq!(emb.pipeline_latency_us, latency);
        assert_eq!(emb.content_hash, hash);
        assert!(emb.aux_data.is_none());
    }

    #[test]
    fn test_with_aux_data_attaches_colbert_vectors() {
        let emb = make_valid_fused();
        let token_vecs = vec![make_token_vector(), make_token_vector()];
        let aux = AuxiliaryEmbeddingData::new(ModelId::LateInteraction, token_vecs);

        let emb_with_aux = emb.with_aux_data(aux);

        assert!(emb_with_aux.has_aux_data());
        let aux_ref = emb_with_aux.aux_data.as_ref().unwrap();
        assert_eq!(aux_ref.num_tokens, 2);
    }

    // ========== Validation Tests ==========

    #[test]
    fn test_validate_passes_for_valid_embedding() {
        let emb = make_valid_fused();
        let result = emb.validate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_rejects_nan_in_vector() {
        let mut emb = make_valid_fused();
        emb.vector[100] = f32::NAN;

        let result = emb.validate();

        assert!(result.is_err());
        match result.unwrap_err() {
            EmbeddingError::InvalidValue { index, value } => {
                assert_eq!(index, 100);
                assert!(value.is_nan());
            }
            e => panic!("Expected InvalidValue, got {:?}", e),
        }
    }

    #[test]
    fn test_validate_rejects_inf_in_vector() {
        let mut emb = make_valid_fused();
        emb.vector[500] = f32::INFINITY;

        let result = emb.validate();

        assert!(result.is_err());
        match result.unwrap_err() {
            EmbeddingError::InvalidValue { index, value } => {
                assert_eq!(index, 500);
                assert!(value.is_infinite() && value.is_sign_positive());
            }
            e => panic!("Expected InvalidValue, got {:?}", e),
        }
    }

    #[test]
    fn test_validate_rejects_neg_inf_in_vector() {
        let mut emb = make_valid_fused();
        emb.vector[200] = f32::NEG_INFINITY;

        let result = emb.validate();

        assert!(result.is_err());
        match result.unwrap_err() {
            EmbeddingError::InvalidValue { index, value } => {
                assert_eq!(index, 200);
                assert!(value.is_infinite() && value.is_sign_negative());
            }
            e => panic!("Expected InvalidValue, got {:?}", e),
        }
    }

    #[test]
    fn test_validate_rejects_expert_weights_sum_not_1() {
        let mut emb = make_valid_fused();
        emb.expert_weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];

        let result = emb.validate();

        assert!(result.is_err());
        match result.unwrap_err() {
            EmbeddingError::FusionError { message } => {
                assert!(message.contains("0.8"));
            }
            e => panic!("Expected FusionError, got {:?}", e),
        }
    }

    #[test]
    fn test_validate_accepts_expert_weights_sum_0_995() {
        let mut emb = make_valid_fused();
        emb.expert_weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.295];

        let result = emb.validate();

        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_accepts_expert_weights_sum_1_005() {
        let mut emb = make_valid_fused();
        emb.expert_weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.305];

        let result = emb.validate();

        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_rejects_expert_weights_sum_1_02() {
        let mut emb = make_valid_fused();
        emb.expert_weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.32];

        let result = emb.validate();

        assert!(result.is_err());
    }

    #[test]
    fn test_edge_case_duplicate_expert_selection() {
        let vector = make_valid_vector();
        let weights = make_valid_weights();
        let selected = [3, 3, 1, 2];

        let result = FusedEmbedding::new(vector, weights, selected, 1000, 12345);

        assert!(result.is_err());
        match result.unwrap_err() {
            EmbeddingError::FusionError { message } => {
                assert!(message.contains("Duplicate"));
            }
            e => panic!("Expected FusionError for duplicate, got {:?}", e),
        }
    }
}
