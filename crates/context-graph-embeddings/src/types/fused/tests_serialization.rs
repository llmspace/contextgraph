//! Serialization tests for FusedEmbedding and AuxiliaryEmbeddingData.

#[cfg(test)]
mod tests {
    use crate::error::EmbeddingError;
    use crate::types::dimensions::{COLBERT_V3_DIM, FUSED_OUTPUT, TOP_K_EXPERTS};
    use crate::types::fused::constants::CORE_BINARY_SIZE;
    use crate::types::fused::{AuxiliaryEmbeddingData, FusedEmbedding};
    use crate::types::ModelId;

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

    // ========== Serialization Tests ==========

    #[test]
    fn test_to_bytes_produces_exactly_6200_bytes_no_aux_data() {
        let emb = make_valid_fused();

        let bytes = emb.to_bytes();

        assert_eq!(
            bytes.len(),
            CORE_BINARY_SIZE,
            "Expected {} bytes, got {}",
            CORE_BINARY_SIZE,
            bytes.len()
        );
    }

    #[test]
    fn test_from_bytes_reconstructs_identical_embedding() {
        let emb = make_valid_fused();
        let bytes = emb.to_bytes();

        let reconstructed = FusedEmbedding::from_bytes(&bytes).expect("Should deserialize");

        assert_eq!(emb.vector, reconstructed.vector);
        assert_eq!(emb.expert_weights, reconstructed.expert_weights);
        assert_eq!(emb.selected_experts, reconstructed.selected_experts);
        assert_eq!(emb.pipeline_latency_us, reconstructed.pipeline_latency_us);
        assert_eq!(emb.content_hash, reconstructed.content_hash);
    }

    #[test]
    fn test_to_bytes_from_bytes_round_trip_preserves_all_fields() {
        let token_vecs = vec![make_token_vector(); 10];
        let aux = AuxiliaryEmbeddingData::new(ModelId::LateInteraction, token_vecs);
        let emb = make_valid_fused().with_aux_data(aux);

        let bytes = emb.to_bytes();
        let reconstructed = FusedEmbedding::from_bytes(&bytes).expect("Should deserialize");

        assert_eq!(emb.vector, reconstructed.vector);
        assert_eq!(emb.expert_weights, reconstructed.expert_weights);
        assert!(reconstructed.aux_data.is_some());
        let aux_rec = reconstructed.aux_data.unwrap();
        assert_eq!(aux_rec.num_tokens, 10);
    }

    #[test]
    fn test_from_bytes_rejects_truncated_data() {
        let bytes = vec![0u8; 100];

        let result = FusedEmbedding::from_bytes(&bytes);

        assert!(result.is_err());
        match result.unwrap_err() {
            EmbeddingError::SerializationError { message } => {
                assert!(message.contains("too short"));
            }
            e => panic!("Expected SerializationError, got {:?}", e),
        }
    }

    #[test]
    fn test_from_bytes_tolerates_extra_bytes_after_valid_data() {
        let emb = make_valid_fused();
        let mut bytes = emb.to_bytes();
        bytes.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

        let result = FusedEmbedding::from_bytes(&bytes);
        assert!(result.is_ok());
    }

    #[test]
    fn test_serde_json_round_trip_works() {
        let emb = make_valid_fused();

        let json = serde_json::to_string(&emb).expect("Should serialize to JSON");
        let reconstructed: FusedEmbedding =
            serde_json::from_str(&json).expect("Should deserialize from JSON");

        assert_eq!(emb.vector, reconstructed.vector);
        assert_eq!(emb.expert_weights, reconstructed.expert_weights);
    }

    #[test]
    fn test_serde_skips_aux_data_when_none() {
        let emb = make_valid_fused();

        let json = serde_json::to_string(&emb).expect("Should serialize");

        assert!(!json.contains("aux_data"));
    }

    #[test]
    fn test_serde_includes_aux_data_when_some() {
        let token_vecs = vec![make_token_vector()];
        let aux = AuxiliaryEmbeddingData::new(ModelId::LateInteraction, token_vecs);
        let emb = make_valid_fused().with_aux_data(aux);

        let json = serde_json::to_string(&emb).expect("Should serialize");

        assert!(json.contains("aux_data"));
        assert!(json.contains("token_vectors"));
    }

    #[test]
    fn test_binary_size_constant() {
        assert_eq!(CORE_BINARY_SIZE, 6200);
    }
}
