//! Tests for binary encoding/decoding.

#[cfg(test)]
mod tests {
    use crate::storage::binary::{
        DecodeError, EmbeddingBinaryCodec, EmbeddingHeader, EncodeError, EMBEDDING_MAGIC,
    };
    use crate::types::dimensions::{FUSED_OUTPUT, NUM_EXPERTS};
    use crate::types::FusedEmbedding;

    fn make_test_embedding() -> FusedEmbedding {
        FusedEmbedding::new(
            vec![0.1; FUSED_OUTPUT],
            [0.125; NUM_EXPERTS], // sum = 1.0
            [0, 1, 2, 3],
            1000,
            0xDEADBEEF,
        )
        .expect("test embedding creation")
    }

    // ========== Header Tests ==========

    #[test]
    fn test_header_is_exactly_64_bytes() {
        let size = std::mem::size_of::<EmbeddingHeader>();
        println!("BEFORE: Expected size = 64 bytes");
        println!("AFTER: Actual size = {} bytes", size);
        assert_eq!(size, 64);
        println!("PASSED: EmbeddingHeader is exactly 64 bytes");
    }

    #[test]
    fn test_header_alignment_is_suitable_for_pod() {
        let align = std::mem::align_of::<EmbeddingHeader>();
        println!("BEFORE: EmbeddingHeader needs Pod-compatible alignment");
        println!("AFTER: Actual alignment = {} bytes", align);
        assert!(align >= 8, "Alignment must be at least 8 bytes for u64 fields");
        println!("PASSED: EmbeddingHeader has Pod-compatible alignment");
    }

    #[test]
    fn test_min_buffer_size_is_6244() {
        println!("BEFORE: Expected MIN_BUFFER_SIZE = 6244");
        println!(
            "AFTER: Actual MIN_BUFFER_SIZE = {}",
            EmbeddingBinaryCodec::MIN_BUFFER_SIZE
        );
        assert_eq!(EmbeddingBinaryCodec::MIN_BUFFER_SIZE, 6244);
        println!("PASSED: MIN_BUFFER_SIZE is exactly 6244 bytes");
    }

    // ========== Encode Tests ==========

    #[test]
    fn test_encode_produces_6244_bytes_no_aux() {
        let codec = EmbeddingBinaryCodec::new();
        let embedding = make_test_embedding();

        println!(
            "BEFORE: embedding.vector.len() = {}",
            embedding.vector.len()
        );

        let bytes = codec.encode(&embedding).expect("encode should succeed");

        println!("AFTER: bytes.len() = {}", bytes.len());
        assert_eq!(bytes.len(), EmbeddingBinaryCodec::MIN_BUFFER_SIZE);
        assert_eq!(bytes.len(), 6244);
        println!("PASSED: encode produces exactly 6244 bytes (no aux_data)");
    }

    #[test]
    fn test_encode_writes_correct_magic() {
        let codec = EmbeddingBinaryCodec::new();
        let embedding = make_test_embedding();

        let bytes = codec.encode(&embedding).expect("encode");

        println!(
            "MAGIC: {:02x} {:02x} {:02x} {:02x}",
            bytes[0], bytes[1], bytes[2], bytes[3]
        );
        assert_eq!(&bytes[0..4], &EMBEDDING_MAGIC);
        println!("PASSED: magic bytes = 'CGEB' (0x43474542)");
    }

    #[test]
    fn test_encode_fails_fast_on_wrong_dimension() {
        let codec = EmbeddingBinaryCodec::new();
        let bad_embedding = FusedEmbedding {
            vector: vec![0.0; 512], // WRONG dimension
            expert_weights: [0.125; 8],
            selected_experts: [0, 1, 2, 3],
            pipeline_latency_us: 0,
            content_hash: 0,
            aux_data: None,
        };

        let result = codec.encode(&bad_embedding);

        println!("Result: {:?}", result);
        assert!(result.is_err());
        match result.unwrap_err() {
            EncodeError::InvalidDimension { expected, actual } => {
                assert_eq!(expected, 1536);
                assert_eq!(actual, 512);
            }
            e => panic!("Expected InvalidDimension, got {:?}", e),
        }
        println!("PASSED: encode fails fast on wrong dimension");
    }

    // ========== Decode Tests ==========

    #[test]
    fn test_decode_round_trip_preserves_all_fields() {
        let codec = EmbeddingBinaryCodec::new();
        let original = make_test_embedding();

        println!("BEFORE: vector[0..3] = {:?}", &original.vector[0..3]);
        println!("BEFORE: expert_weights = {:?}", original.expert_weights);
        println!(
            "BEFORE: selected_experts = {:?}",
            original.selected_experts
        );
        println!("BEFORE: content_hash = {:#x}", original.content_hash);

        let bytes = codec.encode(&original).expect("encode");
        let decoded = codec.decode(&bytes).expect("decode");

        println!("AFTER: vector[0..3] = {:?}", &decoded.vector[0..3]);
        println!("AFTER: expert_weights = {:?}", decoded.expert_weights);
        println!("AFTER: selected_experts = {:?}", decoded.selected_experts);
        println!("AFTER: content_hash = {:#x}", decoded.content_hash);

        assert_eq!(original.vector, decoded.vector);
        assert_eq!(original.expert_weights, decoded.expert_weights);
        assert_eq!(original.selected_experts, decoded.selected_experts);
        assert_eq!(original.pipeline_latency_us, decoded.pipeline_latency_us);
        assert_eq!(original.content_hash, decoded.content_hash);
        println!("PASSED: round-trip preserves all fields");
    }

    #[test]
    fn test_decode_fails_fast_on_invalid_magic() {
        let codec = EmbeddingBinaryCodec::new();
        let mut buffer = vec![0u8; EmbeddingBinaryCodec::MIN_BUFFER_SIZE];
        buffer[0..4].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]); // Bad magic

        println!("BEFORE: buffer[0..4] = {:02x?}", &buffer[0..4]);

        let result = codec.decode(&buffer);

        println!("AFTER: result = {:?}", result);
        assert!(matches!(result, Err(DecodeError::InvalidMagic)));
        println!("PASSED: decode fails fast on invalid magic");
    }

    #[test]
    fn test_decode_fails_fast_on_truncated_buffer() {
        let codec = EmbeddingBinaryCodec::new();
        let buffer = vec![0u8; 100]; // Way too short

        println!("BEFORE: buffer.len() = {}", buffer.len());

        let result = codec.decode(&buffer);

        println!("AFTER: result = {:?}", result);
        match result {
            Err(DecodeError::BufferTooShort { needed, available }) => {
                assert_eq!(needed, 6244);
                assert_eq!(available, 100);
            }
            _ => panic!("Expected BufferTooShort"),
        }
        println!("PASSED: decode fails fast on truncated buffer");
    }

    #[test]
    fn test_decode_fails_fast_on_unsupported_version() {
        let codec = EmbeddingBinaryCodec::new();
        let embedding = make_test_embedding();
        let mut bytes = codec.encode(&embedding).expect("encode");

        // Corrupt version to 99 (big-endian)
        bytes[4] = 0x00;
        bytes[5] = 0x63; // 99 in big-endian

        println!("BEFORE: version bytes = {:02x?}", &bytes[4..6]);

        let result = codec.decode(&bytes);

        println!("AFTER: result = {:?}", result);
        assert!(matches!(result, Err(DecodeError::UnsupportedVersion(99))));
        println!("PASSED: decode fails fast on unsupported version");
    }

    // ========== Big-Endian Tests ==========

    #[test]
    fn test_encode_uses_big_endian_floats() {
        let codec = EmbeddingBinaryCodec::new();
        let mut embedding = make_test_embedding();
        embedding.vector[0] = 1.0f32; // Known value

        let bytes = codec.encode(&embedding).expect("encode");

        // 1.0f32 in big-endian = 0x3F800000
        let expected_be = 1.0f32.to_be_bytes();
        println!("Expected BE bytes for 1.0: {:02x?}", expected_be);
        println!("Actual bytes at offset 64: {:02x?}", &bytes[64..68]);

        assert_eq!(&bytes[64..68], &expected_be);
        println!("PASSED: encode uses big-endian for floats");
    }

    #[test]
    fn test_decode_converts_big_endian_correctly() {
        let codec = EmbeddingBinaryCodec::new();
        let mut embedding = make_test_embedding();
        embedding.vector[0] = std::f32::consts::PI;

        println!("BEFORE: original.vector[0] = {}", embedding.vector[0]);

        let bytes = codec.encode(&embedding).expect("encode");
        let decoded = codec.decode(&bytes).expect("decode");

        println!("AFTER: decoded.vector[0] = {}", decoded.vector[0]);
        assert!((embedding.vector[0] - decoded.vector[0]).abs() < 1e-7);
        println!("PASSED: decode converts big-endian correctly (PI preserved)");
    }

    // ========== Edge Case Tests ==========

    #[test]
    fn test_edge_case_max_content_hash() {
        let codec = EmbeddingBinaryCodec::new();
        let mut embedding = make_test_embedding();
        embedding.content_hash = u64::MAX;

        println!("BEFORE: content_hash = {:#x}", embedding.content_hash);

        let bytes = codec.encode(&embedding).expect("encode");
        let decoded = codec.decode(&bytes).expect("decode");

        println!("AFTER: content_hash = {:#x}", decoded.content_hash);
        assert_eq!(decoded.content_hash, u64::MAX);
        println!("Edge Case PASSED: u64::MAX content_hash preserved");
    }

    #[test]
    fn test_edge_case_zero_vector() {
        let codec = EmbeddingBinaryCodec::new();
        let mut embedding = make_test_embedding();
        for v in &mut embedding.vector {
            *v = 0.0;
        }

        println!("BEFORE: all vector elements = 0.0");

        let bytes = codec.encode(&embedding).expect("encode");
        let decoded = codec.decode(&bytes).expect("decode");

        println!("AFTER: decoded.vector[0..5] = {:?}", &decoded.vector[0..5]);
        assert!(decoded.vector.iter().all(|&v| v == 0.0));
        println!("Edge Case PASSED: zero vector preserved");
    }

    #[test]
    fn test_edge_case_negative_floats() {
        let codec = EmbeddingBinaryCodec::new();
        let mut embedding = make_test_embedding();
        embedding.vector[0] = -1.5;
        embedding.vector[1] = f32::MIN;

        println!(
            "BEFORE: vector[0] = {}, vector[1] = {}",
            embedding.vector[0], embedding.vector[1]
        );

        let bytes = codec.encode(&embedding).expect("encode");
        let decoded = codec.decode(&bytes).expect("decode");

        println!(
            "AFTER: vector[0] = {}, vector[1] = {}",
            decoded.vector[0], decoded.vector[1]
        );
        assert_eq!(decoded.vector[0], -1.5);
        assert_eq!(decoded.vector[1], f32::MIN);
        println!("Edge Case PASSED: negative floats preserved");
    }

    #[test]
    fn test_edge_case_max_pipeline_latency() {
        let codec = EmbeddingBinaryCodec::new();
        let mut embedding = make_test_embedding();
        embedding.pipeline_latency_us = u64::MAX;

        println!(
            "BEFORE: pipeline_latency_us = {}",
            embedding.pipeline_latency_us
        );

        let bytes = codec.encode(&embedding).expect("encode");
        let decoded = codec.decode(&bytes).expect("decode");

        println!(
            "AFTER: pipeline_latency_us = {}",
            decoded.pipeline_latency_us
        );
        assert_eq!(decoded.pipeline_latency_us, u64::MAX);
        println!("Edge Case PASSED: u64::MAX pipeline_latency_us preserved");
    }

    #[test]
    fn test_encode_to_buffer() {
        let codec = EmbeddingBinaryCodec::new();
        let embedding = make_test_embedding();
        let mut buffer = vec![0u8; 10000];

        println!("BEFORE: buffer all zeros");

        let written = codec
            .encode_to_buffer(&embedding, &mut buffer)
            .expect("encode_to_buffer");

        println!("AFTER: written = {} bytes", written);
        println!(
            "AFTER: buffer[0..4] (magic) = {:02x?}",
            &buffer[0..4]
        );

        assert_eq!(written, 6244);
        assert_eq!(&buffer[0..4], &EMBEDDING_MAGIC);
        println!("PASSED: encode_to_buffer works correctly");
    }

    #[test]
    fn test_encode_to_buffer_too_small() {
        let codec = EmbeddingBinaryCodec::new();
        let embedding = make_test_embedding();
        let mut buffer = vec![0u8; 100]; // Too small

        let result = codec.encode_to_buffer(&embedding, &mut buffer);

        println!("Result: {:?}", result);
        match result {
            Err(EncodeError::BufferTooSmall { needed, available }) => {
                assert_eq!(needed, 6244);
                assert_eq!(available, 100);
            }
            _ => panic!("Expected BufferTooSmall"),
        }
        println!("PASSED: encode_to_buffer fails on small buffer");
    }

    #[test]
    fn test_serialized_size() {
        let codec = EmbeddingBinaryCodec::new();
        let embedding = make_test_embedding();

        let size = codec.serialized_size(&embedding);

        println!("BEFORE: Expected size = 6244");
        println!("AFTER: Actual size = {}", size);
        assert_eq!(size, 6244);
        println!("PASSED: serialized_size returns correct value");
    }

    #[test]
    fn test_decode_header_only() {
        let codec = EmbeddingBinaryCodec::new();
        let embedding = make_test_embedding();
        let bytes = codec.encode(&embedding).expect("encode");

        let header = codec.decode_header(&bytes).expect("decode_header");

        println!("Header magic: {:02x?}", header.magic);
        println!("Header version (BE): {:#06x}", header.version);
        println!(
            "Header content_hash (BE): {:#018x}",
            header.content_hash
        );

        assert_eq!(header.magic, EMBEDDING_MAGIC);
        assert_eq!(u16::from_be(header.version), 1);
        assert_eq!(u64::from_be(header.content_hash), 0xDEADBEEF);
        println!("PASSED: decode_header extracts header correctly");
    }

    #[test]
    fn test_special_float_values() {
        let codec = EmbeddingBinaryCodec::new();
        let mut embedding = make_test_embedding();

        // Test special float values
        embedding.vector[0] = f32::INFINITY;
        embedding.vector[1] = f32::NEG_INFINITY;
        embedding.vector[2] = 0.0;
        embedding.vector[3] = -0.0;
        // Note: NaN != NaN, so we skip NaN test for equality

        println!("BEFORE: INF={}, NEG_INF={}, ZERO={}, NEG_ZERO={}",
            embedding.vector[0], embedding.vector[1], embedding.vector[2], embedding.vector[3]);

        let bytes = codec.encode(&embedding).expect("encode");
        let decoded = codec.decode(&bytes).expect("decode");

        println!("AFTER: INF={}, NEG_INF={}, ZERO={}, NEG_ZERO={}",
            decoded.vector[0], decoded.vector[1], decoded.vector[2], decoded.vector[3]);

        assert!(decoded.vector[0].is_infinite() && decoded.vector[0].is_sign_positive());
        assert!(decoded.vector[1].is_infinite() && decoded.vector[1].is_sign_negative());
        assert_eq!(decoded.vector[2], 0.0);
        assert_eq!(decoded.vector[3], -0.0);
        println!("PASSED: special float values preserved");
    }
}
