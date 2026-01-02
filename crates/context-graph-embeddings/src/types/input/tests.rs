//! Tests for input types module.

#![cfg(test)]

use super::*;
use crate::error::EmbeddingError;

// ============================================================
// TEXT CONSTRUCTION TESTS (5 tests)
// ============================================================

#[test]
fn test_text_with_valid_content_succeeds() {
    let input = ModelInput::text("Hello, world!");
    assert!(input.is_ok());
    let input = input.unwrap();
    assert!(input.is_text());
    let (content, instruction) = input.as_text().unwrap();
    assert_eq!(content, "Hello, world!");
    assert!(instruction.is_none());
}

#[test]
fn test_text_with_empty_content_returns_invalid_input_error() {
    println!("BEFORE: Attempting to create ModelInput::text(\"\")");
    let result = ModelInput::text("");
    println!("AFTER: Result = {:?}", result);

    assert!(result.is_err());
    assert!(
        matches!(result, Err(EmbeddingError::EmptyInput)),
        "Expected EmptyInput error"
    );
}

#[test]
fn test_text_with_instruction_succeeds() {
    let input = ModelInput::text_with_instruction("What is Rust?", "query:");
    assert!(input.is_ok());
    let input = input.unwrap();
    let (content, instruction) = input.as_text().unwrap();
    assert_eq!(content, "What is Rust?");
    assert_eq!(instruction, Some("query:"));
}

#[test]
fn test_text_with_instruction_stores_instruction_correctly() {
    let input = ModelInput::text_with_instruction("Document content", "passage:").unwrap();
    let (_, instruction) = input.as_text().unwrap();
    assert_eq!(instruction, Some("passage:"));

    // Empty instruction is allowed
    let input2 = ModelInput::text_with_instruction("Content", "").unwrap();
    let (_, instruction2) = input2.as_text().unwrap();
    assert_eq!(instruction2, Some(""));
}

#[test]
fn test_text_with_instruction_empty_content_returns_invalid_input() {
    let result = ModelInput::text_with_instruction("", "query:");
    assert!(result.is_err());
    assert!(
        matches!(result, Err(EmbeddingError::EmptyInput)),
        "Expected EmptyInput error"
    );
}

// ============================================================
// CODE CONSTRUCTION TESTS (4 tests)
// ============================================================

#[test]
fn test_code_with_valid_content_and_language_succeeds() {
    let input = ModelInput::code("fn main() {}", "rust");
    assert!(input.is_ok());
    let input = input.unwrap();
    assert!(input.is_code());
    let (content, language) = input.as_code().unwrap();
    assert_eq!(content, "fn main() {}");
    assert_eq!(language, "rust");
}

#[test]
fn test_code_with_empty_content_returns_empty_input() {
    let result = ModelInput::code("", "rust");
    assert!(result.is_err());
    assert!(
        matches!(result, Err(EmbeddingError::EmptyInput)),
        "Expected EmptyInput error for empty content"
    );
}

#[test]
fn test_code_with_empty_language_returns_config_error() {
    let result = ModelInput::code("fn main() {}", "");
    assert!(result.is_err());
    match result {
        Err(EmbeddingError::ConfigError { message }) => {
            assert!(message.contains("language") && message.contains("empty"));
        }
        _ => panic!("Expected ConfigError for empty language"),
    }
}

#[test]
fn test_code_stores_language_as_provided() {
    // Language is stored as-is (no normalization)
    let input = ModelInput::code("code", "Rust").unwrap();
    let (_, language) = input.as_code().unwrap();
    assert_eq!(language, "Rust");

    let input2 = ModelInput::code("code", "PYTHON").unwrap();
    let (_, language2) = input2.as_code().unwrap();
    assert_eq!(language2, "PYTHON");
}

// ============================================================
// IMAGE CONSTRUCTION TESTS (3 tests)
// ============================================================

#[test]
fn test_image_with_valid_bytes_succeeds() {
    let bytes = vec![0x89, 0x50, 0x4E, 0x47]; // PNG magic
    let input = ModelInput::image(bytes.clone(), ImageFormat::Png);
    assert!(input.is_ok());
    let input = input.unwrap();
    assert!(input.is_image());
    let (img_bytes, format) = input.as_image().unwrap();
    assert_eq!(img_bytes, bytes.as_slice());
    assert_eq!(format, ImageFormat::Png);
}

#[test]
fn test_image_with_empty_bytes_returns_empty_input() {
    let result = ModelInput::image(vec![], ImageFormat::Png);
    assert!(result.is_err());
    assert!(
        matches!(result, Err(EmbeddingError::EmptyInput)),
        "Expected EmptyInput error"
    );
}

#[test]
fn test_image_stores_format_correctly() {
    let bytes = vec![1, 2, 3, 4];

    for format in [
        ImageFormat::Png,
        ImageFormat::Jpeg,
        ImageFormat::WebP,
        ImageFormat::Gif,
    ] {
        let input = ModelInput::image(bytes.clone(), format).unwrap();
        let (_, stored_format) = input.as_image().unwrap();
        assert_eq!(stored_format, format);
    }
}

// ============================================================
// AUDIO CONSTRUCTION TESTS (5 tests)
// ============================================================

#[test]
fn test_audio_with_valid_parameters_succeeds() {
    let bytes = vec![0u8; 1024];
    let input = ModelInput::audio(bytes.clone(), 16000, 1);
    assert!(input.is_ok());
    let input = input.unwrap();
    assert!(input.is_audio());
    let (audio_bytes, sample_rate, channels) = input.as_audio().unwrap();
    assert_eq!(audio_bytes, bytes.as_slice());
    assert_eq!(sample_rate, 16000);
    assert_eq!(channels, 1);
}

#[test]
fn test_audio_with_empty_bytes_returns_empty_input() {
    let result = ModelInput::audio(vec![], 16000, 1);
    assert!(result.is_err());
    assert!(
        matches!(result, Err(EmbeddingError::EmptyInput)),
        "Expected EmptyInput error"
    );
}

#[test]
fn test_audio_with_sample_rate_zero_returns_config_error() {
    println!("BEFORE: audio bytes=[1,2,3], sample_rate=0, channels=1");
    let result = ModelInput::audio(vec![1, 2, 3], 0, 1);
    println!("AFTER: sample_rate=0 result = {:?}", result);

    assert!(result.is_err());
    match result {
        Err(EmbeddingError::ConfigError { message }) => {
            assert!(message.contains("sample_rate"));
        }
        _ => panic!("Expected ConfigError"),
    }
}

#[test]
fn test_audio_with_channels_zero_returns_config_error() {
    println!("BEFORE: audio bytes=[1,2,3], sample_rate=16000, channels=0");
    let result = ModelInput::audio(vec![1, 2, 3], 16000, 0);
    println!("AFTER: channels=0 result = {:?}", result);

    assert!(result.is_err());
    match result {
        Err(EmbeddingError::ConfigError { message }) => {
            assert!(
                message.contains("channels") && message.contains("1") && message.contains("2")
            );
        }
        _ => panic!("Expected ConfigError"),
    }
}

#[test]
fn test_audio_with_channels_three_returns_config_error() {
    println!("BEFORE: audio bytes=[1,2,3], sample_rate=16000, channels=3");
    let result = ModelInput::audio(vec![1, 2, 3], 16000, 3);
    println!("AFTER: channels=3 result = {:?}", result);

    assert!(result.is_err());
    match result {
        Err(EmbeddingError::ConfigError { message }) => {
            assert!(message.contains("3"));
        }
        _ => panic!("Expected ConfigError"),
    }
}

// ============================================================
// CONTENT HASH TESTS (4 tests)
// ============================================================

#[test]
fn test_content_hash_same_for_identical_text_inputs() {
    let input1 = ModelInput::text("test content").unwrap();
    let hash1 = input1.content_hash();
    println!("BEFORE: First hash = {}", hash1);

    let input2 = ModelInput::text("test content").unwrap();
    let hash2 = input2.content_hash();
    println!("AFTER: Second hash = {}", hash2);

    assert_eq!(
        hash1, hash2,
        "Identical inputs must produce identical hashes"
    );
}

#[test]
fn test_content_hash_different_for_different_content() {
    let input1 = ModelInput::text("Hello").unwrap();
    let input2 = ModelInput::text("World").unwrap();
    let input3 = ModelInput::text("hello").unwrap(); // case sensitive

    assert_ne!(input1.content_hash(), input2.content_hash());
    assert_ne!(input1.content_hash(), input3.content_hash());
}

#[test]
fn test_content_hash_includes_instruction_in_hash() {
    let without_inst = ModelInput::text("content").unwrap();
    let with_inst = ModelInput::text_with_instruction("content", "query:").unwrap();
    let with_empty_inst = ModelInput::text_with_instruction("content", "").unwrap();

    // All three should have different hashes
    assert_ne!(without_inst.content_hash(), with_inst.content_hash());
    assert_ne!(without_inst.content_hash(), with_empty_inst.content_hash());
    assert_ne!(with_inst.content_hash(), with_empty_inst.content_hash());
}

#[test]
fn test_content_hash_includes_all_fields_for_each_variant() {
    // Code: different language = different hash
    let code1 = ModelInput::code("code", "rust").unwrap();
    let code2 = ModelInput::code("code", "python").unwrap();
    assert_ne!(code1.content_hash(), code2.content_hash());

    // Image: different format = different hash
    let img1 = ModelInput::image(vec![1, 2, 3], ImageFormat::Png).unwrap();
    let img2 = ModelInput::image(vec![1, 2, 3], ImageFormat::Jpeg).unwrap();
    assert_ne!(img1.content_hash(), img2.content_hash());

    // Audio: different sample_rate = different hash
    let audio1 = ModelInput::audio(vec![1, 2, 3], 16000, 1).unwrap();
    let audio2 = ModelInput::audio(vec![1, 2, 3], 44100, 1).unwrap();
    assert_ne!(audio1.content_hash(), audio2.content_hash());

    // Audio: different channels = different hash
    let audio3 = ModelInput::audio(vec![1, 2, 3], 16000, 2).unwrap();
    assert_ne!(audio1.content_hash(), audio3.content_hash());
}

// ============================================================
// BYTE SIZE TESTS (3 tests)
// ============================================================

#[test]
fn test_byte_size_returns_correct_size_for_text() {
    let content = "Hello, world!";
    let input = ModelInput::text(content).unwrap();
    let size = input.byte_size();
    assert_eq!(size, content.len());

    // With instruction
    let input_with_inst = ModelInput::text_with_instruction("Hello", "query:").unwrap();
    assert_eq!(input_with_inst.byte_size(), "Hello".len() + "query:".len());
}

#[test]
fn test_byte_size_returns_correct_size_for_image_bytes() {
    let bytes = vec![0u8; 1000];
    let input = ModelInput::image(bytes.clone(), ImageFormat::Png).unwrap();
    assert_eq!(input.byte_size(), 1000);
}

#[test]
fn test_byte_size_includes_string_heap_allocation() {
    let code_content = "fn main() {}";
    let language = "rust";
    let input = ModelInput::code(code_content, language).unwrap();

    // Should include both strings
    assert_eq!(input.byte_size(), code_content.len() + language.len());
}

// ============================================================
// TYPE PREDICATE TESTS (4 tests)
// ============================================================

#[test]
fn test_is_text_returns_true_only_for_text_variant() {
    let text = ModelInput::text("hello").unwrap();
    let code = ModelInput::code("code", "rust").unwrap();
    let image = ModelInput::image(vec![1], ImageFormat::Png).unwrap();
    let audio = ModelInput::audio(vec![1], 16000, 1).unwrap();

    assert!(text.is_text());
    assert!(!code.is_text());
    assert!(!image.is_text());
    assert!(!audio.is_text());
}

#[test]
fn test_is_code_returns_true_only_for_code_variant() {
    let text = ModelInput::text("hello").unwrap();
    let code = ModelInput::code("code", "rust").unwrap();
    let image = ModelInput::image(vec![1], ImageFormat::Png).unwrap();
    let audio = ModelInput::audio(vec![1], 16000, 1).unwrap();

    assert!(!text.is_code());
    assert!(code.is_code());
    assert!(!image.is_code());
    assert!(!audio.is_code());
}

#[test]
fn test_is_image_returns_true_only_for_image_variant() {
    let text = ModelInput::text("hello").unwrap();
    let code = ModelInput::code("code", "rust").unwrap();
    let image = ModelInput::image(vec![1], ImageFormat::Png).unwrap();
    let audio = ModelInput::audio(vec![1], 16000, 1).unwrap();

    assert!(!text.is_image());
    assert!(!code.is_image());
    assert!(image.is_image());
    assert!(!audio.is_image());
}

#[test]
fn test_is_audio_returns_true_only_for_audio_variant() {
    let text = ModelInput::text("hello").unwrap();
    let code = ModelInput::code("code", "rust").unwrap();
    let image = ModelInput::image(vec![1], ImageFormat::Png).unwrap();
    let audio = ModelInput::audio(vec![1], 16000, 1).unwrap();

    assert!(!text.is_audio());
    assert!(!code.is_audio());
    assert!(!image.is_audio());
    assert!(audio.is_audio());
}

// ============================================================
// IMAGE FORMAT TESTS (8 tests)
// ============================================================

#[test]
fn test_image_format_mime_type() {
    assert_eq!(ImageFormat::Png.mime_type(), "image/png");
    assert_eq!(ImageFormat::Jpeg.mime_type(), "image/jpeg");
    assert_eq!(ImageFormat::WebP.mime_type(), "image/webp");
    assert_eq!(ImageFormat::Gif.mime_type(), "image/gif");
}

#[test]
fn test_image_format_extension() {
    assert_eq!(ImageFormat::Png.extension(), "png");
    assert_eq!(ImageFormat::Jpeg.extension(), "jpg");
    assert_eq!(ImageFormat::WebP.extension(), "webp");
    assert_eq!(ImageFormat::Gif.extension(), "gif");
}

#[test]
fn test_image_format_detect_png() {
    let png_bytes = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    assert_eq!(ImageFormat::detect(&png_bytes), Some(ImageFormat::Png));
}

#[test]
fn test_image_format_detect_jpeg() {
    let jpeg_bytes = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46];
    assert_eq!(ImageFormat::detect(&jpeg_bytes), Some(ImageFormat::Jpeg));
}

#[test]
fn test_image_format_detect_gif() {
    let gif_bytes = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61]; // GIF89a
    assert_eq!(ImageFormat::detect(&gif_bytes), Some(ImageFormat::Gif));
}

#[test]
fn test_image_format_detect_webp() {
    let webp_bytes = [
        0x52, 0x49, 0x46, 0x46, // RIFF
        0x00, 0x00, 0x00, 0x00, // size (ignored)
        0x57, 0x45, 0x42, 0x50, // WEBP
    ];
    assert_eq!(ImageFormat::detect(&webp_bytes), Some(ImageFormat::WebP));
}

#[test]
fn test_image_format_detect_unknown() {
    let unknown = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05];
    assert_eq!(ImageFormat::detect(&unknown), None);
}

#[test]
fn test_image_format_detect_too_short() {
    let short = [0x89, 0x50, 0x4E]; // Only 3 bytes, need 4 for PNG
    assert_eq!(ImageFormat::detect(&short), None);
}

// ============================================================
// ACCESSOR TESTS (4 tests)
// ============================================================

#[test]
fn test_as_text_returns_none_for_non_text() {
    let code = ModelInput::code("code", "rust").unwrap();
    assert!(code.as_text().is_none());

    let image = ModelInput::image(vec![1], ImageFormat::Png).unwrap();
    assert!(image.as_text().is_none());
}

#[test]
fn test_as_code_returns_none_for_non_code() {
    let text = ModelInput::text("hello").unwrap();
    assert!(text.as_code().is_none());

    let audio = ModelInput::audio(vec![1], 16000, 1).unwrap();
    assert!(audio.as_code().is_none());
}

#[test]
fn test_as_image_returns_none_for_non_image() {
    let text = ModelInput::text("hello").unwrap();
    assert!(text.as_image().is_none());

    let code = ModelInput::code("code", "rust").unwrap();
    assert!(code.as_image().is_none());
}

#[test]
fn test_as_audio_returns_none_for_non_audio() {
    let text = ModelInput::text("hello").unwrap();
    assert!(text.as_audio().is_none());

    let image = ModelInput::image(vec![1], ImageFormat::Png).unwrap();
    assert!(image.as_audio().is_none());
}

// ============================================================
// DISPLAY AND SERDE TESTS (2 tests)
// ============================================================

#[test]
fn test_display_formatting() {
    let text = ModelInput::text("Hello, world!").unwrap();
    let display = format!("{}", text);
    assert!(display.contains("Text"));
    assert!(display.contains("Hello"));

    let code = ModelInput::code("fn main() {}", "rust").unwrap();
    let display = format!("{}", code);
    assert!(display.contains("Code"));
    assert!(display.contains("rust"));

    let image = ModelInput::image(vec![1, 2, 3], ImageFormat::Png).unwrap();
    let display = format!("{}", image);
    assert!(display.contains("Image"));
    assert!(display.contains("PNG"));
    assert!(display.contains("3 bytes"));

    let audio = ModelInput::audio(vec![1, 2, 3, 4], 44100, 2).unwrap();
    let display = format!("{}", audio);
    assert!(display.contains("Audio"));
    assert!(display.contains("44100"));
    assert!(display.contains("stereo"));
}

#[test]
fn test_serde_round_trip() {
    let text = ModelInput::text("Hello").unwrap();
    let json = serde_json::to_string(&text).unwrap();
    let recovered: ModelInput = serde_json::from_str(&json).unwrap();
    assert_eq!(text, recovered);

    let code = ModelInput::code("fn main() {}", "rust").unwrap();
    let json = serde_json::to_string(&code).unwrap();
    let recovered: ModelInput = serde_json::from_str(&json).unwrap();
    assert_eq!(code, recovered);

    let image = ModelInput::image(vec![1, 2, 3], ImageFormat::Jpeg).unwrap();
    let json = serde_json::to_string(&image).unwrap();
    let recovered: ModelInput = serde_json::from_str(&json).unwrap();
    assert_eq!(image, recovered);

    let audio = ModelInput::audio(vec![1, 2, 3], 16000, 1).unwrap();
    let json = serde_json::to_string(&audio).unwrap();
    let recovered: ModelInput = serde_json::from_str(&json).unwrap();
    assert_eq!(audio, recovered);
}

// ============================================================
// STEREO AUDIO TEST
// ============================================================

#[test]
fn test_audio_stereo_accepted() {
    let result = ModelInput::audio(vec![1, 2, 3], 44100, 2);
    assert!(result.is_ok());
    let input = result.unwrap();
    let (_, sample_rate, channels) = input.as_audio().unwrap();
    assert_eq!(sample_rate, 44100);
    assert_eq!(channels, 2);
}

// ============================================================
// EDGE CASE: HASH DETERMINISM ACROSS CALLS
// ============================================================

#[test]
fn test_hash_determinism_multiple_calls() {
    let input = ModelInput::text("deterministic content").unwrap();

    let hash1 = input.content_hash();
    let hash2 = input.content_hash();
    let hash3 = input.content_hash();

    assert_eq!(hash1, hash2);
    assert_eq!(hash2, hash3);
    println!(
        "Hash determinism verified: {} == {} == {}",
        hash1, hash2, hash3
    );
}

// ============================================================
// EDGE CASE: LARGE INPUTS
// ============================================================

#[test]
fn test_large_text_input() {
    let large_content = "x".repeat(100_000);
    let input = ModelInput::text(&large_content).unwrap();
    assert_eq!(input.byte_size(), 100_000);

    // Hash should still work
    let hash = input.content_hash();
    assert_ne!(hash, 0);
}

#[test]
fn test_large_image_input() {
    let large_bytes = vec![0xFFu8; 1_000_000]; // 1MB
    let input = ModelInput::image(large_bytes, ImageFormat::Jpeg).unwrap();
    assert_eq!(input.byte_size(), 1_000_000);

    // Hash should still work
    let hash = input.content_hash();
    assert_ne!(hash, 0);
}

// ============================================================
// INPUT TYPE TESTS (M03-F07)
// ============================================================

#[test]
fn test_input_type_from_model_input_text() {
    println!("BEFORE: Creating ModelInput::Text");
    let input = ModelInput::text("Hello").unwrap();
    println!("AFTER: Created input = {:?}", input);

    let input_type = InputType::from(&input);
    println!("RESULT: InputType = {:?}", input_type);

    assert_eq!(input_type, InputType::Text);
}

#[test]
fn test_input_type_from_model_input_code() {
    println!("BEFORE: Creating ModelInput::Code");
    let input = ModelInput::code("fn main() {}", "rust").unwrap();
    println!("AFTER: Created input = {:?}", input);

    let input_type = InputType::from(&input);
    println!("RESULT: InputType = {:?}", input_type);

    assert_eq!(input_type, InputType::Code);
}

#[test]
fn test_input_type_from_model_input_image() {
    println!("BEFORE: Creating ModelInput::Image");
    let input = ModelInput::image(vec![1, 2, 3, 4], ImageFormat::Png).unwrap();
    println!("AFTER: Created input = {:?}", input);

    let input_type = InputType::from(&input);
    println!("RESULT: InputType = {:?}", input_type);

    assert_eq!(input_type, InputType::Image);
}

#[test]
fn test_input_type_from_model_input_audio() {
    println!("BEFORE: Creating ModelInput::Audio");
    let input = ModelInput::audio(vec![1, 2, 3, 4], 16000, 1).unwrap();
    println!("AFTER: Created input = {:?}", input);

    let input_type = InputType::from(&input);
    println!("RESULT: InputType = {:?}", input_type);

    assert_eq!(input_type, InputType::Audio);
}

#[test]
fn test_input_type_display_lowercase() {
    assert_eq!(format!("{}", InputType::Text), "text");
    assert_eq!(format!("{}", InputType::Code), "code");
    assert_eq!(format!("{}", InputType::Image), "image");
    assert_eq!(format!("{}", InputType::Audio), "audio");
}

#[test]
fn test_input_type_all_returns_4_variants() {
    let all = InputType::all();
    assert_eq!(all.len(), 4);
    assert_eq!(all[0], InputType::Text);
    assert_eq!(all[1], InputType::Code);
    assert_eq!(all[2], InputType::Image);
    assert_eq!(all[3], InputType::Audio);
}

#[test]
fn test_input_type_can_be_used_as_hashmap_key() {
    use std::collections::HashMap;

    let mut map: HashMap<InputType, &str> = HashMap::new();
    map.insert(InputType::Text, "text_value");
    map.insert(InputType::Code, "code_value");
    map.insert(InputType::Image, "image_value");
    map.insert(InputType::Audio, "audio_value");

    assert_eq!(map.get(&InputType::Text), Some(&"text_value"));
    assert_eq!(map.get(&InputType::Code), Some(&"code_value"));
    assert_eq!(map.get(&InputType::Image), Some(&"image_value"));
    assert_eq!(map.get(&InputType::Audio), Some(&"audio_value"));
}

#[test]
fn test_input_type_can_be_used_in_hashset() {
    use std::collections::HashSet;

    let mut set: HashSet<InputType> = HashSet::new();
    set.insert(InputType::Text);
    set.insert(InputType::Code);

    assert!(set.contains(&InputType::Text));
    assert!(set.contains(&InputType::Code));
    assert!(!set.contains(&InputType::Image));
    assert!(!set.contains(&InputType::Audio));
}

#[test]
fn test_input_type_copy_semantics() {
    let original = InputType::Text;
    let copied = original; // Copy, not move
    assert_eq!(original, copied); // Both still valid
}

#[test]
fn test_input_type_serde_roundtrip() {
    for input_type in InputType::all() {
        let json = serde_json::to_string(input_type).unwrap();
        let recovered: InputType = serde_json::from_str(&json).unwrap();
        assert_eq!(*input_type, recovered);
        println!(
            "Serialized {:?} as {} and recovered successfully",
            input_type, json
        );
    }
}

#[test]
fn test_input_type_discriminant_values() {
    assert_eq!(InputType::Text.discriminant(), 0);
    assert_eq!(InputType::Code.discriminant(), 1);
    assert_eq!(InputType::Image.discriminant(), 2);
    assert_eq!(InputType::Audio.discriminant(), 3);
}

#[test]
fn test_input_type_debug_formatting() {
    // Debug should show variant name
    assert_eq!(format!("{:?}", InputType::Text), "Text");
    assert_eq!(format!("{:?}", InputType::Code), "Code");
    assert_eq!(format!("{:?}", InputType::Image), "Image");
    assert_eq!(format!("{:?}", InputType::Audio), "Audio");
}

#[test]
fn test_input_type_equality() {
    // Same types should be equal
    assert_eq!(InputType::Text, InputType::Text);
    assert_eq!(InputType::Code, InputType::Code);
    assert_eq!(InputType::Image, InputType::Image);
    assert_eq!(InputType::Audio, InputType::Audio);

    // Different types should not be equal
    assert_ne!(InputType::Text, InputType::Code);
    assert_ne!(InputType::Text, InputType::Image);
    assert_ne!(InputType::Text, InputType::Audio);
    assert_ne!(InputType::Code, InputType::Image);
    assert_ne!(InputType::Code, InputType::Audio);
    assert_ne!(InputType::Image, InputType::Audio);
}
