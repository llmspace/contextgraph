//! Manual tests for the Memory module (TASK-P1-001)
//!
//! These tests verify:
//! 1. Memory creation with synthetic data
//! 2. Serialization roundtrips
//! 3. Validation edge cases
//! 4. Source type discrimination

#![cfg(feature = "test-utils")]

use context_graph_core::memory::{
    ChunkMetadata, HookType, Memory, MemorySource, ResponseType, MAX_CONTENT_LENGTH,
};
use context_graph_core::types::fingerprint::SemanticFingerprint;
use uuid::Uuid;

/// Helper to create a test fingerprint
fn test_fingerprint() -> SemanticFingerprint {
    SemanticFingerprint::zeroed()
}

// =============================================================================
// Test 1: Create Memory and Verify Structure
// =============================================================================

#[test]
fn manual_test_memory_creation_structure() {
    println!("=== TEST 1: Create Memory and Verify Structure ===");

    // Synthetic input
    let content = "User asked about implementing a REST API endpoint".to_string();
    let hook_type = HookType::UserPromptSubmit;
    let session_id = "test-session-001".to_string();
    let fp = test_fingerprint();

    println!("Input content: {:?}", content);
    println!("Input hook_type: {:?}", hook_type);
    println!("Input session_id: {:?}", session_id);

    let memory = Memory::new(
        content.clone(),
        MemorySource::HookDescription {
            hook_type,
            tool_name: None,
        },
        session_id.clone(),
        fp,
        None,
    );

    println!("\n--- OUTPUT ---");
    println!("Memory ID: {}", memory.id);
    println!("Memory ID is non-nil: {}", !memory.id.is_nil());
    println!("Memory content: {:?}", memory.content);
    println!("Memory session_id: {:?}", memory.session_id);
    println!("Memory word_count: {}", memory.word_count);
    println!("Memory created_at: {:?}", memory.created_at);

    // Verify expected outputs
    assert!(!memory.id.is_nil(), "ID should be non-nil UUID v4");
    assert_eq!(
        memory.content, content,
        "Content should match input exactly"
    );
    assert_eq!(memory.session_id, session_id, "Session ID should match");
    assert_eq!(memory.word_count, 8, "Word count should be 8");

    // Verify timestamp is recent (within last minute)
    let now = chrono::Utc::now();
    let diff = now - memory.created_at;
    assert!(
        diff.num_seconds() < 60,
        "Created timestamp should be within last minute"
    );

    println!("\n=== TEST 1 PASSED ===\n");
}

// =============================================================================
// Test 2: Serialization Size Check
// =============================================================================

#[test]
fn manual_test_serialization_size() {
    println!("=== TEST 2: Serialization Size Check ===");

    let fp = test_fingerprint();
    let memory = Memory::new(
        "Test content for size check".to_string(),
        MemorySource::HookDescription {
            hook_type: HookType::PostToolUse,
            tool_name: Some("Edit".to_string()),
        },
        "session-size-test".to_string(),
        fp,
        None,
    );

    // Serialize with bincode
    let bytes = bincode::serialize(&memory).expect("serialization failed");
    let size = bytes.len();

    println!(
        "Serialized size: {} bytes ({:.2} KB)",
        size,
        size as f64 / 1024.0
    );
    println!("Estimated size: {} bytes", memory.estimated_size());

    // Verify size is reasonable (< 50KB for zeroed fingerprint)
    assert!(
        size < 50_000,
        "Serialized size should be < 50KB, got {} bytes",
        size
    );

    // Verify deserialization works
    let restored: Memory = bincode::deserialize(&bytes).expect("deserialization failed");
    assert_eq!(memory.id, restored.id);
    assert_eq!(memory.content, restored.content);

    println!("\n=== TEST 2 PASSED ===\n");
}

// =============================================================================
// Test 3: Source Type Discrimination
// =============================================================================

#[test]
fn manual_test_source_type_discrimination() {
    println!("=== TEST 3: Source Type Discrimination ===");

    let fp = test_fingerprint();

    // HookDescription memory
    let hook_mem = Memory::new(
        "Edited file config.rs".to_string(),
        MemorySource::HookDescription {
            hook_type: HookType::PostToolUse,
            tool_name: Some("Edit".to_string()),
        },
        "session".to_string(),
        fp.clone(),
        None,
    );

    println!("Hook memory:");
    println!("  is_hook_description: {}", hook_mem.is_hook_description());
    println!("  is_claude_response: {}", hook_mem.is_claude_response());
    println!("  is_md_file_chunk: {}", hook_mem.is_md_file_chunk());
    println!("  hook_type: {:?}", hook_mem.hook_type());
    println!("  tool_name: {:?}", hook_mem.tool_name());

    assert!(hook_mem.is_hook_description());
    assert!(!hook_mem.is_claude_response());
    assert!(!hook_mem.is_md_file_chunk());
    assert_eq!(hook_mem.hook_type(), Some(HookType::PostToolUse));
    assert_eq!(hook_mem.tool_name(), Some("Edit"));

    // ClaudeResponse memory
    let response_mem = Memory::new(
        "Session summary: User implemented REST API".to_string(),
        MemorySource::ClaudeResponse {
            response_type: ResponseType::SessionSummary,
        },
        "session".to_string(),
        fp.clone(),
        None,
    );

    println!("\nResponse memory:");
    println!(
        "  is_hook_description: {}",
        response_mem.is_hook_description()
    );
    println!(
        "  is_claude_response: {}",
        response_mem.is_claude_response()
    );
    println!("  is_md_file_chunk: {}", response_mem.is_md_file_chunk());
    println!("  response_type: {:?}", response_mem.response_type());

    assert!(!response_mem.is_hook_description());
    assert!(response_mem.is_claude_response());
    assert!(!response_mem.is_md_file_chunk());
    assert_eq!(
        response_mem.response_type(),
        Some(ResponseType::SessionSummary)
    );

    // MDFileChunk memory
    let chunk_mem = Memory::new(
        "# README\n\nThis is documentation chunk.".to_string(),
        MemorySource::MDFileChunk {
            file_path: "README.md".to_string(),
            chunk_index: 0,
            total_chunks: 3,
        },
        "session".to_string(),
        fp,
        Some(ChunkMetadata {
            file_path: "README.md".to_string(),
            chunk_index: 0,
            total_chunks: 3,
            word_offset: 0,
            char_offset: 0,
            original_file_hash: "abc123def456".to_string(),
            start_line: 1,
            end_line: 3,
        }),
    );

    println!("\nMDFileChunk memory:");
    println!("  is_hook_description: {}", chunk_mem.is_hook_description());
    println!("  is_claude_response: {}", chunk_mem.is_claude_response());
    println!("  is_md_file_chunk: {}", chunk_mem.is_md_file_chunk());
    println!(
        "  has_chunk_metadata: {}",
        chunk_mem.chunk_metadata.is_some()
    );

    assert!(!chunk_mem.is_hook_description());
    assert!(!chunk_mem.is_claude_response());
    assert!(chunk_mem.is_md_file_chunk());
    assert!(chunk_mem.chunk_metadata.is_some());

    println!("\n=== TEST 3 PASSED ===\n");
}

// =============================================================================
// Edge Case 1: Empty Content
// =============================================================================

#[test]
fn manual_test_edge_case_empty_content() {
    println!("=== EDGE CASE 1: Empty Content ===");

    let fp = test_fingerprint();
    let memory = Memory::new(
        String::new(),
        MemorySource::HookDescription {
            hook_type: HookType::SessionStart,
            tool_name: None,
        },
        "session".to_string(),
        fp,
        None,
    );

    println!("Input: empty content string");
    println!(
        "State before validation: content.is_empty() = {}",
        memory.content.is_empty()
    );

    let result = memory.validate();

    println!("Validation result: {:?}", result);
    println!(
        "State after validation: result.is_err() = {}",
        result.is_err()
    );

    assert!(result.is_err(), "Validation should fail for empty content");
    let err = result.unwrap_err();
    assert!(
        err.contains("empty"),
        "Error message should mention 'empty', got: {}",
        err
    );

    println!("\n=== EDGE CASE 1 PASSED ===\n");
}

// =============================================================================
// Edge Case 2: Maximum Content Length
// =============================================================================

#[test]
fn manual_test_edge_case_max_content_length() {
    println!("=== EDGE CASE 2: Maximum Content Length ===");

    let fp = test_fingerprint();

    // Exactly at limit should pass
    let exactly_limit = "x".repeat(MAX_CONTENT_LENGTH);
    let memory_at_limit = Memory::new(
        exactly_limit.clone(),
        MemorySource::HookDescription {
            hook_type: HookType::SessionStart,
            tool_name: None,
        },
        "session".to_string(),
        fp.clone(),
        None,
    );

    println!(
        "Test 1: Content exactly at limit ({} chars)",
        MAX_CONTENT_LENGTH
    );
    println!("  Content length: {}", memory_at_limit.content.len());

    let result = memory_at_limit.validate();
    println!("  Validation result: {:?}", result.is_ok());
    assert!(
        result.is_ok(),
        "Content at exactly {} should pass",
        MAX_CONTENT_LENGTH
    );

    // Over limit should fail
    let over_limit = "x".repeat(MAX_CONTENT_LENGTH + 1);
    let memory_over_limit = Memory::new(
        over_limit,
        MemorySource::HookDescription {
            hook_type: HookType::SessionStart,
            tool_name: None,
        },
        "session".to_string(),
        fp,
        None,
    );

    println!(
        "\nTest 2: Content over limit ({} chars)",
        MAX_CONTENT_LENGTH + 1
    );
    println!("  Content length: {}", memory_over_limit.content.len());

    let result = memory_over_limit.validate();
    println!("  Validation result: {:?}", result);
    assert!(
        result.is_err(),
        "Content over {} should fail",
        MAX_CONTENT_LENGTH
    );
    let err = result.unwrap_err();
    assert!(
        err.contains("exceeds"),
        "Error should mention 'exceeds', got: {}",
        err
    );

    println!("\n=== EDGE CASE 2 PASSED ===\n");
}

// =============================================================================
// Edge Case 3: MDFileChunk without ChunkMetadata
// =============================================================================

#[test]
fn manual_test_edge_case_mdfilechunk_missing_metadata() {
    println!("=== EDGE CASE 3: MDFileChunk without ChunkMetadata ===");

    let fp = test_fingerprint();
    let memory = Memory::new(
        "Some markdown content".to_string(),
        MemorySource::MDFileChunk {
            file_path: "test.md".to_string(),
            chunk_index: 0,
            total_chunks: 1,
        },
        "session".to_string(),
        fp,
        None, // Missing chunk_metadata!
    );

    println!("Input: MDFileChunk source with chunk_metadata = None");
    println!("State before validation:");
    println!("  source: {:?}", memory.source);
    println!(
        "  chunk_metadata.is_none(): {}",
        memory.chunk_metadata.is_none()
    );

    let result = memory.validate();

    println!("Validation result: {:?}", result);
    println!(
        "State after validation: result.is_err() = {}",
        result.is_err()
    );

    assert!(
        result.is_err(),
        "Validation should fail for MDFileChunk without metadata"
    );
    let err = result.unwrap_err();
    assert!(
        err.contains("chunk_metadata"),
        "Error should mention 'chunk_metadata', got: {}",
        err
    );

    println!("\n=== EDGE CASE 3 PASSED ===\n");
}

// =============================================================================
// All Hook Types Test
// =============================================================================

#[test]
fn manual_test_all_hook_types() {
    println!("=== TEST: All Hook Types ===");

    let fp = test_fingerprint();
    let hook_types = [
        HookType::SessionStart,
        HookType::UserPromptSubmit,
        HookType::PreToolUse,
        HookType::PostToolUse,
        HookType::Stop,
        HookType::SessionEnd,
    ];

    for hook_type in hook_types {
        let memory = Memory::new(
            format!("Content for {:?}", hook_type),
            MemorySource::HookDescription {
                hook_type,
                tool_name: None,
            },
            "session".to_string(),
            fp.clone(),
            None,
        );

        println!("HookType: {:?}", hook_type);
        println!("  Display: {}", hook_type);
        println!("  Memory hook_type(): {:?}", memory.hook_type());

        assert_eq!(memory.hook_type(), Some(hook_type));
        assert!(memory.validate().is_ok());
    }

    println!("\n=== ALL HOOK TYPES PASSED ===\n");
}

// =============================================================================
// All Response Types Test
// =============================================================================

#[test]
fn manual_test_all_response_types() {
    println!("=== TEST: All Response Types ===");

    let fp = test_fingerprint();
    let response_types = [
        ResponseType::SessionSummary,
        ResponseType::StopResponse,
        ResponseType::SignificantResponse,
    ];

    for response_type in response_types {
        let memory = Memory::new(
            format!("Response for {:?}", response_type),
            MemorySource::ClaudeResponse { response_type },
            "session".to_string(),
            fp.clone(),
            None,
        );

        println!("ResponseType: {:?}", response_type);
        println!("  Display: {}", response_type);
        println!("  Memory response_type(): {:?}", memory.response_type());

        assert_eq!(memory.response_type(), Some(response_type));
        assert!(memory.validate().is_ok());
    }

    println!("\n=== ALL RESPONSE TYPES PASSED ===\n");
}

// =============================================================================
// Memory with specific ID and timestamp
// =============================================================================

#[test]
fn manual_test_memory_reconstruction() {
    println!("=== TEST: Memory Reconstruction (with_id_and_timestamp) ===");

    let fp = test_fingerprint();
    let specific_id = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").expect("parse uuid");
    let specific_time = chrono::DateTime::parse_from_rfc3339("2025-01-15T10:30:00Z")
        .expect("parse time")
        .with_timezone(&chrono::Utc);

    println!("Input ID: {}", specific_id);
    println!("Input timestamp: {:?}", specific_time);

    let memory = Memory::with_id_and_timestamp(
        specific_id,
        "Reconstructed memory".to_string(),
        MemorySource::HookDescription {
            hook_type: HookType::SessionStart,
            tool_name: None,
        },
        "session".to_string(),
        fp,
        None,
        specific_time,
    );

    println!("\nOutput:");
    println!("  memory.id: {}", memory.id);
    println!("  memory.created_at: {:?}", memory.created_at);

    assert_eq!(memory.id, specific_id, "ID should match specific value");
    assert_eq!(
        memory.created_at, specific_time,
        "Timestamp should match specific value"
    );

    println!("\n=== MEMORY RECONSTRUCTION PASSED ===\n");
}

// =============================================================================
// MemorySource Display Format Test
// =============================================================================

#[test]
fn manual_test_memory_source_display() {
    println!("=== TEST: MemorySource Display Format ===");

    let sources = [
        (
            MemorySource::HookDescription {
                hook_type: HookType::SessionStart,
                tool_name: None,
            },
            "HookDescription(SessionStart)",
        ),
        (
            MemorySource::HookDescription {
                hook_type: HookType::PostToolUse,
                tool_name: Some("Edit".to_string()),
            },
            "HookDescription(PostToolUse, tool=Edit)",
        ),
        (
            MemorySource::ClaudeResponse {
                response_type: ResponseType::SessionSummary,
            },
            "ClaudeResponse(SessionSummary)",
        ),
        (
            MemorySource::MDFileChunk {
                file_path: "README.md".to_string(),
                chunk_index: 0,
                total_chunks: 5,
            },
            "MDFileChunk(README.md, 1/5)", // 1-indexed display
        ),
    ];

    for (source, expected_pattern) in sources {
        let display = source.to_string();
        println!("Source: {:?}", source);
        println!("  Display: {}", display);
        println!("  Expected pattern: {}", expected_pattern);

        assert!(
            display.contains(&expected_pattern.split(',').next().unwrap().to_string()),
            "Display '{}' should contain expected pattern",
            display
        );
    }

    println!("\n=== MEMORY SOURCE DISPLAY PASSED ===\n");
}
