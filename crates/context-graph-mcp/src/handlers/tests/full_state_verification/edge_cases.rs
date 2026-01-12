//! Edge case verification tests.
//!
//! Tests verify handling of edge cases and error conditions:
//! - Empty content
//! - Invalid UUID formats
//! - Non-existent fingerprint IDs

use serde_json::json;

use crate::protocol::JsonRpcId;

use super::helpers::{create_handlers_with_store_access, exists_in_store, make_request};

// =============================================================================
// EDGE CASE VERIFICATION WITH BEFORE/AFTER STATE
// =============================================================================

/// EDGE CASE 1: Empty content string
#[tokio::test]
async fn verify_edge_case_empty_content() {
    println!("\n================================================================================");
    println!("EDGE CASE VERIFICATION: Empty Content String");
    println!("================================================================================");

    let (handlers, store, _provider) = create_handlers_with_store_access();

    // === BEFORE STATE ===
    let count_before = store.count().await.unwrap();
    println!("\n[BEFORE] Store count: {}", count_before);

    // === ATTEMPT OPERATION WITH EMPTY CONTENT ===
    let params = json!({
        "content": "",
        "importance": 0.5
    });
    let request = make_request("memory/store", Some(JsonRpcId::Number(1)), Some(params));
    let response = handlers.dispatch(request).await;

    println!("\n[OPERATION] Attempted store with empty content");
    println!("  - Error returned: {}", response.error.is_some());
    if let Some(error) = &response.error {
        println!("  - Error code: {}", error.code);
        println!("  - Error message: {}", error.message);
    }

    // === AFTER STATE - VERIFY NO CHANGE ===
    let count_after = store.count().await.unwrap();
    println!("\n[AFTER] Store count: {}", count_after);

    assert!(response.error.is_some(), "Empty content must return error");
    assert_eq!(
        response.error.unwrap().code,
        -32602,
        "Must be INVALID_PARAMS"
    );
    assert_eq!(
        count_before, count_after,
        "Store count must not change on error"
    );

    println!("\n[VERIFICATION PASSED] Empty content rejected, store unchanged");
    println!("================================================================================\n");
}

/// EDGE CASE 2: Invalid UUID format
#[tokio::test]
async fn verify_edge_case_invalid_uuid() {
    println!("\n================================================================================");
    println!("EDGE CASE VERIFICATION: Invalid UUID Format");
    println!("================================================================================");

    let (handlers, store, _provider) = create_handlers_with_store_access();

    // Store one valid fingerprint first
    let store_params = json!({ "content": "Valid content", "importance": 0.5 });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    handlers.dispatch(store_request).await;

    // === BEFORE STATE ===
    let count_before = store.count().await.unwrap();
    println!("\n[BEFORE] Store count: {}", count_before);

    // === ATTEMPT RETRIEVE WITH INVALID UUID ===
    let invalid_uuids = [
        "not-a-uuid",
        "12345",
        "00000000-0000-0000-0000",              // truncated
        "zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz", // invalid chars
    ];

    for invalid_uuid in &invalid_uuids {
        let params = json!({ "fingerprintId": invalid_uuid });
        let request = make_request("memory/retrieve", Some(JsonRpcId::Number(10)), Some(params));
        let response = handlers.dispatch(request).await;

        println!(
            "\n[OPERATION] Retrieve with invalid UUID: '{}'",
            invalid_uuid
        );
        println!("  - Error returned: {}", response.error.is_some());
        if let Some(error) = &response.error {
            println!("  - Error code: {}", error.code);
        }

        assert!(response.error.is_some(), "Invalid UUID must return error");
        assert_eq!(
            response.error.unwrap().code,
            -32602,
            "Must be INVALID_PARAMS"
        );
    }

    // === AFTER STATE - VERIFY NO CHANGE ===
    let count_after = store.count().await.unwrap();
    println!("\n[AFTER] Store count: {}", count_after);
    assert_eq!(count_before, count_after, "Store must be unchanged");

    println!("\n[VERIFICATION PASSED] Invalid UUIDs rejected, store unchanged");
    println!("================================================================================\n");
}

/// EDGE CASE 3: Non-existent fingerprint ID
#[tokio::test]
async fn verify_edge_case_nonexistent_id() {
    println!("\n================================================================================");
    println!("EDGE CASE VERIFICATION: Non-existent Fingerprint ID");
    println!("================================================================================");

    let (handlers, store, _provider) = create_handlers_with_store_access();

    // Use a valid but non-existent UUID
    let nonexistent_id = "00000000-0000-0000-0000-000000000000";

    // === VERIFY NOT IN SOURCE OF TRUTH ===
    let nonexistent_uuid = uuid::Uuid::parse_str(nonexistent_id).unwrap();
    let exists = exists_in_store(&store, nonexistent_uuid).await;
    println!(
        "\n[SOURCE OF TRUTH] ID {} exists: {}",
        nonexistent_id, exists
    );
    assert!(!exists, "Non-existent ID must not exist in store");

    // === ATTEMPT RETRIEVE ===
    let retrieve_params = json!({ "fingerprintId": nonexistent_id });
    let retrieve_request = make_request(
        "memory/retrieve",
        Some(JsonRpcId::Number(1)),
        Some(retrieve_params),
    );
    let retrieve_response = handlers.dispatch(retrieve_request).await;

    println!("\n[OPERATION] Retrieve non-existent ID");
    println!("  - Error returned: {}", retrieve_response.error.is_some());
    if let Some(error) = &retrieve_response.error {
        println!("  - Error code: {}", error.code);
        println!("  - Error message: {}", error.message);
    }

    assert!(
        retrieve_response.error.is_some(),
        "Non-existent ID must return error"
    );
    assert_eq!(
        retrieve_response.error.unwrap().code,
        -32010,
        "Must be FINGERPRINT_NOT_FOUND (-32010)"
    );

    // === ATTEMPT DELETE ===
    let delete_params = json!({ "fingerprintId": nonexistent_id, "soft": false });
    let delete_request = make_request(
        "memory/delete",
        Some(JsonRpcId::Number(2)),
        Some(delete_params),
    );
    let delete_response = handlers.dispatch(delete_request).await;

    println!("\n[OPERATION] Delete non-existent ID");
    // Delete of non-existent should succeed with deleted=false or return error depending on implementation
    if let Some(result) = &delete_response.result {
        let deleted = result
            .get("deleted")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        println!("  - Result deleted: {}", deleted);
    }
    if let Some(error) = &delete_response.error {
        println!("  - Error code: {}", error.code);
    }

    println!("\n[VERIFICATION PASSED] Non-existent ID handled correctly");
    println!("================================================================================\n");
}
