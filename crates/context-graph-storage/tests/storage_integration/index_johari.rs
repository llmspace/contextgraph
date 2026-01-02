//! Johari quadrant index tests.
//!
//! Tests the Johari Window quadrant indexing and transitions.

use super::common::{create_test_node, setup_db};
use context_graph_core::types::JohariQuadrant;

#[test]
fn test_johari_quadrant_index_consistency() {
    let (db, _tmp) = setup_db();

    let mut open_node = create_test_node();
    open_node.quadrant = JohariQuadrant::Open;
    let mut hidden_node = create_test_node();
    hidden_node.quadrant = JohariQuadrant::Hidden;

    println!("=== JOHARI INDEX TEST ===");
    println!(
        "BEFORE: open_node quadrant={:?}, hidden_node quadrant={:?}",
        open_node.quadrant, hidden_node.quadrant
    );

    db.store_node(&open_node).expect("store open");
    db.store_node(&hidden_node).expect("store hidden");

    // Query the indexes - Source of Truth
    let open_ids = db
        .get_nodes_by_quadrant(JohariQuadrant::Open, None, 0)
        .expect("query open");
    let hidden_ids = db
        .get_nodes_by_quadrant(JohariQuadrant::Hidden, None, 0)
        .expect("query hidden");

    println!(
        "VERIFY: Open has {} nodes, Hidden has {} nodes",
        open_ids.len(),
        hidden_ids.len()
    );
    assert!(
        open_ids.contains(&open_node.id),
        "open_node should be in Open index"
    );
    assert!(
        hidden_ids.contains(&hidden_node.id),
        "hidden_node should be in Hidden index"
    );

    // Transition test: move open_node to Hidden
    let mut transitioned = open_node.clone();
    transitioned.quadrant = JohariQuadrant::Hidden;
    println!(
        "TRIGGER: Transitioning node {} from Open to Hidden",
        transitioned.id
    );
    db.update_node(&transitioned).expect("update");

    // Verify indexes updated correctly
    let open_after = db
        .get_nodes_by_quadrant(JohariQuadrant::Open, None, 0)
        .expect("query");
    let hidden_after = db
        .get_nodes_by_quadrant(JohariQuadrant::Hidden, None, 0)
        .expect("query");

    println!(
        "AFTER TRANSITION: Open={}, Hidden={}",
        open_after.len(),
        hidden_after.len()
    );
    assert!(
        !open_after.contains(&open_node.id),
        "transitioned node should NOT be in Open index"
    );
    assert!(
        hidden_after.contains(&open_node.id),
        "transitioned node SHOULD be in Hidden index"
    );
    println!("RESULT: PASSED");
}

#[test]
fn test_all_four_johari_quadrants() {
    let (db, _tmp) = setup_db();

    println!("=== ALL FOUR JOHARI QUADRANTS TEST ===");

    // Create one node for each quadrant
    let mut node_open = create_test_node();
    node_open.quadrant = JohariQuadrant::Open;
    let mut node_hidden = create_test_node();
    node_hidden.quadrant = JohariQuadrant::Hidden;
    let mut node_blind = create_test_node();
    node_blind.quadrant = JohariQuadrant::Blind;
    let mut node_unknown = create_test_node();
    node_unknown.quadrant = JohariQuadrant::Unknown;

    db.store_node(&node_open).expect("store open");
    db.store_node(&node_hidden).expect("store hidden");
    db.store_node(&node_blind).expect("store blind");
    db.store_node(&node_unknown).expect("store unknown");

    // Query each quadrant
    for (quadrant, expected_id) in [
        (JohariQuadrant::Open, node_open.id),
        (JohariQuadrant::Hidden, node_hidden.id),
        (JohariQuadrant::Blind, node_blind.id),
        (JohariQuadrant::Unknown, node_unknown.id),
    ] {
        let ids = db
            .get_nodes_by_quadrant(quadrant, None, 0)
            .expect("query quadrant");
        println!("VERIFY: {:?} contains {} nodes", quadrant, ids.len());
        assert!(
            ids.contains(&expected_id),
            "{:?} index should contain the corresponding node",
            quadrant
        );
    }
    println!("RESULT: PASSED");
}
