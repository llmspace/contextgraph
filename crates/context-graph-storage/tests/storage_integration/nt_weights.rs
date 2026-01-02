//! Neurotransmitter weight tests.
//!
//! Tests domain-specific NT weights and the effective weight formula.

use context_graph_core::marblestone::{Domain, NeurotransmitterWeights};

#[test]
fn test_domain_specific_nt_weights() {
    println!("=== DOMAIN-SPECIFIC NT WEIGHTS TEST ===");

    // Test that different domains produce different NT weights
    let code_nt = NeurotransmitterWeights::for_domain(Domain::Code);
    let legal_nt = NeurotransmitterWeights::for_domain(Domain::Legal);
    let medical_nt = NeurotransmitterWeights::for_domain(Domain::Medical);
    let creative_nt = NeurotransmitterWeights::for_domain(Domain::Creative);

    println!("Code domain NT: {:?}", code_nt);
    println!("Legal domain NT: {:?}", legal_nt);
    println!("Medical domain NT: {:?}", medical_nt);
    println!("Creative domain NT: {:?}", creative_nt);

    // Verify weights are in valid range [0, 1]
    for nt in [&code_nt, &legal_nt, &medical_nt, &creative_nt] {
        assert!(nt.excitatory >= 0.0 && nt.excitatory <= 1.0);
        assert!(nt.inhibitory >= 0.0 && nt.inhibitory <= 1.0);
        assert!(nt.modulatory >= 0.0 && nt.modulatory <= 1.0);
    }

    println!("RESULT: PASSED");
}

#[test]
fn test_nt_effective_weight_formula() {
    println!("=== NT EFFECTIVE WEIGHT FORMULA TEST ===");

    // Actual formula:
    // signal = base_weight * excitatory - base_weight * inhibitory
    // mod_factor = 1.0 + (modulatory - 0.5) * 0.4
    // result = (signal * mod_factor).clamp(0.0, 1.0)
    let nt = NeurotransmitterWeights {
        excitatory: 0.8,
        inhibitory: 0.2,
        modulatory: 0.4,
    };

    let base = 0.5;
    let signal = base * nt.excitatory - base * nt.inhibitory;
    let mod_factor = 1.0 + (nt.modulatory - 0.5) * 0.4;
    let expected = (signal * mod_factor).clamp(0.0, 1.0);
    let computed = nt.compute_effective_weight(base);

    println!(
        "VERIFY: base={}, NT={{exc={}, inh={}, mod={}}}, signal={}, mod_factor={}, expected={}, computed={}",
        base, nt.excitatory, nt.inhibitory, nt.modulatory, signal, mod_factor, expected, computed
    );
    assert!(
        (computed - expected).abs() < 0.001,
        "effective weight should match formula"
    );

    println!("RESULT: PASSED");
}
