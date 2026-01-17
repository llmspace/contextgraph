//! Johari Window Threshold Management
//!
//! Provides domain-aware thresholds for Johari quadrant classification.
//! Replaces hardcoded constants with adaptive threshold calibration (ATC).
//!
//! # Constitution Reference
//!
//! From `docs2/constitution.yaml` lines 153-158:
//! - Open: delta_s<0.5, delta_c>0.5 -> DirectRecall
//! - Blind: delta_s>0.5, delta_c<0.5 -> TriggerDream
//! - Hidden: delta_s<0.5, delta_c<0.5 -> GetNeighborhood
//! - Unknown: delta_s>0.5, delta_c>0.5 -> EpistemicAction
//!
//! # Legacy Values (MUST preserve for default_general)
//!
//! - ENTROPY_THRESHOLD = 0.50
//! - COHERENCE_THRESHOLD = 0.50
//! - BLIND_SPOT_THRESHOLD = 0.50
//!
//! # ATC Domain Thresholds
//!
//! From `atc/domain.rs`:
//! - theta_johari: [0.35, 0.65] Classification boundary
//! - theta_blind_spot: [0.35, 0.65] Blind spot detection

use crate::atc::{AdaptiveThresholdCalibration, Domain};
use crate::error::{CoreError, CoreResult};
use crate::types::JohariQuadrant;

/// Johari Window classification thresholds.
///
/// These thresholds define the boundaries between Johari quadrants:
/// - `entropy`: Threshold for delta_s - below = low, above/equal = high
/// - `coherence`: Threshold for delta_c - above = high, below/equal = low
/// - `blind_spot`: Threshold for explicit blind spot detection
///
/// # Constitution Reference
///
/// - theta_johari: [0.35, 0.65]
/// - theta_blind_spot: [0.35, 0.65]
///
/// # Invariants
///
/// All values must be within `[0.35, 0.65]` per constitution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JohariThresholds {
    /// Threshold for entropy (delta_s) classification.
    /// Below threshold = low entropy, at/above = high entropy.
    pub entropy: f32,

    /// Threshold for coherence (delta_c) classification.
    /// Above threshold = high coherence, at/below = low coherence.
    pub coherence: f32,

    /// Threshold for blind spot detection.
    pub blind_spot: f32,
}

impl JohariThresholds {
    /// Create from ATC for a specific domain.
    ///
    /// # Arguments
    ///
    /// * `atc` - Reference to the AdaptiveThresholdCalibration system
    /// * `domain` - The domain to retrieve thresholds for
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - ATC doesn't have the requested domain
    /// - Retrieved thresholds fail validation (out of `[0.35, 0.65]` range)
    pub fn from_atc(atc: &AdaptiveThresholdCalibration, domain: Domain) -> CoreResult<Self> {
        let domain_thresholds = atc.get_domain_thresholds(domain).ok_or_else(|| {
            CoreError::ConfigError(format!(
                "ATC missing domain thresholds for {:?}. \
                Ensure AdaptiveThresholdCalibration is properly initialized.",
                domain
            ))
        })?;

        let johari = Self {
            entropy: domain_thresholds.theta_johari,
            coherence: domain_thresholds.theta_johari, // Same threshold for symmetric quadrants
            blind_spot: domain_thresholds.theta_blind_spot,
        };

        if !johari.is_valid() {
            return Err(CoreError::ValidationError {
                field: "JohariThresholds".to_string(),
                message: format!(
                    "Invalid thresholds from ATC domain {:?}: entropy={}, coherence={}, blind_spot={}. \
                    Required: all values in [0.35, 0.65].",
                    domain, johari.entropy, johari.coherence, johari.blind_spot
                ),
            });
        }

        Ok(johari)
    }

    /// Create with legacy General domain defaults.
    ///
    /// These values MUST match the old hardcoded constants:
    /// - ENTROPY_THRESHOLD = 0.50
    /// - COHERENCE_THRESHOLD = 0.50
    /// - BLIND_SPOT_THRESHOLD = 0.50
    #[inline]
    pub fn default_general() -> Self {
        Self {
            entropy: 0.50,
            coherence: 0.50,
            blind_spot: 0.50,
        }
    }

    /// Validate thresholds are within constitution ranges.
    ///
    /// Per constitution: all values must be in [0.35, 0.65]
    pub fn is_valid(&self) -> bool {
        (0.35..=0.65).contains(&self.entropy)
            && (0.35..=0.65).contains(&self.coherence)
            && (0.35..=0.65).contains(&self.blind_spot)
    }

    /// Check if entropy value indicates "low entropy" (below threshold).
    #[inline]
    pub fn is_low_entropy(&self, delta_s: f32) -> bool {
        delta_s < self.entropy
    }

    /// Check if coherence value indicates "high coherence" (above threshold).
    #[inline]
    pub fn is_high_coherence(&self, delta_c: f32) -> bool {
        delta_c > self.coherence
    }

    /// Classify into Johari quadrant using these thresholds.
    ///
    /// # Classification Rules (per constitution)
    ///
    /// | Entropy | Coherence | Quadrant | Action |
    /// |---------|-----------|----------|--------|
    /// | Low (<) | High (>)  | Open     | DirectRecall |
    /// | Low (<) | Low (<=)  | Hidden   | GetNeighborhood |
    /// | High (>=)| Low (<=) | Blind    | TriggerDream |
    /// | High (>=)| High (>) | Unknown  | EpistemicAction |
    #[inline]
    pub fn classify(&self, delta_s: f32, delta_c: f32) -> JohariQuadrant {
        let low_entropy = self.is_low_entropy(delta_s);
        let high_coherence = self.is_high_coherence(delta_c);

        match (low_entropy, high_coherence) {
            (true, true) => JohariQuadrant::Open,     // Low S, High C
            (true, false) => JohariQuadrant::Hidden,  // Low S, Low C
            (false, false) => JohariQuadrant::Blind,  // High S, Low C
            (false, true) => JohariQuadrant::Unknown, // High S, High C
        }
    }

    /// Check if values indicate a blind spot (Blind quadrant).
    #[inline]
    pub fn is_blind_spot(&self, delta_s: f32, delta_c: f32) -> bool {
        delta_s >= self.blind_spot && delta_c <= self.coherence
    }
}

impl Default for JohariThresholds {
    fn default() -> Self {
        Self::default_general()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================
    // LEGACY VALUE COMPATIBILITY TESTS
    // ========================================================

    #[test]
    fn test_default_matches_legacy_constants() {
        let t = JohariThresholds::default_general();

        // These MUST match the old hardcoded values EXACTLY
        assert_eq!(
            t.entropy, 0.50,
            "entropy must match ENTROPY_THRESHOLD (0.50), got {}",
            t.entropy
        );
        assert_eq!(
            t.coherence, 0.50,
            "coherence must match COHERENCE_THRESHOLD (0.50), got {}",
            t.coherence
        );
        assert_eq!(
            t.blind_spot, 0.50,
            "blind_spot must match BLIND_SPOT_THRESHOLD (0.50), got {}",
            t.blind_spot
        );

        println!("[VERIFIED] default_general() matches legacy constants:");
        println!("  entropy: {} == 0.50 (ENTROPY_THRESHOLD)", t.entropy);
        println!("  coherence: {} == 0.50 (COHERENCE_THRESHOLD)", t.coherence);
        println!(
            "  blind_spot: {} == 0.50 (BLIND_SPOT_THRESHOLD)",
            t.blind_spot
        );
    }

    #[test]
    fn test_default_is_valid() {
        let t = JohariThresholds::default_general();
        assert!(
            t.is_valid(),
            "default_general() must produce valid thresholds"
        );
        println!("[VERIFIED] default_general() produces valid thresholds");
    }

    #[test]
    fn test_default_trait_matches_default_general() {
        let default_trait = JohariThresholds::default();
        let default_general = JohariThresholds::default_general();

        assert_eq!(
            default_trait, default_general,
            "Default trait must match default_general()"
        );
        println!("[VERIFIED] Default trait == default_general()");
    }

    // ========================================================
    // ATC INTEGRATION TESTS
    // ========================================================

    #[test]
    fn test_from_atc_all_domains() {
        let atc = AdaptiveThresholdCalibration::new();

        for domain in [
            Domain::Code,
            Domain::Medical,
            Domain::Legal,
            Domain::Creative,
            Domain::Research,
            Domain::General,
        ] {
            let result = JohariThresholds::from_atc(&atc, domain);
            assert!(
                result.is_ok(),
                "Domain {:?} should produce valid thresholds, got error: {:?}",
                domain,
                result.err()
            );

            let t = result.unwrap();
            assert!(
                t.is_valid(),
                "Domain {:?} thresholds should be valid: entropy={}, coherence={}, blind_spot={}",
                domain,
                t.entropy,
                t.coherence,
                t.blind_spot
            );
        }
        println!("[VERIFIED] All 6 domains produce valid JohariThresholds from ATC");
    }

    #[test]
    fn test_atc_values_match_domain_thresholds() {
        let atc = AdaptiveThresholdCalibration::new();

        for domain in [
            Domain::Code,
            Domain::Medical,
            Domain::Legal,
            Domain::Creative,
            Domain::Research,
            Domain::General,
        ] {
            let johari = JohariThresholds::from_atc(&atc, domain).unwrap();
            let domain_t = atc.get_domain_thresholds(domain).unwrap();

            assert_eq!(
                johari.entropy, domain_t.theta_johari,
                "entropy should equal theta_johari for {:?}",
                domain
            );
            assert_eq!(
                johari.blind_spot, domain_t.theta_blind_spot,
                "blind_spot should equal theta_blind_spot for {:?}",
                domain
            );
        }
        println!("[VERIFIED] JohariThresholds values correctly mapped from DomainThresholds");
    }

    // ========================================================
    // CLASSIFICATION TESTS
    // ========================================================

    #[test]
    fn test_classify_all_quadrants() {
        let t = JohariThresholds::default_general();

        // Open: Low S, High C
        assert_eq!(
            t.classify(0.3, 0.7),
            JohariQuadrant::Open,
            "Low S (0.3), High C (0.7) -> Open"
        );

        // Hidden: Low S, Low C
        assert_eq!(
            t.classify(0.3, 0.3),
            JohariQuadrant::Hidden,
            "Low S (0.3), Low C (0.3) -> Hidden"
        );

        // Blind: High S, Low C
        assert_eq!(
            t.classify(0.7, 0.3),
            JohariQuadrant::Blind,
            "High S (0.7), Low C (0.3) -> Blind"
        );

        // Unknown: High S, High C
        assert_eq!(
            t.classify(0.7, 0.7),
            JohariQuadrant::Unknown,
            "High S (0.7), High C (0.7) -> Unknown"
        );

        println!("[VERIFIED] All quadrant classifications correct");
    }

    #[test]
    fn test_classify_exact_boundary() {
        let t = JohariThresholds::default_general();

        // At (0.5, 0.5): entropy >= 0.5 is HIGH, coherence <= 0.5 is LOW -> Blind
        assert_eq!(
            t.classify(0.5, 0.5),
            JohariQuadrant::Blind,
            "At boundary (0.5, 0.5) should be Blind"
        );

        println!("[VERIFIED] Exact boundary (0.5, 0.5) -> Blind");
    }

    #[test]
    fn test_classify_just_below_boundary() {
        let t = JohariThresholds::default_general();

        // At (0.49, 0.51): entropy < 0.5 is LOW, coherence > 0.5 is HIGH -> Open
        assert_eq!(
            t.classify(0.49, 0.51),
            JohariQuadrant::Open,
            "Just below boundary (0.49, 0.51) should be Open"
        );

        println!("[VERIFIED] Just below boundary (0.49, 0.51) -> Open");
    }

    #[test]
    fn test_classify_extreme_values() {
        let t = JohariThresholds::default_general();

        // Extreme corners
        assert_eq!(t.classify(0.0, 1.0), JohariQuadrant::Open);
        assert_eq!(t.classify(0.0, 0.0), JohariQuadrant::Hidden);
        assert_eq!(t.classify(1.0, 0.0), JohariQuadrant::Blind);
        assert_eq!(t.classify(1.0, 1.0), JohariQuadrant::Unknown);

        println!("[VERIFIED] Extreme value classifications correct");
    }

    #[test]
    fn test_threshold_affects_classification() {
        let low = JohariThresholds {
            entropy: 0.4,
            coherence: 0.4,
            blind_spot: 0.4,
        };
        let high = JohariThresholds {
            entropy: 0.6,
            coherence: 0.6,
            blind_spot: 0.6,
        };

        // At (0.5, 0.5):
        // Low threshold (0.4): 0.5 >= 0.4 is HIGH, 0.5 > 0.4 is HIGH -> Unknown
        assert_eq!(
            low.classify(0.5, 0.5),
            JohariQuadrant::Unknown,
            "With threshold=0.4, (0.5, 0.5) should be Unknown"
        );

        // High threshold (0.6): 0.5 < 0.6 is LOW, 0.5 <= 0.6 is LOW -> Hidden
        assert_eq!(
            high.classify(0.5, 0.5),
            JohariQuadrant::Hidden,
            "With threshold=0.6, (0.5, 0.5) should be Hidden"
        );

        println!("[VERIFIED] Threshold value affects classification outcome");
    }

    // ========================================================
    // HELPER METHOD TESTS
    // ========================================================

    #[test]
    fn test_is_low_entropy() {
        let t = JohariThresholds::default_general();

        assert!(t.is_low_entropy(0.49), "0.49 < 0.50 should be low entropy");
        assert!(
            !t.is_low_entropy(0.50),
            "0.50 >= 0.50 should NOT be low entropy"
        );
        assert!(
            !t.is_low_entropy(0.51),
            "0.51 >= 0.50 should NOT be low entropy"
        );
    }

    #[test]
    fn test_is_high_coherence() {
        let t = JohariThresholds::default_general();

        assert!(
            t.is_high_coherence(0.51),
            "0.51 > 0.50 should be high coherence"
        );
        assert!(
            !t.is_high_coherence(0.50),
            "0.50 <= 0.50 should NOT be high coherence"
        );
        assert!(
            !t.is_high_coherence(0.49),
            "0.49 <= 0.50 should NOT be high coherence"
        );
    }

    #[test]
    fn test_is_blind_spot() {
        let t = JohariThresholds::default_general();

        // Blind spot: high entropy (>= threshold), low coherence (<= threshold)
        assert!(
            t.is_blind_spot(0.7, 0.3),
            "High entropy, low coherence is blind spot"
        );
        assert!(
            t.is_blind_spot(0.5, 0.5),
            "At boundary should be blind spot"
        );
        assert!(
            !t.is_blind_spot(0.3, 0.7),
            "Low entropy, high coherence is NOT blind spot"
        );
        assert!(
            !t.is_blind_spot(0.3, 0.3),
            "Low entropy, low coherence is NOT blind spot (Hidden)"
        );
    }

    // ========================================================
    // VALIDATION TESTS
    // ========================================================

    #[test]
    fn test_invalid_entropy_below_min() {
        let t = JohariThresholds {
            entropy: 0.30,
            coherence: 0.50,
            blind_spot: 0.50,
        };
        assert!(!t.is_valid(), "entropy=0.30 below min 0.35 should fail");
    }

    #[test]
    fn test_invalid_entropy_above_max() {
        let t = JohariThresholds {
            entropy: 0.70,
            coherence: 0.50,
            blind_spot: 0.50,
        };
        assert!(!t.is_valid(), "entropy=0.70 above max 0.65 should fail");
    }

    #[test]
    fn test_invalid_coherence_below_min() {
        let t = JohariThresholds {
            entropy: 0.50,
            coherence: 0.30,
            blind_spot: 0.50,
        };
        assert!(!t.is_valid(), "coherence=0.30 below min 0.35 should fail");
    }

    #[test]
    fn test_invalid_coherence_above_max() {
        let t = JohariThresholds {
            entropy: 0.50,
            coherence: 0.70,
            blind_spot: 0.50,
        };
        assert!(!t.is_valid(), "coherence=0.70 above max 0.65 should fail");
    }

    #[test]
    fn test_invalid_blind_spot_below_min() {
        let t = JohariThresholds {
            entropy: 0.50,
            coherence: 0.50,
            blind_spot: 0.30,
        };
        assert!(!t.is_valid(), "blind_spot=0.30 below min 0.35 should fail");
    }

    #[test]
    fn test_invalid_blind_spot_above_max() {
        let t = JohariThresholds {
            entropy: 0.50,
            coherence: 0.50,
            blind_spot: 0.70,
        };
        assert!(!t.is_valid(), "blind_spot=0.70 above max 0.65 should fail");
    }

    #[test]
    fn test_valid_at_boundaries() {
        let t_min = JohariThresholds {
            entropy: 0.35,
            coherence: 0.35,
            blind_spot: 0.35,
        };
        assert!(t_min.is_valid(), "All values at min 0.35 should be valid");

        let t_max = JohariThresholds {
            entropy: 0.65,
            coherence: 0.65,
            blind_spot: 0.65,
        };
        assert!(t_max.is_valid(), "All values at max 0.65 should be valid");
    }

    // ========================================================
    // FSV (FULL STATE VERIFICATION) TESTS
    // ========================================================

    #[test]
    fn test_fsv_johari_threshold_verification() {
        println!("\n=== FSV: Johari Threshold Verification ===\n");

        // 1. Verify default_general matches legacy
        let default = JohariThresholds::default_general();
        println!(
            "Default General Thresholds: entropy={}, coherence={}, blind_spot={}",
            default.entropy, default.coherence, default.blind_spot
        );
        assert_eq!(default.entropy, 0.50);
        assert_eq!(default.coherence, 0.50);
        assert_eq!(default.blind_spot, 0.50);
        println!("  [VERIFIED] Default matches legacy constants\n");

        // 2. Verify ATC retrieval for all domains
        let atc = AdaptiveThresholdCalibration::new();
        println!("ATC Domain Thresholds:");
        for domain in [
            Domain::Medical,
            Domain::Code,
            Domain::Legal,
            Domain::General,
            Domain::Research,
            Domain::Creative,
        ] {
            let t = JohariThresholds::from_atc(&atc, domain).unwrap();
            println!(
                "  {:?} (strictness={:.1}): entropy={:.3}, coherence={:.3}, blind_spot={:.3}",
                domain,
                domain.strictness(),
                t.entropy,
                t.coherence,
                t.blind_spot
            );
            assert!(t.is_valid());
        }
        println!("  [VERIFIED] All domains produce valid thresholds\n");

        // 3. Classification tests with state printout
        println!("Classification Tests:");
        let t = JohariThresholds::default_general();

        let cases: [(f32, f32, JohariQuadrant, &str); 8] = [
            (0.30, 0.70, JohariQuadrant::Open, "Low S, High C"),
            (0.30, 0.30, JohariQuadrant::Hidden, "Low S, Low C"),
            (0.70, 0.30, JohariQuadrant::Blind, "High S, Low C"),
            (0.70, 0.70, JohariQuadrant::Unknown, "High S, High C"),
            (0.50, 0.50, JohariQuadrant::Blind, "S>=0.5=HIGH, C<=0.5=LOW"),
            (0.49, 0.51, JohariQuadrant::Open, "S<0.5=LOW, C>0.5=HIGH"),
            (0.00, 1.00, JohariQuadrant::Open, "Min S, Max C"),
            (1.00, 0.00, JohariQuadrant::Blind, "Max S, Min C"),
        ];

        for (s, c, expected, reason) in cases {
            let actual = t.classify(s, c);
            println!(
                "  classify({:.2}, {:.2}) = {:?} (expected {:?}) - {}",
                s, c, actual, expected, reason
            );
            assert_eq!(
                actual, expected,
                "Failed: classify({}, {}) should be {:?}, got {:?}",
                s, c, expected, actual
            );
        }
        println!("  [VERIFIED] All classifications correct\n");

        // 4. Edge case: Invalid thresholds rejected
        println!("Edge Case Tests:");
        let invalid_low = JohariThresholds {
            entropy: 0.30,
            coherence: 0.50,
            blind_spot: 0.50,
        };
        assert!(
            !invalid_low.is_valid(),
            "entropy=0.30 should fail validation"
        );
        println!(
            "  entropy=0.30: is_valid()={} (expected false)",
            invalid_low.is_valid()
        );

        let invalid_high = JohariThresholds {
            entropy: 0.70,
            coherence: 0.50,
            blind_spot: 0.50,
        };
        assert!(
            !invalid_high.is_valid(),
            "entropy=0.70 should fail validation"
        );
        println!(
            "  entropy=0.70: is_valid()={} (expected false)",
            invalid_high.is_valid()
        );
        println!("  [VERIFIED] Invalid thresholds correctly rejected\n");

        println!("=== FSV COMPLETE: All verifications passed ===\n");
    }

    #[test]
    fn test_print_all_domain_thresholds() {
        println!("\n=== ATC Domain Johari Thresholds ===\n");

        let atc = AdaptiveThresholdCalibration::new();

        for domain in [
            Domain::Medical,
            Domain::Code,
            Domain::Legal,
            Domain::General,
            Domain::Research,
            Domain::Creative,
        ] {
            let t = JohariThresholds::from_atc(&atc, domain).unwrap();
            println!(
                "{:?} (strictness={:.1}): entropy={:.3}, coherence={:.3}, blind_spot={:.3}",
                domain,
                domain.strictness(),
                t.entropy,
                t.coherence,
                t.blind_spot
            );
        }

        println!("\nLegacy defaults: entropy=0.50, coherence=0.50, blind_spot=0.50");
    }

    // ========================================================
    // CONSISTENCY WITH EXISTING CODE TESTS
    // ========================================================

    #[test]
    fn test_consistency_with_johari_fingerprint_classify() {
        use crate::types::fingerprint::JohariFingerprint;

        let t = JohariThresholds::default_general();

        // Test that JohariThresholds.classify produces the same results as
        // JohariFingerprint::classify_quadrant for all test cases
        let test_cases = [
            (0.3, 0.7),
            (0.3, 0.3),
            (0.7, 0.3),
            (0.7, 0.7),
            (0.5, 0.5),
            (0.49, 0.51),
            (0.0, 1.0),
            (1.0, 0.0),
        ];

        for (s, c) in test_cases {
            let new_result = t.classify(s, c);
            let old_result = JohariFingerprint::classify_quadrant(s, c);

            assert_eq!(
                new_result, old_result,
                "JohariThresholds.classify({}, {}) = {:?} but JohariFingerprint::classify_quadrant = {:?}",
                s, c, new_result, old_result
            );
        }

        println!("[VERIFIED] JohariThresholds.classify produces identical results to JohariFingerprint::classify_quadrant");
    }
}
