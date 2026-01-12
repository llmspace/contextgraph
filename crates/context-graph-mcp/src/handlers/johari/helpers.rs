//! Johari helper functions.
//!
//! Parsing and conversion utilities for Johari quadrants and triggers.

use context_graph_core::types::{JohariQuadrant, TransitionTrigger};

/// Parse quadrant string to enum.
pub fn parse_quadrant(s: &str) -> Option<JohariQuadrant> {
    match s.to_lowercase().as_str() {
        "open" => Some(JohariQuadrant::Open),
        "hidden" => Some(JohariQuadrant::Hidden),
        "blind" => Some(JohariQuadrant::Blind),
        "unknown" => Some(JohariQuadrant::Unknown),
        _ => None,
    }
}

/// Parse trigger string to enum.
pub fn parse_trigger(s: &str) -> Option<TransitionTrigger> {
    match s.to_lowercase().replace('-', "_").as_str() {
        "explicit_share" | "explicitshare" => Some(TransitionTrigger::ExplicitShare),
        "self_recognition" | "selfrecognition" => Some(TransitionTrigger::SelfRecognition),
        "pattern_discovery" | "patterndiscovery" => Some(TransitionTrigger::PatternDiscovery),
        "privatize" => Some(TransitionTrigger::Privatize),
        "external_observation" | "externalobservation" => {
            Some(TransitionTrigger::ExternalObservation)
        }
        "dream_consolidation" | "dreamconsolidation" => Some(TransitionTrigger::DreamConsolidation),
        _ => None,
    }
}

/// Convert quadrant enum to string.
pub fn quadrant_to_string(q: JohariQuadrant) -> String {
    match q {
        JohariQuadrant::Open => "open".to_string(),
        JohariQuadrant::Hidden => "hidden".to_string(),
        JohariQuadrant::Blind => "blind".to_string(),
        JohariQuadrant::Unknown => "unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_quadrant() {
        assert_eq!(parse_quadrant("open"), Some(JohariQuadrant::Open));
        assert_eq!(parse_quadrant("HIDDEN"), Some(JohariQuadrant::Hidden));
        assert_eq!(parse_quadrant("Blind"), Some(JohariQuadrant::Blind));
        assert_eq!(parse_quadrant("unknown"), Some(JohariQuadrant::Unknown));
        assert_eq!(parse_quadrant("invalid"), None);

        println!("[VERIFIED] test_parse_quadrant: All quadrant parsing works correctly");
    }

    #[test]
    fn test_parse_trigger() {
        assert_eq!(
            parse_trigger("explicit_share"),
            Some(TransitionTrigger::ExplicitShare)
        );
        assert_eq!(
            parse_trigger("dream_consolidation"),
            Some(TransitionTrigger::DreamConsolidation)
        );
        assert_eq!(
            parse_trigger("external_observation"),
            Some(TransitionTrigger::ExternalObservation)
        );
        assert_eq!(
            parse_trigger("privatize"),
            Some(TransitionTrigger::Privatize)
        );
        assert_eq!(parse_trigger("invalid"), None);

        println!("[VERIFIED] test_parse_trigger: All trigger parsing works correctly");
    }

    #[test]
    fn test_quadrant_to_string() {
        assert_eq!(quadrant_to_string(JohariQuadrant::Open), "open");
        assert_eq!(quadrant_to_string(JohariQuadrant::Hidden), "hidden");
        assert_eq!(quadrant_to_string(JohariQuadrant::Blind), "blind");
        assert_eq!(quadrant_to_string(JohariQuadrant::Unknown), "unknown");

        println!("[VERIFIED] test_quadrant_to_string: All conversions work correctly");
    }
}
