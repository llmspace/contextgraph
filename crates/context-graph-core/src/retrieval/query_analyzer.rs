//! Query Type Analyzer for intelligent embedder selection.
//!
//! # Overview
//!
//! This module provides `QueryTypeAnalyzer` which detects query types and
//! extracts relevant features to enable intelligent embedder prioritization.
//!
//! # Query Types
//!
//! | Type | Description | Primary Embedders |
//! |------|-------------|-------------------|
//! | General | Default semantic search | E1 |
//! | Causal | "Why", "what caused", effect queries | E1 + E5 |
//! | Code | Programming queries | E1 + E7 |
//! | Entity | Named entity queries | E1 + E11 |
//! | Intent | Goal/purpose queries | E1 + E10 |
//! | Keyword | Exact term/jargon queries | E1 + E6 + E13 |
//! | Graph | Relationship/connection queries | E1 + E8 |
//!
//! # Philosophy
//!
//! E1 (semantic) is ALWAYS the foundation. Other embedders ENHANCE E1 by
//! finding things E1 misses. The analyzer determines which enhancers are
//! most likely to provide value for a given query.
//!
//! # FAIL FAST Policy
//!
//! No silent failures. All errors are explicit with detailed context.

use std::collections::HashSet;

use tracing::debug;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Minimum query length for analysis.
const MIN_QUERY_LEN: usize = 2;

/// Common programming keywords for code detection.
const CODE_KEYWORDS: &[&str] = &[
    "function", "class", "struct", "impl", "def", "fn", "async", "await",
    "const", "let", "var", "mut", "pub", "private", "public", "protected",
    "interface", "trait", "enum", "type", "import", "export", "module",
    "package", "crate", "use", "from", "require", "include", "return",
    "if", "else", "match", "switch", "case", "for", "while", "loop",
    "break", "continue", "try", "catch", "finally", "throw", "error",
    "null", "none", "nil", "undefined", "true", "false", "self", "this",
    "api", "endpoint", "request", "response", "http", "rest", "graphql",
    "database", "query", "sql", "insert", "update", "delete", "select",
    "test", "mock", "fixture", "assert", "expect", "should",
];

/// Code symbols that indicate programming context.
const CODE_SYMBOLS: &[&str] = &[
    "()", "[]", "{}", "->", "=>", "::", ".", ",", ";",
    "==", "!=", ">=", "<=", "&&", "||", "++", "--",
    "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
];

/// Programming language names/extensions.
const LANGUAGE_NAMES: &[&str] = &[
    "rust", "python", "javascript", "typescript", "java", "go", "golang",
    "c++", "cpp", "csharp", "c#", "ruby", "php", "swift", "kotlin",
    "scala", "haskell", "elixir", "erlang", "clojure", "lua", "perl",
    "r", "matlab", "julia", "sql", "bash", "shell", "powershell",
    "html", "css", "scss", "less", "json", "yaml", "toml", "xml",
];

/// Causal query patterns.
const CAUSAL_PATTERNS: &[&str] = &[
    "why", "because", "cause", "caused", "reason", "result", "effect",
    "lead to", "led to", "leads to", "due to", "owing to", "as a result",
    "consequently", "therefore", "thus", "hence", "so that", "in order to",
    "what happened", "what caused", "what led to", "what results",
    "how did", "how does", "how come", "explain why",
];

/// Intent/goal query patterns.
const INTENT_PATTERNS: &[&str] = &[
    "goal", "purpose", "intent", "intention", "objective", "aim", "target",
    "mission", "accomplish", "achieve", "trying to", "want to", "need to",
    "plan to", "going to", "should", "must", "have to", "ought to",
    "what was the goal", "what were we doing", "what is the intent",
    "why are we", "why did we", "what for", "for what purpose",
];

/// Graph/relationship query patterns.
const GRAPH_PATTERNS: &[&str] = &[
    "connect", "connection", "link", "linked", "relate", "related",
    "relationship", "dependency", "depends", "import", "imports",
    "use", "uses", "call", "calls", "reference", "references",
    "what uses", "what calls", "what imports", "what depends",
    "connected to", "linked to", "related to", "depends on",
];

/// Keyword-focused query patterns (exact match needed).
const KEYWORD_PATTERNS: &[&str] = &[
    "exactly", "exact", "specific", "specifically", "literal", "literally",
    "term", "keyword", "phrase", "named", "called", "titled",
    "\"", "'", // Quoted terms
];

// ============================================================================
// TYPES
// ============================================================================

/// Query type detected by the analyzer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryType {
    /// Default semantic search - use E1 foundation.
    General,
    /// Causal queries ("why", "what caused") - E1 + E5.
    Causal,
    /// Code/programming queries - E1 + E7.
    Code,
    /// Named entity queries - E1 + E11.
    Entity,
    /// Intent/goal queries - E1 + E10.
    Intent,
    /// Keyword/exact match queries - E1 + E6 + E13.
    Keyword,
    /// Graph/relationship queries - E1 + E8.
    Graph,
}

impl QueryType {
    /// Get the primary enhancer embedders for this query type.
    ///
    /// E1 is always included as the foundation.
    ///
    /// # Returns
    /// Vec of embedder indices to use (0=E1, 4=E5, etc.)
    pub fn recommended_embedders(&self) -> Vec<usize> {
        match self {
            QueryType::General => vec![0], // E1 only
            QueryType::Causal => vec![0, 4], // E1 + E5
            QueryType::Code => vec![0, 6], // E1 + E7
            QueryType::Entity => vec![0, 10], // E1 + E11
            QueryType::Intent => vec![0, 9], // E1 + E10
            QueryType::Keyword => vec![0, 5, 12], // E1 + E6 + E13
            QueryType::Graph => vec![0, 7], // E1 + E8
        }
    }

    /// Get the name of this query type.
    pub fn name(&self) -> &'static str {
        match self {
            QueryType::General => "General",
            QueryType::Causal => "Causal",
            QueryType::Code => "Code",
            QueryType::Entity => "Entity",
            QueryType::Intent => "Intent",
            QueryType::Keyword => "Keyword",
            QueryType::Graph => "Graph",
        }
    }
}

/// Causal direction detected in the query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalDirection {
    /// Query is seeking causes ("why did X happen?").
    SeekingCause,
    /// Query is seeking effects ("what happens if X?").
    SeekingEffect,
    /// Direction unclear or not applicable.
    Unknown,
}

/// Programming language detected in the query.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectedLanguage {
    /// Language name (lowercase).
    pub name: String,
    /// Confidence (0.0 - 1.0).
    pub confidence: f32,
}

/// Result of query analysis.
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    /// Original query text.
    pub query: String,

    /// Primary detected query type.
    pub query_type: QueryType,

    /// All detected query types (may be multiple).
    pub detected_types: HashSet<QueryType>,

    /// Confidence in the primary type (0.0 - 1.0).
    pub confidence: f32,

    /// Causal direction (if causal query).
    pub causal_direction: CausalDirection,

    /// Detected programming language (if code query).
    pub detected_language: Option<DetectedLanguage>,

    /// Extracted keywords/terms.
    pub keywords: Vec<String>,

    /// Extracted potential entities (capitalized terms).
    pub potential_entities: Vec<String>,

    /// Whether the query appears to be a question.
    pub is_question: bool,

    /// Recommended embedders based on analysis.
    pub recommended_embedders: Vec<usize>,
}

impl QueryAnalysis {
    /// Create a default analysis for a query.
    fn new(query: String) -> Self {
        Self {
            query,
            query_type: QueryType::General,
            detected_types: HashSet::new(),
            confidence: 1.0,
            causal_direction: CausalDirection::Unknown,
            detected_language: None,
            keywords: Vec::new(),
            potential_entities: Vec::new(),
            is_question: false,
            recommended_embedders: vec![0], // E1 only by default
        }
    }
}

// ============================================================================
// QUERY TYPE ANALYZER
// ============================================================================

/// Query type analyzer for intelligent embedder selection.
///
/// Analyzes queries to detect type and extract features that help
/// determine which embedders will be most useful.
///
/// # Thread Safety
///
/// This struct is stateless and thread-safe.
pub struct QueryTypeAnalyzer;

impl QueryTypeAnalyzer {
    /// Create a new query analyzer.
    pub fn new() -> Self {
        Self
    }

    /// Analyze a query to detect type and extract features.
    ///
    /// # Arguments
    /// * `query` - The query text to analyze
    ///
    /// # Returns
    /// `QueryAnalysis` with detected type and features
    pub fn analyze(&self, query: &str) -> QueryAnalysis {
        let query = query.trim();

        if query.len() < MIN_QUERY_LEN {
            return QueryAnalysis::new(query.to_string());
        }

        let query_lower = query.to_lowercase();
        let mut analysis = QueryAnalysis::new(query.to_string());

        // Detect if question
        analysis.is_question = query.ends_with('?')
            || query_lower.starts_with("what ")
            || query_lower.starts_with("why ")
            || query_lower.starts_with("how ")
            || query_lower.starts_with("where ")
            || query_lower.starts_with("when ")
            || query_lower.starts_with("who ")
            || query_lower.starts_with("which ");

        // Extract potential entities (capitalized words)
        analysis.potential_entities = self.extract_entities(query);

        // Extract keywords (significant words)
        analysis.keywords = self.extract_keywords(&query_lower);

        // Detect query types
        let mut type_scores: Vec<(QueryType, f32)> = Vec::new();

        // Check for Code type
        let code_score = self.score_code_query(&query_lower, query);
        if code_score > 0.3 {
            analysis.detected_types.insert(QueryType::Code);
            type_scores.push((QueryType::Code, code_score));

            // Detect programming language
            analysis.detected_language = self.detect_language(&query_lower);
        }

        // Check for Causal type
        let (causal_score, causal_dir) = self.score_causal_query(&query_lower);
        if causal_score > 0.3 {
            analysis.detected_types.insert(QueryType::Causal);
            analysis.causal_direction = causal_dir;
            type_scores.push((QueryType::Causal, causal_score));
        }

        // Check for Intent type
        let intent_score = self.score_intent_query(&query_lower);
        if intent_score > 0.3 {
            analysis.detected_types.insert(QueryType::Intent);
            type_scores.push((QueryType::Intent, intent_score));
        }

        // Check for Graph type
        let graph_score = self.score_graph_query(&query_lower);
        if graph_score > 0.3 {
            analysis.detected_types.insert(QueryType::Graph);
            type_scores.push((QueryType::Graph, graph_score));
        }

        // Check for Keyword type (quoted terms or explicit keyword markers)
        let keyword_score = self.score_keyword_query(&query_lower, query);
        if keyword_score > 0.3 {
            analysis.detected_types.insert(QueryType::Keyword);
            type_scores.push((QueryType::Keyword, keyword_score));
        }

        // Check for Entity type (many capitalized terms or known entity patterns)
        let entity_score = self.score_entity_query(&analysis.potential_entities, &query_lower);
        if entity_score > 0.3 {
            analysis.detected_types.insert(QueryType::Entity);
            type_scores.push((QueryType::Entity, entity_score));
        }

        // Select primary type (highest score)
        if let Some((primary_type, score)) = type_scores.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)) {
            analysis.query_type = *primary_type;
            analysis.confidence = *score;
        } else {
            analysis.query_type = QueryType::General;
            analysis.confidence = 1.0;
        }

        // Build recommended embedders
        analysis.recommended_embedders = self.build_recommended_embedders(&analysis);

        debug!(
            query = %analysis.query,
            query_type = %analysis.query_type.name(),
            confidence = analysis.confidence,
            detected_types = ?analysis.detected_types.iter().map(|t| t.name()).collect::<Vec<_>>(),
            recommended_embedders = ?analysis.recommended_embedders,
            "Query analysis complete"
        );

        analysis
    }

    /// Score how likely this is a code query.
    fn score_code_query(&self, query_lower: &str, query_original: &str) -> f32 {
        let mut score = 0.0f32;

        // Check for code keywords
        let keyword_matches = CODE_KEYWORDS.iter()
            .filter(|kw| query_lower.contains(*kw))
            .count();
        score += (keyword_matches as f32 * 0.15).min(0.6);

        // Check for code symbols
        let symbol_matches = CODE_SYMBOLS.iter()
            .filter(|sym| query_original.contains(*sym))
            .count();
        score += (symbol_matches as f32 * 0.1).min(0.3);

        // Check for language names
        if LANGUAGE_NAMES.iter().any(|lang| query_lower.contains(lang)) {
            score += 0.3;
        }

        // Check for camelCase or snake_case patterns
        if query_original.chars().any(|c| c == '_') || has_camel_case(query_original) {
            score += 0.15;
        }

        score.min(1.0)
    }

    /// Score how likely this is a causal query and detect direction.
    fn score_causal_query(&self, query_lower: &str) -> (f32, CausalDirection) {
        let mut score = 0.0f32;
        let mut direction = CausalDirection::Unknown;

        // Check for causal patterns
        let pattern_matches = CAUSAL_PATTERNS.iter()
            .filter(|p| query_lower.contains(*p))
            .count();
        score += (pattern_matches as f32 * 0.25).min(0.8);

        // Detect direction
        if query_lower.contains("why") || query_lower.contains("what caused")
            || query_lower.contains("reason") || query_lower.contains("due to") {
            direction = CausalDirection::SeekingCause;
            score += 0.2;
        } else if query_lower.contains("what happens") || query_lower.contains("effect")
            || query_lower.contains("result") || query_lower.contains("lead to") {
            direction = CausalDirection::SeekingEffect;
            score += 0.2;
        }

        (score.min(1.0), direction)
    }

    /// Score how likely this is an intent query.
    fn score_intent_query(&self, query_lower: &str) -> f32 {
        let pattern_matches = INTENT_PATTERNS.iter()
            .filter(|p| query_lower.contains(*p))
            .count();
        (pattern_matches as f32 * 0.3).min(1.0)
    }

    /// Score how likely this is a graph/relationship query.
    fn score_graph_query(&self, query_lower: &str) -> f32 {
        let pattern_matches = GRAPH_PATTERNS.iter()
            .filter(|p| query_lower.contains(*p))
            .count();
        (pattern_matches as f32 * 0.25).min(1.0)
    }

    /// Score how likely this is a keyword/exact match query.
    fn score_keyword_query(&self, query_lower: &str, query_original: &str) -> f32 {
        let mut score = 0.0f32;

        // Check for quoted terms (strongest signal)
        if query_original.contains('"') || query_original.contains('\'') {
            score += 0.6;
        }

        // Check for keyword patterns
        let pattern_matches = KEYWORD_PATTERNS.iter()
            .filter(|p| query_lower.contains(*p))
            .count();
        score += (pattern_matches as f32 * 0.2).min(0.4);

        score.min(1.0)
    }

    /// Score how likely this is an entity query.
    fn score_entity_query(&self, entities: &[String], query_lower: &str) -> f32 {
        let mut score = 0.0f32;

        // Many potential entities suggests entity query
        score += (entities.len() as f32 * 0.15).min(0.5);

        // Check for entity-related patterns
        if query_lower.contains("who is") || query_lower.contains("what is")
            || query_lower.contains("about ") || query_lower.contains("named ") {
            score += 0.3;
        }

        score.min(1.0)
    }

    /// Detect programming language mentioned in query.
    fn detect_language(&self, query_lower: &str) -> Option<DetectedLanguage> {
        for lang in LANGUAGE_NAMES {
            if query_lower.contains(lang) {
                return Some(DetectedLanguage {
                    name: lang.to_string(),
                    confidence: 0.9,
                });
            }
        }
        None
    }

    /// Extract potential named entities (capitalized terms).
    fn extract_entities(&self, query: &str) -> Vec<String> {
        let mut entities = Vec::new();

        for word in query.split_whitespace() {
            let word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if word.len() >= 2 {
                let first_char = word.chars().next().unwrap();
                if first_char.is_uppercase() && !is_sentence_start(query, word) {
                    entities.push(word.to_string());
                }
            }
        }

        entities
    }

    /// Extract significant keywords from query.
    fn extract_keywords(&self, query_lower: &str) -> Vec<String> {
        const STOP_WORDS: &[&str] = &[
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "also", "now", "about",
            "it", "its", "this", "that", "these", "those", "i", "me", "my",
            "we", "our", "you", "your", "he", "him", "his", "she", "her",
            "they", "them", "their", "what", "which", "who", "whom", "and", "or",
        ];

        query_lower
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| w.len() >= 2 && !STOP_WORDS.contains(w))
            .map(String::from)
            .collect()
    }

    /// Build recommended embedders list based on analysis.
    fn build_recommended_embedders(&self, analysis: &QueryAnalysis) -> Vec<usize> {
        let mut embedders: HashSet<usize> = HashSet::new();

        // E1 is always included (foundation)
        embedders.insert(0);

        // Add embedders for all detected types
        for query_type in &analysis.detected_types {
            for embedder in query_type.recommended_embedders() {
                embedders.insert(embedder);
            }
        }

        // If no special types detected, just use E1
        if analysis.detected_types.is_empty() {
            return vec![0];
        }

        let mut result: Vec<_> = embedders.into_iter().collect();
        result.sort();
        result
    }
}

impl Default for QueryTypeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Check if a word is at the start of a sentence.
fn is_sentence_start(text: &str, word: &str) -> bool {
    if let Some(pos) = text.find(word) {
        if pos == 0 {
            return true;
        }
        let prev_char = text[..pos].chars().last();
        matches!(prev_char, Some('.') | Some('!') | Some('?') | Some('\n'))
    } else {
        false
    }
}

/// Check if text contains camelCase patterns.
fn has_camel_case(text: &str) -> bool {
    let mut prev_lower = false;
    for c in text.chars() {
        if prev_lower && c.is_uppercase() {
            return true;
        }
        prev_lower = c.is_lowercase();
    }
    false
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn analyzer() -> QueryTypeAnalyzer {
        QueryTypeAnalyzer::new()
    }

    // ========================================================================
    // CODE DETECTION TESTS
    // ========================================================================

    #[test]
    fn test_code_query_with_keywords() {
        println!("=== TEST: Code Query With Keywords ===");

        let result = analyzer().analyze("How do I implement a function in Rust?");

        println!("Query: {}", result.query);
        println!("Type: {:?}", result.query_type);
        println!("Confidence: {}", result.confidence);
        println!("Detected language: {:?}", result.detected_language);

        assert_eq!(result.query_type, QueryType::Code);
        assert!(result.detected_language.is_some());
        assert_eq!(result.detected_language.unwrap().name, "rust");

        println!("[VERIFIED] Code query detection with keywords");
    }

    #[test]
    fn test_code_query_with_symbols() {
        println!("=== TEST: Code Query With Symbols ===");

        let result = analyzer().analyze("What does fn foo() -> Result<T, E> do?");

        println!("Type: {:?}", result.query_type);
        println!("Confidence: {}", result.confidence);

        assert_eq!(result.query_type, QueryType::Code);

        println!("[VERIFIED] Code query detection with symbols");
    }

    // ========================================================================
    // CAUSAL DETECTION TESTS
    // ========================================================================

    #[test]
    fn test_causal_query_seeking_cause() {
        println!("=== TEST: Causal Query Seeking Cause ===");

        let result = analyzer().analyze("Why did the server crash?");

        println!("Type: {:?}", result.query_type);
        println!("Causal direction: {:?}", result.causal_direction);

        assert_eq!(result.query_type, QueryType::Causal);
        assert_eq!(result.causal_direction, CausalDirection::SeekingCause);

        println!("[VERIFIED] Causal query seeking cause");
    }

    #[test]
    fn test_causal_query_seeking_effect() {
        println!("=== TEST: Causal Query Seeking Effect ===");

        let result = analyzer().analyze("What happens as a result of the deadline being missed?");

        println!("Type: {:?}", result.query_type);
        println!("Causal direction: {:?}", result.causal_direction);

        assert_eq!(result.query_type, QueryType::Causal);
        assert_eq!(result.causal_direction, CausalDirection::SeekingEffect);

        println!("[VERIFIED] Causal query seeking effect");
    }

    // ========================================================================
    // INTENT DETECTION TESTS
    // ========================================================================

    #[test]
    fn test_intent_query() {
        println!("=== TEST: Intent Query ===");

        let result = analyzer().analyze("What was the goal of the authentication refactor?");

        println!("Type: {:?}", result.query_type);
        println!("Confidence: {}", result.confidence);

        assert_eq!(result.query_type, QueryType::Intent);

        println!("[VERIFIED] Intent query detection");
    }

    // ========================================================================
    // GRAPH DETECTION TESTS
    // ========================================================================

    #[test]
    fn test_graph_query() {
        println!("=== TEST: Graph Query ===");

        let result = analyzer().analyze("What modules import the database connection?");

        println!("Type: {:?}", result.query_type);
        println!("Confidence: {}", result.confidence);

        assert_eq!(result.query_type, QueryType::Graph);

        println!("[VERIFIED] Graph query detection");
    }

    // ========================================================================
    // KEYWORD DETECTION TESTS
    // ========================================================================

    #[test]
    fn test_keyword_query_quoted() {
        println!("=== TEST: Keyword Query Quoted ===");

        let result = analyzer().analyze("Find all mentions of \"TeleologicalFingerprint\"");

        println!("Type: {:?}", result.query_type);
        println!("Confidence: {}", result.confidence);

        assert_eq!(result.query_type, QueryType::Keyword);

        println!("[VERIFIED] Keyword query with quotes");
    }

    // ========================================================================
    // ENTITY DETECTION TESTS
    // ========================================================================

    #[test]
    fn test_entity_query() {
        println!("=== TEST: Entity Query ===");

        let result = analyzer().analyze("What is Diesel ORM and how does it work with PostgreSQL?");

        println!("Type: {:?}", result.query_type);
        println!("Entities: {:?}", result.potential_entities);

        // This should detect Entity type due to capitalized terms
        assert!(result.detected_types.contains(&QueryType::Entity));

        println!("[VERIFIED] Entity query detection");
    }

    // ========================================================================
    // GENERAL QUERY TESTS
    // ========================================================================

    #[test]
    fn test_general_query() {
        println!("=== TEST: General Query ===");

        let result = analyzer().analyze("memory consolidation patterns");

        println!("Type: {:?}", result.query_type);
        println!("Detected types: {:?}", result.detected_types);

        assert_eq!(result.query_type, QueryType::General);

        println!("[VERIFIED] General query detection");
    }

    // ========================================================================
    // RECOMMENDED EMBEDDERS TESTS
    // ========================================================================

    #[test]
    fn test_recommended_embedders() {
        println!("=== TEST: Recommended Embedders ===");

        // Code query should recommend E1 + E7
        let code = analyzer().analyze("How do I implement async in Rust?");
        println!("Code query embedders: {:?}", code.recommended_embedders);
        assert!(code.recommended_embedders.contains(&0)); // E1
        assert!(code.recommended_embedders.contains(&6)); // E7

        // Causal query should recommend E1 + E5
        let causal = analyzer().analyze("Why did the test fail?");
        println!("Causal query embedders: {:?}", causal.recommended_embedders);
        assert!(causal.recommended_embedders.contains(&0)); // E1
        assert!(causal.recommended_embedders.contains(&4)); // E5

        println!("[VERIFIED] Recommended embedders");
    }

    // ========================================================================
    // KEYWORD EXTRACTION TESTS
    // ========================================================================

    #[test]
    fn test_keyword_extraction() {
        println!("=== TEST: Keyword Extraction ===");

        let result = analyzer().analyze("implementing async database connections in Rust");

        println!("Keywords: {:?}", result.keywords);

        assert!(result.keywords.contains(&"implementing".to_string()));
        assert!(result.keywords.contains(&"async".to_string()));
        assert!(result.keywords.contains(&"database".to_string()));
        assert!(result.keywords.contains(&"connections".to_string()));
        assert!(result.keywords.contains(&"rust".to_string()));

        // Should not contain stop words
        assert!(!result.keywords.contains(&"in".to_string()));

        println!("[VERIFIED] Keyword extraction");
    }

    // ========================================================================
    // VERIFICATION LOG
    // ========================================================================

    #[test]
    fn test_verification_log() {
        println!("\n=== QUERY_ANALYZER.RS VERIFICATION LOG ===\n");

        println!("Query Types:");
        for qt in [QueryType::General, QueryType::Causal, QueryType::Code,
                   QueryType::Entity, QueryType::Intent, QueryType::Keyword,
                   QueryType::Graph] {
            println!("  - {}: embedders {:?}", qt.name(), qt.recommended_embedders());
        }

        println!("\nPattern Counts:");
        println!("  - Code keywords: {}", CODE_KEYWORDS.len());
        println!("  - Code symbols: {}", CODE_SYMBOLS.len());
        println!("  - Language names: {}", LANGUAGE_NAMES.len());
        println!("  - Causal patterns: {}", CAUSAL_PATTERNS.len());
        println!("  - Intent patterns: {}", INTENT_PATTERNS.len());
        println!("  - Graph patterns: {}", GRAPH_PATTERNS.len());
        println!("  - Keyword patterns: {}", KEYWORD_PATTERNS.len());

        println!("\nTest Coverage:");
        println!("  - Code detection: 2 tests");
        println!("  - Causal detection: 2 tests");
        println!("  - Intent detection: 1 test");
        println!("  - Graph detection: 1 test");
        println!("  - Keyword detection: 1 test");
        println!("  - Entity detection: 1 test");
        println!("  - General detection: 1 test");
        println!("  - Recommended embedders: 1 test");
        println!("  - Keyword extraction: 1 test");
        println!("  - Total: 11 tests");

        println!("\nVERIFICATION COMPLETE");
    }
}
