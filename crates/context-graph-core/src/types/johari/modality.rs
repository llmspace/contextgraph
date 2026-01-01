//! Input modality classification for memory nodes.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Input modality classification.
///
/// Classifies the type of content stored in a memory node.
/// Used for embedding model selection and content type detection.
///
/// # Embedding Model Mapping
/// From constitution.yaml Section 12-MODEL EMBEDDING:
/// - Text: E1_Semantic (1024D)
/// - Code: E7_Code (1536D)
/// - Image: E10_Multimodal (1024D)
/// - Audio: E10_Multimodal (1024D)
/// - Structured: E1_Semantic (1024D)
/// - Mixed: E1_Semantic (1024D)
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    /// Plain text content
    #[default]
    Text,
    /// Source code
    Code,
    /// Image data
    Image,
    /// Audio data
    Audio,
    /// Structured data (JSON, XML, etc.)
    Structured,
    /// Mixed modalities
    Mixed,
}

impl Modality {
    /// Detect modality from content string by analyzing patterns.
    ///
    /// # Detection Order (most specific first):
    /// 1. Code patterns (fn, def, class, import, etc.)
    /// 2. Structured data (JSON/YAML markers)
    /// 3. Data URIs (image/audio)
    /// 4. Default: Text
    ///
    /// # Examples
    /// ```
    /// use context_graph_core::types::Modality;
    /// assert_eq!(Modality::detect("fn main() {}"), Modality::Code);
    /// assert_eq!(Modality::detect("{\"key\": 1}"), Modality::Structured);
    /// assert_eq!(Modality::detect("Hello world"), Modality::Text);
    /// ```
    pub fn detect(content: &str) -> Self {
        // Code patterns - case-sensitive, must include space after keyword
        const CODE_PATTERNS: &[&str] = &[
            "fn ",
            "def ",
            "class ",
            "import ",
            "function ",
            "const ",
            "let ",
            "var ",
            "pub ",
            "async ",
            "impl ",
            "struct ",
            "enum ",
            "mod ",
            "use ",
            "package ",
            "func ",
            "export ",
            "from ",
            "#include",
            "#define",
        ];

        for pattern in CODE_PATTERNS {
            if content.contains(pattern) {
                return Self::Code;
            }
        }

        // Structured data detection
        let trimmed = content.trim();
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            return Self::Structured;
        }

        // YAML detection: lines starting with word followed by colon
        if content.lines().any(|line| {
            let t = line.trim();
            !t.is_empty() && !t.starts_with('#') && t.contains(": ")
        }) {
            return Self::Structured;
        }

        // Data URI detection
        if content.starts_with("data:image") {
            return Self::Image;
        }
        if content.starts_with("data:audio") {
            return Self::Audio;
        }

        Self::Text
    }

    /// Returns common file extensions for this modality (lowercase, no dots).
    ///
    /// # Returns
    /// Static slice of extension strings. Empty slice for Mixed modality.
    pub fn file_extensions(&self) -> &'static [&'static str] {
        match self {
            Self::Text => &["txt", "md", "rst", "adoc"],
            Self::Code => &[
                "rs", "py", "js", "ts", "go", "java", "c", "cpp", "h", "rb", "php",
            ],
            Self::Image => &["png", "jpg", "jpeg", "gif", "svg", "webp", "bmp"],
            Self::Audio => &["mp3", "wav", "ogg", "flac", "m4a", "aac"],
            Self::Structured => &["json", "yaml", "yml", "toml", "xml"],
            Self::Mixed => &[],
        }
    }

    /// Returns the primary embedding model ID per PRD spec.
    ///
    /// # Model Mapping (from constitution.yaml Section 12-MODEL EMBEDDING)
    /// - Text: E1_Semantic (1024D)
    /// - Code: E7_Code (1536D)
    /// - Image: E10_Multimodal (1024D)
    /// - Audio: E10_Multimodal (1024D)
    /// - Structured: E1_Semantic (1024D)
    /// - Mixed: E1_Semantic (1024D)
    pub fn primary_embedding_model(&self) -> &'static str {
        match self {
            Self::Text => "E1_Semantic",
            Self::Code => "E7_Code",
            Self::Image => "E10_Multimodal",
            Self::Audio => "E10_Multimodal",
            Self::Structured => "E1_Semantic",
            Self::Mixed => "E1_Semantic",
        }
    }

    /// Returns all modality variants as a fixed-size array.
    pub fn all() -> [Modality; 6] {
        [
            Self::Text,
            Self::Code,
            Self::Image,
            Self::Audio,
            Self::Structured,
            Self::Mixed,
        ]
    }
}

impl fmt::Display for Modality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text => write!(f, "Text"),
            Self::Code => write!(f, "Code"),
            Self::Image => write!(f, "Image"),
            Self::Audio => write!(f, "Audio"),
            Self::Structured => write!(f, "Structured"),
            Self::Mixed => write!(f, "Mixed"),
        }
    }
}
