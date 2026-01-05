//! Type conversions for ModelId.

use super::core::ModelId;

impl TryFrom<u8> for ModelId {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Semantic),
            1 => Ok(Self::TemporalRecent),
            2 => Ok(Self::TemporalPeriodic),
            3 => Ok(Self::TemporalPositional),
            4 => Ok(Self::Causal),
            5 => Ok(Self::Sparse),
            6 => Ok(Self::Code),
            7 => Ok(Self::Graph),
            8 => Ok(Self::Hdc),
            9 => Ok(Self::Multimodal),
            10 => Ok(Self::Entity),
            11 => Ok(Self::LateInteraction),
            12 => Ok(Self::Splade),
            _ => Err("Invalid ModelId: must be 0-12"),
        }
    }
}
