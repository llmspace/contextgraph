//! GwtSystem - Kuramoto synchronization methods
//!
//! This module contains impl blocks for GwtSystem that handle
//! Kuramoto oscillator network operations and consciousness updates.

use std::time::Duration;

use crate::layers::KURAMOTO_DT;

use super::{GwtSystem, StateTransition};

impl GwtSystem {
    /// Step the Kuramoto network forward by elapsed duration
    ///
    /// Advances the oscillator phases according to Kuramoto dynamics:
    /// dθᵢ/dt = ωᵢ + (K/N)Σⱼ sin(θⱼ-θᵢ)
    ///
    /// # Arguments
    /// * `elapsed` - Time duration to advance the oscillators
    ///
    /// # Notes
    /// Uses multiple integration steps for numerical stability.
    /// The KURAMOTO_DT constant (0.01) is used as the base time step.
    pub async fn step_kuramoto(&self, elapsed: Duration) {
        let mut network = self.kuramoto.write().await;
        // Convert Duration to f32 seconds for the step function
        let dt = elapsed.as_secs_f32();
        // Use multiple integration steps for stability
        let steps = (dt / KURAMOTO_DT).ceil() as usize;
        for _ in 0..steps.max(1) {
            network.step(KURAMOTO_DT);
        }
    }

    /// Get current Kuramoto order parameter r (synchronization level)
    ///
    /// The order parameter measures phase synchronization:
    /// r = |1/N Σⱼ exp(iθⱼ)|
    ///
    /// # Returns
    /// * `f32` in [0, 1] where 1 = perfect sync, 0 = no sync
    pub async fn get_kuramoto_r(&self) -> f32 {
        let network = self.kuramoto.read().await;
        network.order_parameter()
    }

    /// Update consciousness with internal Kuramoto r value
    ///
    /// This method fetches r from the internal Kuramoto network
    /// instead of requiring the caller to pass it.
    ///
    /// # Arguments
    /// * `meta_accuracy` - Meta-UTL prediction accuracy [0,1]
    /// * `purpose_vector` - 13D purpose alignment vector
    ///
    /// # Returns
    /// * Consciousness level C(t) in [0, 1]
    pub async fn update_consciousness_auto(
        &self,
        meta_accuracy: f32,
        purpose_vector: &[f32; 13],
    ) -> crate::CoreResult<f32> {
        let kuramoto_r = self.get_kuramoto_r().await;
        self.update_consciousness(kuramoto_r, meta_accuracy, purpose_vector)
            .await
    }

    /// Update consciousness state with current Kuramoto order parameter and meta metrics
    pub async fn update_consciousness(
        &self,
        kuramoto_r: f32,
        meta_accuracy: f32,
        purpose_vector: &[f32; 13],
    ) -> crate::CoreResult<f32> {
        // Calculate consciousness level
        let consciousness = self.consciousness_calc.compute_consciousness(
            kuramoto_r,
            meta_accuracy,
            purpose_vector,
        )?;

        // Update state machine with new consciousness level
        let mut state_mgr = self.state_machine.write().await;
        let old_state = state_mgr.current_state();
        let new_state = state_mgr.update(consciousness).await?;

        if old_state != new_state {
            // Log state transition
            let transition = StateTransition {
                from: old_state,
                to: new_state,
                timestamp: chrono::Utc::now(),
                consciousness_level: consciousness,
            };
            // Transition logged for debugging
            tracing::debug!("State transition: {:?}", transition);
        }

        Ok(consciousness)
    }
}
