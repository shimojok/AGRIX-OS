"""
AGRIX-OS :: agriware_engine.py
==============================
AgriWare™ — MBT55 Injection Timing & Prescription Engine

Implements the 5-phase Eco-Phenotype diagnostic system:
  ACCELERATING_IMPROVEMENT  → sustain current state
  DIMINISHING_IMPROVEMENT   → prepare next intervention
  STABLE                    → maintenance dosing
  DIMINISHING_DEGRADATION   → intervention in effect; monitor
  ACCELERATING_DEGRADATION  → EMERGENCY: immediate MBT55 activation

Design rule: "Soil is a battery. Every decision starts from the electron
budget (H₂, Fe²⁺/Fe³⁺, Mn²⁺/Mn⁴⁺)."  — AGENTS.md Domain Rule #1
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ─────────────────────────────────────────────
# 1. Phase classification
# ─────────────────────────────────────────────

class EcoPhenotype(str, Enum):
    ACCELERATING_IMPROVEMENT  = "ACCELERATING_IMPROVEMENT"
    DIMINISHING_IMPROVEMENT   = "DIMINISHING_IMPROVEMENT"
    STABLE                    = "STABLE"
    DIMINISHING_DEGRADATION   = "DIMINISHING_DEGRADATION"
    ACCELERATING_DEGRADATION  = "ACCELERATING_DEGRADATION"


@dataclass
class PhaseResult:
    phase:       EcoPhenotype
    confidence:  float          # 0.0 – 1.0
    velocity:    float          # rate of change of SHE score (score/h)
    acceleration: float         # second derivative (score/h²)
    primary_cause: str          # e.g. "microbial_diversity_loss"
    she_score:   float          # current SHE™ Health Index (0–1)


# ─────────────────────────────────────────────
# 2. SHE™ score calculator
# ─────────────────────────────────────────────

def compute_she_score(
    X_A: float, X_B: float, X_C: float,
    P: float, C: float, N: float,
    pH: float, Eh_mV: float,
    X_A_ref: float = 1.0, C_ref: float = 45.0, N_ref: float = 3.5,
) -> float:
    """
    Soil Health & Ecosystem (SHE™) composite score.
    Weighted sum of normalised biological and chemical indicators.
    Score range: 0.0 (severely degraded) – 1.0 (optimal).
    """
    # Microbial activity sub-score (40% weight)
    total_biomass = X_A + X_B + X_C
    micro_score = np.tanh(total_biomass / (X_A_ref * 3.0))

    # Humic product sub-score (25% weight) — proxy for soil structure
    humic_score = np.tanh(P / 10.0)

    # Carbon & Nitrogen sub-score (20% weight)
    cn_score = (np.tanh(C / C_ref) + np.tanh(N / N_ref)) / 2.0

    # Physico-chemical sub-score (15% weight)
    ph_score  = 1.0 - abs(pH - 7.0) / 3.5
    eh_score  = 1.0 / (1.0 + np.exp(-0.01 * (Eh_mV - 50)))
    phys_score = (ph_score + eh_score) / 2.0

    she = (0.40 * micro_score
           + 0.25 * humic_score
           + 0.20 * cn_score
           + 0.15 * phys_score)

    return float(np.clip(she, 0.0, 1.0))


# ─────────────────────────────────────────────
# 3. Phase diagnostic engine
# ─────────────────────────────────────────────

def diagnose_phase(
    she_history: list[float],
    window_h: int = 3,
    confidence_threshold: float = 0.60,
) -> PhaseResult:
    """
    Classify current eco-phenotype from SHE™ score time-series.

    Uses first-order (velocity) and second-order (acceleration) derivatives
    to distinguish actively improving vs. recovering, etc.

    Args:
        she_history : list of SHE scores, most recent LAST, 1 per hour
        window_h    : number of recent points used for derivative estimation

    Returns:
        PhaseResult with phase, confidence, velocity, acceleration
    """
    if len(she_history) < 2:
        return PhaseResult(EcoPhenotype.STABLE, 0.5, 0.0, 0.0, "insufficient_data",
                           she_history[-1] if she_history else 0.5)

    scores = np.array(she_history[-max(window_h + 2, 4):])

    # Fit linear trend to last window (velocity)
    n = len(scores)
    t = np.arange(n, dtype=float)
    if n >= 3:
        coeffs = np.polyfit(t, scores, 2)  # quadratic fit
        velocity     = float(np.polyval(np.polyder(coeffs, 1), n - 1))
        acceleration = float(coeffs[0] * 2)   # second derivative of quadratic
    else:
        velocity     = float(scores[-1] - scores[-2])
        acceleration = 0.0

    she_now = float(scores[-1])

    # ── Phase classification ──────────────────────────────────────
    v_thr = 0.005   # meaningful velocity threshold (score/h)
    a_thr = 0.001   # meaningful acceleration threshold

    if velocity > v_thr and acceleration >= -a_thr:
        phase = EcoPhenotype.ACCELERATING_IMPROVEMENT
    elif velocity > v_thr and acceleration < -a_thr:
        phase = EcoPhenotype.DIMINISHING_IMPROVEMENT
    elif velocity < -v_thr and acceleration <= a_thr:
        phase = EcoPhenotype.ACCELERATING_DEGRADATION
    elif velocity < -v_thr and acceleration > a_thr:
        phase = EcoPhenotype.DIMINISHING_DEGRADATION
    else:
        phase = EcoPhenotype.STABLE

    # ── Confidence based on signal-to-noise in the window ────────
    noise = float(np.std(np.diff(scores)))
    signal = abs(velocity)
    confidence = float(np.clip(signal / (signal + noise + 1e-6), 0.0, 1.0))
    if confidence < confidence_threshold:
        phase = EcoPhenotype.STABLE   # revert to safe state when uncertain

    # ── Root-cause inference (simplified rule-based) ──────────────
    cause = _infer_primary_cause(she_history, velocity, acceleration, she_now)

    return PhaseResult(phase, confidence, velocity, acceleration, cause, she_now)


def _infer_primary_cause(
    she_history: list[float],
    velocity: float,
    acceleration: float,
    she_now: float,
) -> str:
    """Simple rule-based root-cause for prescription routing."""
    if she_now < 0.35:
        return "severe_soil_degradation"
    if she_now < 0.50 and velocity < -0.01:
        return "microbial_diversity_loss"
    if 0.50 <= she_now < 0.65 and velocity < 0:
        return "mineral_imbalance"
    if velocity < -0.005 and acceleration < -0.001:
        return "cascade_failure_risk"
    if she_now >= 0.75 and velocity < 0:
        return "maintenance_needed"
    return "general_stress"


# ─────────────────────────────────────────────
# 4. Prescription generator
# ─────────────────────────────────────────────

@dataclass
class Prescription:
    """MBT55 / HMT / NASARA intervention specification."""
    mbt55_L_ha:  float = 0.0     # MBT55 dose (L/ha)
    hmt_L_ha:    float = 0.0     # HMT mineral supplement (L/ha)
    nasara_kg_ha: float = 0.0    # NASARA soil conditioner (kg/ha)
    application_mode: str = "irrigation_mix"  # irrigation_mix | foliar | direct
    urgency: str = "routine"     # emergency | urgent | routine | none
    rationale: str = ""
    monitoring_interval_h: int = 6  # next sensor check in hours
    target_phenotype: str = ""   # expected SHE score direction

    def to_dict(self) -> dict:
        return self.__dict__

    def is_zero_dose(self) -> bool:
        return (self.mbt55_L_ha == 0 and
                self.hmt_L_ha == 0 and
                self.nasara_kg_ha == 0)


def generate_prescription(
    phase_result: PhaseResult,
    current_temperature_C: float = 25.0,
    cumulative_mbt55_this_week: float = 0.0,
    max_weekly_dose_L_ha: float = 20.0,
) -> Prescription:
    """
    Map eco-phenotype phase to concrete MBT55/HMT/NASARA prescription.

    Respects the 24-hour biological clock of MBT55 (AGENTS.md rule).
    No synthetic NPK recommendations — microbial management path only.

    Args:
        phase_result    : output of diagnose_phase()
        current_temperature_C : for dose adjustment (heat reduces viability)
        cumulative_mbt55_this_week : rate-limiter to avoid over-application
    """
    phase = phase_result.phase
    cause = phase_result.primary_cause
    she   = phase_result.she_score

    # Remaining weekly budget
    budget = max(0.0, max_weekly_dose_L_ha - cumulative_mbt55_this_week)

    # Temperature dose modifier: above 100°C in reactor context inoculum
    # is added post-cool-down; field temperature > 38°C reduces viability
    temp_factor = 1.0 if current_temperature_C <= 38 else max(0.3,
                   1.0 - (current_temperature_C - 38) / 40)

    # ── Phase-specific prescriptions ─────────────────────────────

    if phase == EcoPhenotype.ACCELERATING_DEGRADATION:
        base_mbt55 = 5.0 if she < 0.40 else 3.0
        rx = Prescription(
            mbt55_L_ha   = min(base_mbt55 * temp_factor, budget),
            hmt_L_ha     = 1.0,
            nasara_kg_ha = 0.0,
            application_mode = "irrigation_mix",
            urgency      = "emergency",
            monitoring_interval_h = 3,
            rationale    = (f"Emergency: {cause}. SHE={she:.2f}. "
                            f"Immediate MBT55 activation + mineral support."),
            target_phenotype = "SHE velocity → positive within 12h",
        )

    elif phase == EcoPhenotype.DIMINISHING_DEGRADATION:
        rx = Prescription(
            mbt55_L_ha   = min(2.0 * temp_factor, budget),
            hmt_L_ha     = 0.5,
            nasara_kg_ha = 10.0,
            application_mode = "irrigation_mix",
            urgency      = "urgent",
            monitoring_interval_h = 6,
            rationale    = ("Degradation decelerating — intervention taking effect. "
                            "NASARA added to lock in structural improvement."),
            target_phenotype = "SHE enters STABLE within 24h",
        )

    elif phase == EcoPhenotype.STABLE:
        rx = Prescription(
            mbt55_L_ha   = min(1.0 * temp_factor, budget),
            hmt_L_ha     = 0.3,
            nasara_kg_ha = 5.0,
            application_mode = "irrigation_mix",
            urgency      = "routine",
            monitoring_interval_h = 12,
            rationale    = "Maintenance dosing. Sustain microbial diversity.",
            target_phenotype = "Maintain SHE ≥ current level",
        )

    elif phase == EcoPhenotype.DIMINISHING_IMPROVEMENT:
        rx = Prescription(
            mbt55_L_ha   = min(1.5 * temp_factor, budget),
            hmt_L_ha     = 0.5,
            nasara_kg_ha = 15.0,
            application_mode = "irrigation_mix",
            urgency      = "routine",
            monitoring_interval_h = 8,
            rationale    = ("Improvement slowing. NASARA boost to consolidate "
                            "humic structure and extend improvement trajectory."),
            target_phenotype = "Re-accelerate improvement or maintain plateau",
        )

    else:  # ACCELERATING_IMPROVEMENT
        rx = Prescription(
            mbt55_L_ha   = 0.0,
            hmt_L_ha     = 0.2,
            nasara_kg_ha = 5.0,
            application_mode = "none",
            urgency      = "none",
            monitoring_interval_h = 24,
            rationale    = ("System in optimal hypercycle phase. "
                            "Minimal intervention. Monitor only."),
            target_phenotype = "Sustain ACCELERATING_IMPROVEMENT",
        )

    return rx


# ─────────────────────────────────────────────
# 5. PBPE impact forecaster
# ─────────────────────────────────────────────

@dataclass
class ImpactForecast:
    """Predicted environmental & economic impact of a prescription."""
    carbon_seq_delta_tCO2_ha_yr: float   # incremental sequestration
    she_score_48h_projected: float
    pbpe_tokens_predicted: float         # 1 PBPE = 10 kg CO2e
    confidence: float


def forecast_impact(
    phase: PhaseResult,
    prescription: Prescription,
    current_carbon_seq_rate: float,      # tCO2/ha/year from soil_dynamics
    field_area_ha: float = 1.0,
) -> ImpactForecast:
    """
    Predict PBPE token issuance from a given prescription.
    Feeds the SafelyChain™ Impact Ledger (status: Predicted).
    """
    # Phase-response multipliers on carbon sequestration
    phase_multiplier = {
        EcoPhenotype.ACCELERATING_IMPROVEMENT:  1.20,
        EcoPhenotype.DIMINISHING_IMPROVEMENT:   1.05,
        EcoPhenotype.STABLE:                    1.00,
        EcoPhenotype.DIMINISHING_DEGRADATION:   0.85,
        EcoPhenotype.ACCELERATING_DEGRADATION:  0.60,
    }

    multiplier = phase_multiplier[phase.phase]
    mbt55_boost = prescription.mbt55_L_ha * 0.08   # empirical: 0.08 tCO2/L/ha

    delta_seq = (current_carbon_seq_rate * (multiplier - 1.0)
                 + mbt55_boost)

    # SHE projection: simple extrapolation
    velocity_boost = 0.02 * prescription.mbt55_L_ha / 5.0
    she_48h = float(np.clip(
        phase.she_score + velocity_boost * 48,
        0.0, 1.0
    ))

    # PBPE token calculation (1 PBPE = 10 kg CO2e = 0.01 tCO2e)
    annual_carbon = (current_carbon_seq_rate + delta_seq) * field_area_ha
    pbpe_annual = annual_carbon / 0.01   # tokens per year

    return ImpactForecast(
        carbon_seq_delta_tCO2_ha_yr  = delta_seq,
        she_score_48h_projected      = she_48h,
        pbpe_tokens_predicted        = pbpe_annual,
        confidence                   = phase.confidence,
    )


# ─────────────────────────────────────────────
# 6. AgriWare™ main engine loop
# ─────────────────────────────────────────────

class AgriWareEngine:
    """
    AgriWare™ — top-level orchestrator.

    Usage:
        engine = AgriWareEngine(farm_id="KE_COFFEE_001")
        metadata = engine.run_cycle(
            she_history=[...],
            sim_result={...},       # from soil_dynamics.run_soil_simulation
            sensor=current_sensor,
        )
    """

    def __init__(self, farm_id: str, field_area_ha: float = 1.0):
        self.farm_id       = farm_id
        self.field_area_ha = field_area_ha
        self._cumulative_mbt55_this_week = 0.0
        self._cycle_count  = 0

    def run_cycle(
        self,
        she_history: list[float],
        sim_result:  dict,
        sensor_pH:   float = 7.0,
        sensor_Eh:   float = 200.0,
        sensor_temp: float = 25.0,
    ) -> dict:
        """Execute one diagnostic–prescription–forecast cycle."""
        self._cycle_count += 1

        # 1. Diagnose
        phase = diagnose_phase(she_history)

        # 2. Prescribe
        rx = generate_prescription(
            phase,
            current_temperature_C         = sensor_temp,
            cumulative_mbt55_this_week    = self._cumulative_mbt55_this_week,
        )
        self._cumulative_mbt55_this_week += rx.mbt55_L_ha

        # 3. Forecast impact → PBPE
        carbon_seq = float(sim_result.get("carbon_seq_rate", np.array([1.5]))[-1])
        forecast = forecast_impact(phase, rx, carbon_seq, self.field_area_ha)

        # 4. Build SafelyChain™ metadata payload
        metadata = {
            "farm_id":        self.farm_id,
            "cycle":          self._cycle_count,
            "phase":          phase.phase.value,
            "phase_confidence": round(phase.confidence, 3),
            "she_score":      round(phase.she_score, 3),
            "velocity":       round(phase.velocity, 5),
            "acceleration":   round(phase.acceleration, 6),
            "primary_cause":  phase.primary_cause,
            "prescription":   rx.to_dict(),
            "impact_forecast": {
                "carbon_seq_delta_tCO2_ha_yr": round(forecast.carbon_seq_delta_tCO2_ha_yr, 3),
                "she_score_48h_projected":     round(forecast.she_score_48h_projected, 3),
                "pbpe_tokens_predicted":       round(forecast.pbpe_tokens_predicted, 1),
                "confidence":                  round(forecast.confidence, 3),
            },
            "safelychain_status": "Predicted",   # → Safely Chain™ ledger
        }

        return metadata


# ─────────────────────────────────────────────
# 7. CLI smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate declining SHE score → emergency response
    she_series = [0.72, 0.70, 0.67, 0.63, 0.58, 0.51, 0.44]

    engine = AgriWareEngine("AGRIX_DEMO_FARM_001", field_area_ha=5.0)

    # Fake sim result
    fake_sim = {"carbon_seq_rate": np.array([1.5, 1.4, 1.2, 1.1, 1.0])}

    result = engine.run_cycle(
        she_history = she_series,
        sim_result  = fake_sim,
        sensor_pH   = 6.2,
        sensor_Eh   = 80.0,
        sensor_temp = 28.0,
    )

    print("=== AgriWare™ Diagnostic Cycle ===")
    print(f"  Phase         : {result['phase']}")
    print(f"  SHE score     : {result['she_score']}")
    print(f"  Primary cause : {result['primary_cause']}")
    print(f"  Prescription  : MBT55={result['prescription']['mbt55_L_ha']} L/ha  "
          f"| HMT={result['prescription']['hmt_L_ha']} L/ha  "
          f"| Urgency={result['prescription']['urgency']}")
    print(f"  PBPE forecast : {result['impact_forecast']['pbpe_tokens_predicted']} tokens/year")
    print(f"  Chain status  : {result['safelychain_status']}")
