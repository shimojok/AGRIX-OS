"""
AGRIX-OS :: soil_dynamics.py
============================
M³-BioSynergy Core — Soil Dynamics Engine
Extension of the MBT55 ODE system with real-time soil sensor inputs.

Theory basis: M³-BioSynergy System (Shimojo, BioNexus)
  - Microbial layer   : 120-species consortium (55% aerobic / 45% anaerobic)
  - Metabolic layer   : 3-stage enzyme cascade (0-4h / 4-12h / 12-24h)
  - Modular layer     : Ecological hypercycle with self-regulating feedback

Sensor inputs mapped to ODE parameters:
  pH  → modifies β_i (mortality / inhibition coefficients)
  Eh  → selects aerobic vs anaerobic pathway dominance
  θ   → moisture-dependent oxygen availability (aerobic/anaerobic switch)
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Optional
import json
import os


# ─────────────────────────────────────────────
# 1. State vector & parameter structures
# ─────────────────────────────────────────────

@dataclass
class SoilSensorReading:
    """Real-time sensor payload from IoT field sensors."""
    timestamp_h: float       # hours since start of simulation window
    pH: float                # soil solution pH  (typical: 4.5 – 8.5)
    Eh_mV: float             # redox potential   (mV, typical: -200 – +600)
    moisture_pct: float      # volumetric water content %  (0–100)
    temperature_C: float     # soil temperature °C (MBT55 range: 20–100)
    mbt55_dose_L_ha: float = 0.0   # MBT55 application this step (L/ha)


@dataclass
class M3Parameters:
    """
    ODE parameters for the M³-BioSynergy model.
    Loaded from parameters_optimized.json when available.
    Defaults represent calibrated MBT55 reference conditions.
    """
    # ── Monod kinetics ──────────────────────────────────────────
    mu_A: float = 0.80   # max growth rate, substrate degraders (h⁻¹)
    K_S:  float = 2.50   # half-saturation, substrate (g/L)
    mu_B: float = 0.60   # max growth rate, metabolite converters (h⁻¹)
    K_M:  float = 1.20   # half-saturation, metabolite (g/L)

    # ── Yield & decay ───────────────────────────────────────────
    alpha_A: float = 0.40  # biomass yield, group A
    beta_A:  float = 0.05  # decay rate, group A (h⁻¹)
    alpha_B: float = 0.35
    beta_B:  float = 0.04
    alpha_C: float = 0.30
    beta_C:  float = 0.03

    # ── Hypercycle coupling ─────────────────────────────────────
    gamma_C:  float = 0.20   # control signal sensitivity (L/g/h)
    kC0:      float = 0.15   # baseline control activation
    lambda_C: float = 0.50   # rate-of-change sensitivity

    # ── Product dynamics ────────────────────────────────────────
    pi_B:    float = 0.25   # product formation coefficient
    delta_P: float = 0.02   # product degradation rate (h⁻¹)
    K_P:     float = 5.00   # Hill half-saturation for control response

    # ── Intermediate & coupling ──────────────────────────────────
    eta_A:  float = 0.50   # metabolite generation fraction
    xi_B:   float = 0.08   # inhibition of X_B by metabolite
    omega_B: float = 0.12  # X_B stimulation of X_C

    # ── Carbon/Nitrogen dynamics ─────────────────────────────────
    sigma_B: float = 0.10   # carbon synthesis by X_B
    rho_B:   float = 0.08   # nitrogen synthesis by X_B
    k_C:     float = 0.05   # carbon loss rate
    k_N:     float = 0.04   # nitrogen loss rate

    # ── Environmental sensitivity ────────────────────────────────
    T_opt:    float = 90.0   # optimal temperature for MBT55 (°C)
    T_decay:  float = 20.0   # temperature decay constant
    pH_opt:   float = 7.2    # optimal pH
    phi_T:    float = 0.03   # temperature sensitivity coefficient
    psi_pH:   float = 0.15   # pH sensitivity coefficient


def load_parameters(json_path: Optional[str] = None) -> M3Parameters:
    """Load parameters from JSON; fall back to defaults."""
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        return M3Parameters(**{k: v for k, v in data.items()
                                if k in M3Parameters.__dataclass_fields__})
    return M3Parameters()


# ─────────────────────────────────────────────
# 2. Environmental modifier functions
# ─────────────────────────────────────────────

def environmental_stress(T: float, pH: float, p: M3Parameters) -> float:
    """
    β multiplier: how much does current T/pH deviate from optimum?
    Returns value > 1.0 when stressed (elevated decay), ≈ 1.0 at optimum.
    """
    delta_T  = ((T - p.T_opt) / p.T_decay) ** 2
    delta_pH = (pH - p.pH_opt) ** 2
    return np.exp(p.phi_T * delta_T + p.psi_pH * delta_pH)


def aerobic_fraction(Eh_mV: float, moisture_pct: float) -> float:
    """
    Estimate fraction of aerobic activity as function of Eh and moisture.
    MBT55 design: 55% aerobic / 45% anaerobic at reference conditions.
    High moisture or negative Eh → shift toward anaerobic dominance.
    """
    # Sigmoid on Eh: +200 mV → fully aerobic, -200 mV → fully anaerobic
    f_Eh = 1.0 / (1.0 + np.exp(-0.02 * (Eh_mV - 50)))
    # Moisture penalty: above 70% VWC oxygen diffusion drops sharply
    f_moisture = 1.0 - max(0.0, (moisture_pct - 70.0) / 30.0)
    f_moisture = max(0.1, f_moisture)
    return np.clip(f_Eh * f_moisture, 0.05, 0.95)


def mbt55_activation_pulse(dose_L_ha: float, X_A: float) -> float:
    """
    Inoculum boost: MBT55 application adds active microbial biomass.
    Converts dose (L/ha) to equivalent biomass density increment (g/L).
    Assumes MBT55 at 10^8 CFU/mL, ~0.1 g dry biomass per L applied.
    """
    return dose_L_ha * 0.1 / 10000.0   # L/ha → g/L (diluted across soil volume)


# ─────────────────────────────────────────────
# 3. Core ODE system (M³-BioSynergy + sensor coupling)
# ─────────────────────────────────────────────

def m3_ode(t: float, y: np.ndarray, p: M3Parameters,
           sensor_fn, carbon_nitrogen: bool = True) -> np.ndarray:
    """
    M³-BioSynergy ODE system — extended with real-time soil sensors.

    State vector y:
        [0] S   – substrate concentration (g/L)
        [1] X_A – substrate-degrading microbial biomass (g/L)
        [2] M   – metabolic intermediate (g/L)
        [3] X_B – metabolite-converting biomass (g/L)
        [4] P   – functional product / humic precursor (g/L)
        [5] X_C – hypercycle control biomass (g/L)
        [6] C   – soil organic carbon (g/L)
        [7] N   – available nitrogen (g/L)
    """
    S, X_A, M, X_B, P, X_C, C, N = y

    # ── Retrieve current sensor state ─────────────────────────────
    sensor: SoilSensorReading = sensor_fn(t)
    f_env  = environmental_stress(sensor.temperature_C, sensor.pH, p)
    f_aero = aerobic_fraction(sensor.Eh_mV, sensor.moisture_pct)
    pulse  = mbt55_activation_pulse(sensor.mbt55_dose_L_ha, X_A)

    # ── Effective kinetic rates (modified by environment) ─────────
    beta_A_eff = p.beta_A * f_env
    beta_B_eff = p.beta_B * f_env
    beta_C_eff = p.beta_C * f_env

    # Aerobic dominance amplifies X_A (55% fraction target)
    aero_boost = 0.55 + 0.45 * f_aero   # scales between 0.55 and 1.0

    # ── Monod growth terms ────────────────────────────────────────
    monod_S = max(S, 0) / (p.K_S + max(S, 0))
    monod_M = max(M, 0) / (p.K_M + max(M, 0))

    # ── Control activation (κ_C) — rate-of-change of metabolite ──
    # Approximated as proportional to current metabolite level
    kappa_C = p.kC0 * (1.0 - np.exp(-p.lambda_C * max(M, 1e-6)))

    # ── Hill function for product-driven control response ─────────
    hill_P = P ** 2 / (p.K_P ** 2 + P ** 2) if P > 0 else 0.0

    # ── ODE right-hand sides ─────────────────────────────────────
    dS_dt  = (-p.mu_A * monod_S * X_A * aero_boost
              + p.delta_P * P
              + pulse * 0.1)   # small carbon input from inoculum

    dXA_dt = (p.alpha_A * p.mu_A * monod_S * X_A * aero_boost
              - beta_A_eff * X_A
              + p.gamma_C * kappa_C * X_C
              + pulse)         # direct biomass addition

    dM_dt  = (p.eta_A * p.mu_A * monod_S * X_A
              - p.mu_B * monod_M * X_B)

    dXB_dt = (p.alpha_B * p.mu_B * monod_M * X_B
              - beta_B_eff * X_B
              - p.xi_B * max(M, 0) * X_B)

    dP_dt  = (p.pi_B * p.mu_B * monod_M * X_B
              - p.delta_P * P)

    dXC_dt = (p.alpha_C * hill_P * X_C
              + p.omega_B * p.mu_B * monod_M * X_B
              - beta_C_eff * X_C)

    # ── Carbon & Nitrogen dynamics ────────────────────────────────
    if carbon_nitrogen:
        dC_dt = (p.mu_A * monod_S * X_A
                 - p.k_C * C
                 + p.sigma_B * X_B)

        dN_dt = (p.alpha_A * p.mu_A * monod_S * X_A * 0.1
                 - p.k_N * N
                 + p.rho_B * X_B)
    else:
        dC_dt = 0.0
        dN_dt = 0.0

    return np.array([dS_dt, dXA_dt, dM_dt, dXB_dt, dP_dt, dXC_dt,
                     dC_dt, dN_dt])


# ─────────────────────────────────────────────
# 4. Simulation runner
# ─────────────────────────────────────────────

def run_soil_simulation(
    sensor_readings: list[SoilSensorReading],
    params: Optional[M3Parameters] = None,
    initial_state: Optional[np.ndarray] = None,
    t_span_h: tuple = (0.0, 24.0),
    n_eval: int = 200,
) -> dict:
    """
    Run the M³-BioSynergy soil dynamics simulation.

    Args:
        sensor_readings : chronological list of SoilSensorReading
        params          : M3Parameters (defaults to reference MBT55 params)
        initial_state   : y0 [S, X_A, M, X_B, P, X_C, C, N] (g/L)
        t_span_h        : simulation window in hours
        n_eval          : output resolution

    Returns:
        dict with keys: t, S, X_A, M, X_B, P, X_C, C, N,
                        aerobic_fraction, env_stress, carbon_seq_rate
    """
    if params is None:
        params = M3Parameters()

    if initial_state is None:
        initial_state = np.array([
            100.0,   # S   – substrate (g/L)
            1.0,     # X_A – aerobic degraders
            0.0,     # M   – intermediate
            0.5,     # X_B – converters
            0.0,     # P   – product
            0.1,     # X_C – control community
            45.0,    # C   – soil organic carbon
            3.5,     # N   – available nitrogen
        ])

    # Build interpolating sensor function
    times = np.array([r.timestamp_h for r in sensor_readings])

    def sensor_fn(t: float) -> SoilSensorReading:
        idx = np.searchsorted(times, t, side='right') - 1
        idx = np.clip(idx, 0, len(sensor_readings) - 1)
        return sensor_readings[int(idx)]

    t_eval = np.linspace(t_span_h[0], t_span_h[1], n_eval)

    sol = solve_ivp(
        fun=lambda t, y: m3_ode(t, y, params, sensor_fn),
        t_span=t_span_h,
        y0=initial_state,
        method='BDF',           # stiff solver — microbial dynamics are stiff
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
        max_step=0.5,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    S, X_A, M, X_B, P, X_C, C, N = sol.y

    # Derived outputs
    aero_frac   = np.array([aerobic_fraction(sensor_fn(t).Eh_mV,
                                              sensor_fn(t).moisture_pct)
                             for t in sol.t])
    env_stress  = np.array([environmental_stress(sensor_fn(t).temperature_C,
                                                  sensor_fn(t).pH, params)
                             for t in sol.t])

    # Carbon sequestration rate (tCO2/ha/year) — key PBPE Finance output
    # Approx: delta_C (g/L) × soil depth 0.3m × bulk density 1200 kg/m³
    # × 10000 m²/ha × 1e-6 t/g × 44/12 CO2/C × 8760h/year / sim_hours
    delta_C = np.maximum(C - initial_state[6], 0)
    CONVERSION = 0.3 * 1200 * 10000 * 1e-6 * (44/12) * (8760 / (t_span_h[1] - t_span_h[0]))
    carbon_seq_rate = delta_C * CONVERSION   # tCO2/ha/year

    return {
        "t":               sol.t,
        "S":               S,
        "X_A":             X_A,
        "M":               M,
        "X_B":             X_B,
        "P":               P,
        "X_C":             X_C,
        "C":               C,
        "N":               N,
        "aerobic_fraction": aero_frac,
        "env_stress":      env_stress,
        "carbon_seq_rate": carbon_seq_rate,   # → feeds PBPE Finance Engine
    }


# ─────────────────────────────────────────────
# 5. Quick smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate a 24-hour MBT55 fermentation cycle
    readings = [
        SoilSensorReading(timestamp_h=0,  pH=6.8, Eh_mV=200, moisture_pct=60,
                          temperature_C=25,  mbt55_dose_L_ha=5.0),
        SoilSensorReading(timestamp_h=4,  pH=6.5, Eh_mV=150, moisture_pct=62,
                          temperature_C=55),
        SoilSensorReading(timestamp_h=12, pH=6.9, Eh_mV=80,  moisture_pct=65,
                          temperature_C=85),
        SoilSensorReading(timestamp_h=24, pH=7.2, Eh_mV=120, moisture_pct=63,
                          temperature_C=90),
    ]

    result = run_soil_simulation(readings)

    print("=== AGRIX-OS :: Soil Dynamics (24h MBT55 Cycle) ===")
    print(f"  Final substrate depletion : {result['S'][-1]:.2f} g/L  "
          f"(from {result['S'][0]:.1f})")
    print(f"  Peak metabolite (M)       : {result['M'].max():.2f} g/L")
    print(f"  Humic product (P) final   : {result['P'][-1]:.2f} g/L")
    print(f"  Control biomass (X_C)     : {result['X_C'][-1]:.3f} g/L")
    print(f"  Soil organic carbon Δ     : "
          f"+{result['C'][-1] - result['C'][0]:.2f} g/L")
    print(f"  Carbon seq rate           : {result['carbon_seq_rate'][-1]:.2f} tCO₂/ha/year")
    print("  → Ready to feed PBPE Finance Engine ✓")
