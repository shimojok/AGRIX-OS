"""
AGRIX-OS :: dashboard/vitality_engine.py
=========================================
SafelyChain™ Vitality Score (V-Score) computation engine.

Feeds dashboard/app.py with:
  - V-Score  : ATP-decay-based metabolic vitality (0–100)
  - P_total  : 4-layer dynamic price decomposition
  - MetLife  : Metabolic Life countdown in days
  - Time-series for real-time simulation
"""

from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from safelychain_freshness import (
    HarvestPhenoprint, StorageConditions,
    predict_freshness, scfa_quality_index, FreshnessResult
)


# ─────────────────────────────────────────────────────────────────
# 1. Crop catalogue
# ─────────────────────────────────────────────────────────────────

CROP_CATALOGUE = {
    "coffee_cherry":   {"label": "Coffee Cherry 🍒",  "base_price_usd_kg": 2.80,  "chill_thr_C": 10.0},
    "strawberry":      {"label": "Strawberry 🍓",     "base_price_usd_kg": 4.20,  "chill_thr_C": 2.0},
    "carrot":          {"label": "Carrot 🥕",          "base_price_usd_kg": 0.90,  "chill_thr_C": -1.0},
    "eggplant":        {"label": "Eggplant 🍆",        "base_price_usd_kg": 1.60,  "chill_thr_C": 12.0},
    "chinese_cabbage": {"label": "Chinese Cabbage 🥬", "base_price_usd_kg": 0.75,  "chill_thr_C": 0.0},
    "tomato":          {"label": "Tomato 🍅",          "base_price_usd_kg": 1.40,  "chill_thr_C": 8.0},
}

MBT55_PRESETS = {
    "MBT55": dict(
        scfa_acetate_mmol_L=1.20, scfa_propionate_mmol_L=0.70,
        scfa_butyrate_mmol_L=0.28, antioxidant_density_umol_TE_g=26.0,
        ethylene_production_uL_kg_h=6.5, chilling_injury_score=0.30,
        she_score_at_harvest=0.78, humic_product_P=5.2, pre_harvest_temp_C=22.0,
    ),
    "Conventional": dict(
        scfa_acetate_mmol_L=0.20, scfa_propionate_mmol_L=0.10,
        scfa_butyrate_mmol_L=0.05, antioxidant_density_umol_TE_g=18.0,
        ethylene_production_uL_kg_h=12.0, chilling_injury_score=0.60,
        she_score_at_harvest=0.42, humic_product_P=1.5, pre_harvest_temp_C=22.0,
    ),
}


# ─────────────────────────────────────────────────────────────────
# 2. V-Score  — ATP-decay metabolic vitality
# ─────────────────────────────────────────────────────────────────

def compute_vscore(
    phenoprint: HarvestPhenoprint,
    elapsed_days: float,
    storage_temp_C: float = 4.0,
    storage_humidity_pct: float = 90.0,
) -> float:
    """
    Vitality Score (0–100): residual metabolic energy relative to harvest baseline.

    Model:
      ATP_t = ATP_0 × exp(-k_atp × t)
      k_atp depends on: storage temp, SCFA protection, antioxidant density, SHE score

    MBT55 crops: k_atp ~30% lower than conventional due to:
      - Higher SCFA → stronger membrane integrity
      - Higher antioxidant density → slower oxidative degradation
      - Higher SHE score → more complete nutrient loading at harvest
    """
    scfa_idx = scfa_quality_index(phenoprint)

    # Base ATP decay constant (per day): temp-sensitive Arrhenius-like
    # Reference: k_base ≈ 0.12/day at 20°C; halves per 10°C reduction (Q10≈2)
    temp_factor = 2.0 ** ((storage_temp_C - 20.0) / 10.0)   # <1 when cold
    k_base = 0.12 * temp_factor

    # MBT55 protection term — SCFA, antioxidants, SHE
    mbt55_protection = (
        0.40 * scfa_idx
        + 0.35 * min(phenoprint.antioxidant_density_umol_TE_g / 30.0, 1.0)
        + 0.25 * phenoprint.she_score_at_harvest
    )
    k_atp = k_base * (1.0 - 0.45 * mbt55_protection)  # max 45% reduction

    # Humidity modifier: desiccation stress accelerates ATP loss
    if storage_humidity_pct < 80:
        k_atp *= 1.0 + 0.02 * (80.0 - storage_humidity_pct)

    atp_ratio = np.exp(-k_atp * elapsed_days)
    return float(np.clip(atp_ratio * 100.0, 0.0, 100.0))


def vscore_timeseries(
    phenoprint: HarvestPhenoprint,
    max_days: float,
    storage_temp_C: float = 4.0,
    storage_humidity_pct: float = 90.0,
    n_points: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (days, v_scores) arrays for plotting."""
    days = np.linspace(0, max_days, n_points)
    scores = np.array([
        compute_vscore(phenoprint, d, storage_temp_C, storage_humidity_pct)
        for d in days
    ])
    return days, scores


# ─────────────────────────────────────────────────────────────────
# 3. Metabolic Life countdown
# ─────────────────────────────────────────────────────────────────

VSCORE_FUNCTIONAL_THRESHOLD = 35.0   # below this: functional compounds degraded

def metabolic_life_remaining(
    phenoprint: HarvestPhenoprint,
    elapsed_days: float,
    storage_temp_C: float = 4.0,
    storage_humidity_pct: float = 90.0,
) -> float:
    """
    Days remaining until V-Score drops below functional threshold (35).
    Returns 0.0 if already below threshold.
    """
    current_vscore = compute_vscore(
        phenoprint, elapsed_days, storage_temp_C, storage_humidity_pct
    )
    if current_vscore <= VSCORE_FUNCTIONAL_THRESHOLD:
        return 0.0

    # Binary search for crossing point
    lo, hi = elapsed_days, elapsed_days + 60.0
    for _ in range(40):
        mid = (lo + hi) / 2
        v   = compute_vscore(phenoprint, mid, storage_temp_C, storage_humidity_pct)
        if v > VSCORE_FUNCTIONAL_THRESHOLD:
            lo = mid
        else:
            hi = mid

    return max(0.0, round(hi - elapsed_days, 1))


# ─────────────────────────────────────────────────────────────────
# 4. P_total — 4-layer dynamic price decomposition
# ─────────────────────────────────────────────────────────────────

def compute_ptotal(
    phenoprint: HarvestPhenoprint,
    freshness: FreshnessResult,
    crop_type: str,
    elapsed_days: float = 0.0,
    storage_temp_C: float = 4.0,
    storage_humidity_pct: float = 90.0,
    carbon_price_usd_tCO2: float = 28.0,
    carbon_seq_rate_tCO2_ha_yr: float = 1.5,
    field_area_ha: float = 1.0,
    annual_yield_kg_ha: float = 2500.0,
) -> dict:
    """
    P_total = P_market + V(機能性) + L(ロス削減) + m(医療費削減) + C(炭素隔離)

    Returns dict with each layer value (USD/kg) and total.
    """
    crop = CROP_CATALOGUE.get(crop_type, CROP_CATALOGUE["coffee_cherry"])
    P_market = crop["base_price_usd_kg"]

    # ── V: Functional / Vitality premium ──────────────────────────
    # V-Score relative to conventional benchmark
    conv_pp  = HarvestPhenoprint(**MBT55_PRESETS["Conventional"])
    v_mbt55  = compute_vscore(phenoprint, elapsed_days, storage_temp_C, storage_humidity_pct)
    v_conv   = compute_vscore(conv_pp,    elapsed_days, storage_temp_C, storage_humidity_pct)
    v_delta  = max(0.0, (v_mbt55 - v_conv) / 100.0)          # 0–1
    antioxidant_premium = (
        phenoprint.antioxidant_density_umol_TE_g - 18.0
    ) / 18.0 * 0.30                                            # up to +$0.30/kg
    V_functional = P_market * v_delta * 0.35 + antioxidant_premium

    # ── L: Loss reduction premium ──────────────────────────────────
    # MBT55 loss fraction vs conventional
    conv_storage = StorageConditions(
        temperature_C=storage_temp_C,
        humidity_pct=storage_humidity_pct,
        duration_days=elapsed_days or 1.0,
    )
    freshness_conv = predict_freshness(conv_pp, conv_storage, crop_type)
    loss_reduction = max(0.0, freshness_conv.loss_fraction - freshness.loss_fraction)
    # FAO: post-harvest loss costs $400B/year; L premium = saved value
    L_loss_reduction = P_market * loss_reduction * 0.80  # 80% of saved value captured

    # ── m: Healthcare / nutrition premium ─────────────────────────
    # Antioxidant density linked to reduced chronic disease burden
    # Proxy: each 1 μmol TE/g above 18 baseline → $0.008/kg healthcare saving
    antioxidant_uplift = max(0.0, phenoprint.antioxidant_density_umol_TE_g - 18.0)
    m_healthcare = antioxidant_uplift * 0.008

    # ── C: Carbon sequestration layer ─────────────────────────────
    # Allocated per kg of produce from the farm
    if annual_yield_kg_ha > 0:
        carbon_value_per_ha_yr = carbon_seq_rate_tCO2_ha_yr * carbon_price_usd_tCO2
        C_carbon = carbon_value_per_ha_yr * field_area_ha / annual_yield_kg_ha
    else:
        C_carbon = 0.0

    P_total = P_market + V_functional + L_loss_reduction + m_healthcare + C_carbon

    return {
        "P_market":        round(P_market, 3),
        "V_functional":    round(max(0.0, V_functional), 3),
        "L_loss_reduction": round(max(0.0, L_loss_reduction), 3),
        "m_healthcare":    round(max(0.0, m_healthcare), 3),
        "C_carbon":        round(max(0.0, C_carbon), 3),
        "P_total":         round(P_total, 3),
        "premium_pct":     round((P_total - P_market) / P_market * 100, 1),
    }


# ─────────────────────────────────────────────────────────────────
# 5. Dashboard data bundle
# ─────────────────────────────────────────────────────────────────

def build_dashboard_data(
    crop_type: str = "coffee_cherry",
    treatment: str = "MBT55",          # "MBT55" or "Conventional"
    elapsed_days: float = 0.0,
    storage_temp_C: float = 4.0,
    storage_humidity_pct: float = 90.0,
    carbon_price_usd: float = 28.0,
) -> dict:
    """Single entry point for Dash callbacks."""
    preset = MBT55_PRESETS[treatment]
    pp = HarvestPhenoprint(**preset)

    storage = StorageConditions(
        temperature_C=storage_temp_C,
        humidity_pct=storage_humidity_pct,
        duration_days=max(elapsed_days, 1.0),
    )
    freshness = predict_freshness(pp, storage, crop_type)
    vscore    = compute_vscore(pp, elapsed_days, storage_temp_C, storage_humidity_pct)
    metlife   = metabolic_life_remaining(pp, elapsed_days, storage_temp_C, storage_humidity_pct)
    ptotal    = compute_ptotal(
        pp, freshness, crop_type, elapsed_days,
        storage_temp_C, storage_humidity_pct,
        carbon_price_usd_tCO2=carbon_price_usd,
    )

    # Timeseries — both treatments for comparison
    max_t = max(freshness.shelf_life_days * 1.3, 14.0)
    days_mbt55, vs_mbt55 = vscore_timeseries(
        HarvestPhenoprint(**MBT55_PRESETS["MBT55"]),
        max_t, storage_temp_C, storage_humidity_pct,
    )
    days_conv, vs_conv = vscore_timeseries(
        HarvestPhenoprint(**MBT55_PRESETS["Conventional"]),
        max_t, storage_temp_C, storage_humidity_pct,
    )

    return {
        "crop_type":    crop_type,
        "treatment":    treatment,
        "elapsed_days": elapsed_days,
        "vscore":       round(vscore, 1),
        "metlife_days": metlife,
        "freshness":    freshness,
        "ptotal":       ptotal,
        "timeseries": {
            "days":     days_mbt55.tolist(),
            "mbt55":    vs_mbt55.tolist(),
            "conv":     vs_conv.tolist(),
            "threshold": VSCORE_FUNCTIONAL_THRESHOLD,
        },
        "grade":        freshness.safelychain_grade,
        "shelf_life":   freshness.shelf_life_days,
        "loss_pct":     round(freshness.loss_fraction * 100, 1),
        "antioxidant":  round(freshness.antioxidant_retained_pct, 1),
        "scfa_index":   round(scfa_quality_index(pp) * 100, 1),
    }
