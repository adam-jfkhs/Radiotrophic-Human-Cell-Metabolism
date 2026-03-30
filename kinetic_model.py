"""
Radiotrophic Cell ODE Kinetic Model (Phase 3)
==============================================
Validates steady-state FBA predictions with dynamic simulation.
Models transient ROS kinetics during radiation exposure to determine:
  1. Maximum tolerable radiotrophic flux before transient ROS exceeds lethal threshold
  2. Time to steady-state ROS after radiation pulse onset
  3. Whether engineered defenses (Dsup, Mn-AOX, Nrf2) prevent transient lethality

Uses scipy ODE solver. Alternative to COPASI for reproducibility.

Kinetic parameters from literature:
  - SOD1 kcat: 2e9 M^-1 s^-1 (McCord & Fridovich 1969)
  - Catalase kcat: 4e7 s^-1, Km: 1.1 M (Ogura 1955)
  - GPX Km: 35 uM for H2O2 (Flohe 1971)
  - GR kcat: 200 s^-1 (Carlberg & Mannervik 1975)
  - Fenton: k = 76 M^-1 s^-1 (Walling 1975)
  - Water radiolysis G-values (Buxton et al. 1988)

Authors: Adam Labban
Date: March 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import os

# ============================================================
# KINETIC PARAMETERS (literature-sourced)
# ============================================================

# Enzyme concentrations (intracellular estimates for HEK293-like cells)
SOD_CONC = 10e-6        # 10 uM Cu/Zn-SOD1 (Crapo et al. 1992)
CAT_CONC = 1e-6         # 1 uM catalase (limited cytosolic, mainly peroxisomal)
GPX_CONC = 0.5e-6       # 0.5 uM GPX1 (selenium-dependent)
GR_CONC = 0.2e-6        # 0.2 uM glutathione reductase

# Michaelis-Menten parameters
SOD_KCAT = 2e9          # M^-1 s^-1 (effectively diffusion-limited)
CAT_KCAT = 4e7          # s^-1
CAT_KM = 1.1            # M (very high Km)
GPX_KCAT = 5e2          # s^-1
GPX_KM = 35e-6          # M (35 uM for H2O2)
GR_KCAT = 200           # s^-1
GR_KM = 65e-6           # M (for GSSG)

# Fenton chemistry
FENTON_K = 76           # M^-1 s^-1 (Fe2+ + H2O2 -> OH + OH-)
FE2_CONC = 1e-6         # 1 uM labile iron pool (Kakhlon & Bhatt 2002)

# Dsup parameters (Hashimoto et al. 2016)
DSUP_EFFICIENCY = 0.40  # 40% of OH radicals intercepted at chromatin

# Mn-antioxidant (Daly et al. 2004)
MN_CONC = 0.5e-3        # 0.5 mM (engineered expression level)
MN_K_H2O2 = 1e3         # M^-1 s^-1 (Mn2+ + H2O2 scavenging, Archibald & Fridovich 1982)
MN_K_O2S = 1e6          # M^-1 s^-1 (Mn2+ + O2•⁻, Barnese et al. 2012)

# Nrf2-enhanced glutathione recycling (Lewis et al. 2015)
NRF2_GR_BOOST = 2.0     # 2x GR activity under Nrf2 overexpression

# GSH pool
GSH_TOTAL = 5e-3        # 5 mM total glutathione (Meister 1988)

# Cellular thresholds
LETHAL_O2S = 50e-6      # 50 uM superoxide -> apoptosis trigger
LETHAL_OH = 1e-6        # 1 uM hydroxyl radical -> severe DNA damage
LETHAL_H2O2 = 100e-6    # 100 uM H2O2 -> oxidative stress threshold

# Radiotrophic NADH generation rate per unit flux (from Dadachova 2007)
# 120 nmol/min NADH reduction at 14 Gy/min -> ~8.6 nmol NADH per Gy
RADIO_NADH_PER_FLUX = 120e-9 / 60  # mol/s per unit flux (normalized)

# ROS generation per RADIO flux (grounded from G-values)
# G(O2•⁻) ≈ 0.28 per NADH, G(•OH) ≈ 0.24 per NADH (after melanin quenching)
ROS_O2S_PER_FLUX = 0.28 * RADIO_NADH_PER_FLUX
ROS_OH_PER_FLUX = 0.24 * RADIO_NADH_PER_FLUX


def michaelis_menten(vmax, km, substrate):
    """Standard Michaelis-Menten kinetics."""
    return vmax * substrate / (km + substrate)


def radiotrophic_ode(t, y, radio_flux, use_dsup=True, use_mn=True, use_nrf2=True,
                     radiation_on=True):
    """
    ODE system for transient ROS dynamics in a radiotrophic cell.

    State variables y = [O2S, H2O2, OH, GSH, GSSG, DNA_damage]
      O2S:        cytosolic superoxide (M)
      H2O2:       cytosolic hydrogen peroxide (M)
      OH:         hydroxyl radical (M)
      GSH:        reduced glutathione (M)
      GSSG:       oxidized glutathione (M)
      DNA_damage: accumulated DNA lesions (arbitrary units)
    """
    O2S, H2O2, OH, GSH, GSSG, DNA_dmg = y
    O2S = max(O2S, 0)
    H2O2 = max(H2O2, 0)
    OH = max(OH, 0)
    GSH = max(GSH, 0)
    GSSG = max(GSSG, 0)
    DNA_dmg = max(DNA_dmg, 0)

    # --- ROS PRODUCTION ---
    # Radiotrophic pathway (only when radiation is on)
    if radiation_on:
        radio_o2s = radio_flux * ROS_O2S_PER_FLUX
        radio_oh = radio_flux * ROS_OH_PER_FLUX
    else:
        radio_o2s = 0
        radio_oh = 0

    # Basal mitochondrial ETC leak (~1-2% of electron flow)
    basal_o2s = 0.5e-9  # ~0.5 nM/s basal superoxide production

    # --- NATIVE DEFENSES ---
    # SOD: O2S -> H2O2 (diffusion-limited, pseudo-first-order at physiological [SOD])
    v_sod = SOD_KCAT * SOD_CONC * O2S

    # Catalase: 2 H2O2 -> 2 H2O + O2
    v_cat = michaelis_menten(CAT_KCAT * CAT_CONC, CAT_KM, H2O2)

    # GPX: H2O2 + 2 GSH -> 2 H2O + GSSG
    v_gpx = michaelis_menten(GPX_KCAT * GPX_CONC, GPX_KM, H2O2) * (GSH / (GSH + 1e-4))

    # GR: GSSG + NADH -> 2 GSH + NAD+ (NADH assumed non-limiting for radiotrophic cell)
    gr_boost = NRF2_GR_BOOST if use_nrf2 else 1.0
    v_gr = gr_boost * michaelis_menten(GR_KCAT * GR_CONC, GR_KM, GSSG)

    # --- ENGINEERED DEFENSES ---
    # Mn-antioxidant complex (Daly et al. 2004)
    if use_mn:
        v_mn_o2s = MN_K_O2S * MN_CONC * O2S
        v_mn_h2o2 = MN_K_H2O2 * MN_CONC * H2O2
    else:
        v_mn_o2s = 0
        v_mn_h2o2 = 0

    # Dsup: intercepts OH radicals at chromatin (40% efficiency)
    if use_dsup:
        v_dsup = DSUP_EFFICIENCY * OH / (OH + 1e-9)  # saturating at low [OH]
        # Scale to be fast enough to intercept 40% before Fenton damage
        v_dsup = v_dsup * 1e6 * OH
    else:
        v_dsup = 0

    # --- DAMAGE PATHWAYS ---
    # Fenton: H2O2 + Fe2+ -> OH + OH- (generates hydroxyl radicals)
    v_fenton = FENTON_K * FE2_CONC * H2O2

    # OH radical -> DNA damage (pseudo-first-order, very fast)
    v_oh_damage = 1e9 * OH  # OH attacks DNA at near-diffusion limit

    # GSH scavenging of OH radicals
    v_oh_gsh = 1e10 * GSH * OH  # GSH + OH -> GS• + H2O (Buxton et al.)

    # DNA repair (BER, slow enzymatic process)
    v_repair = 0.01 * DNA_dmg  # first-order repair, ~100s half-life

    # --- DIFFERENTIAL EQUATIONS ---
    dO2S = (radio_o2s + basal_o2s
            - v_sod
            - v_mn_o2s)

    dH2O2 = (0.5 * v_sod       # SOD produces 1 H2O2 per 2 O2S
             + v_fenton * 0     # Fenton consumes H2O2 (accounted below)
             - v_cat
             - v_gpx
             - v_mn_h2o2
             - v_fenton)

    dOH = (radio_oh
           + v_fenton           # Fenton produces OH
           - v_dsup
           - v_oh_gsh
           - v_oh_damage)

    # GSH/GSSG with pool conservation: GSH + 2*GSSG ≈ constant
    total_gsh_equiv = GSH + 2 * GSSG
    pool_excess = total_gsh_equiv - GSH_TOTAL
    pool_correction = 0.1 * pool_excess  # gently enforce conservation

    dGSH = (2 * v_gr            # GR regenerates 2 GSH per GSSG
            - 2 * v_gpx         # GPX uses 2 GSH per H2O2
            - v_oh_gsh          # OH scavenging uses GSH
            - pool_correction)  # conservation enforcement

    dGSSG = (v_gpx              # GPX produces GSSG
             + 0.5 * v_oh_gsh   # OH scavenging produces 0.5 GSSG
             - v_gr)            # GR consumes GSSG

    dDNA_dmg = v_oh_damage - v_repair

    return [dO2S, dH2O2, dOH, dGSH, dGSSG, dDNA_dmg]


def run_simulation(radio_flux, duration=300, use_dsup=True, use_mn=True,
                   use_nrf2=True, pulse_off_time=None):
    """
    Run a single kinetic simulation.

    Args:
        radio_flux: radiotrophic flux intensity (model units)
        duration: simulation time in seconds
        use_dsup: enable Dsup protein
        use_mn: enable Mn-antioxidant
        use_nrf2: enable Nrf2-enhanced GR
        pulse_off_time: if set, radiation turns off at this time (seconds)

    Returns:
        DataFrame with time-course of all species
    """
    # Initial conditions: low basal ROS, full GSH pool, no damage
    y0 = [
        1e-9,       # O2S: 1 nM basal
        1e-7,       # H2O2: 100 nM basal
        1e-12,      # OH: ~0 (very reactive, never accumulates)
        GSH_TOTAL * 0.95,  # GSH: 95% of pool reduced
        GSH_TOTAL * 0.05,  # GSSG: 5% oxidized
        0.0         # DNA damage: none
    ]

    def ode_wrapper(t, y):
        rad_on = True
        if pulse_off_time is not None and t > pulse_off_time:
            rad_on = False
        return radiotrophic_ode(t, y, radio_flux, use_dsup, use_mn, use_nrf2, rad_on)

    t_eval = np.linspace(0, duration, 1000)
    sol = solve_ivp(ode_wrapper, [0, duration], y0, t_eval=t_eval,
                    method='LSODA', rtol=1e-8, atol=1e-12,
                    max_step=0.1)

    if not sol.success:
        print(f"Warning: ODE solver failed: {sol.message}")

    df = pd.DataFrame({
        'time_s': sol.t,
        'superoxide_M': np.maximum(sol.y[0], 0),
        'h2o2_M': np.maximum(sol.y[1], 0),
        'oh_radical_M': np.maximum(sol.y[2], 0),
        'gsh_M': np.maximum(sol.y[3], 0),
        'gssg_M': np.maximum(sol.y[4], 0),
        'dna_damage': np.maximum(sol.y[5], 0)
    })
    return df


def run_all_kinetic_experiments():
    """Run all Phase 3 kinetic experiments."""

    results = {}

    # ----------------------------------------------------------
    # EXPERIMENT K1: Radiation onset transient (all defenses ON)
    # Shows ROS spike when radiation starts, time to steady-state
    # ----------------------------------------------------------
    print("  K1: Radiation onset transient...")
    df = run_simulation(radio_flux=25, duration=300,
                        use_dsup=True, use_mn=True, use_nrf2=True)
    results['k1_onset_transient'] = df

    # ----------------------------------------------------------
    # EXPERIMENT K2: Dose-response (max ROS vs flux)
    # Find maximum tolerable flux before lethal ROS threshold
    # ----------------------------------------------------------
    print("  K2: Dose-response kinetics...")
    rows = []
    for flux in [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]:
        df = run_simulation(radio_flux=flux, duration=120,
                            use_dsup=True, use_mn=True, use_nrf2=True)
        peak_o2s = df['superoxide_M'].max()
        peak_h2o2 = df['h2o2_M'].max()
        peak_oh = df['oh_radical_M'].max()
        ss_o2s = df['superoxide_M'].iloc[-1]
        ss_h2o2 = df['h2o2_M'].iloc[-1]
        final_damage = df['dna_damage'].iloc[-1]
        rows.append({
            'flux': flux,
            'peak_superoxide_uM': round(peak_o2s * 1e6, 4),
            'peak_h2o2_uM': round(peak_h2o2 * 1e6, 4),
            'peak_oh_nM': round(peak_oh * 1e9, 4),
            'steady_superoxide_uM': round(ss_o2s * 1e6, 4),
            'steady_h2o2_uM': round(ss_h2o2 * 1e6, 4),
            'dna_damage_120s': round(final_damage, 6),
            'superoxide_lethal': 'YES' if peak_o2s > LETHAL_O2S else 'no',
            'h2o2_lethal': 'YES' if peak_h2o2 > LETHAL_H2O2 else 'no'
        })
    results['k2_dose_response'] = pd.DataFrame(rows)

    # ----------------------------------------------------------
    # EXPERIMENT K3: Defense ablation kinetics
    # Compare transient ROS with each defense knocked out
    # ----------------------------------------------------------
    print("  K3: Defense ablation kinetics...")
    configs = [
        ('All defenses',    True, True, True),
        ('No Dsup',         False, True, True),
        ('No Mn-AOX',       True, False, True),
        ('No Nrf2',         True, True, False),
        ('No Dsup+Mn-AOX',  False, False, True),
        ('Native only',     False, False, False),
    ]
    rows = []
    for label, dsup, mn, nrf2 in configs:
        df = run_simulation(radio_flux=25, duration=120,
                            use_dsup=dsup, use_mn=mn, use_nrf2=nrf2)
        rows.append({
            'config': label,
            'peak_superoxide_uM': round(df['superoxide_M'].max() * 1e6, 4),
            'peak_h2o2_uM': round(df['h2o2_M'].max() * 1e6, 4),
            'peak_oh_nM': round(df['oh_radical_M'].max() * 1e9, 4),
            'dna_damage_120s': round(df['dna_damage'].iloc[-1], 6),
            'gsh_depletion_pct': round((1 - df['gsh_M'].iloc[-1] / GSH_TOTAL) * 100, 1)
        })
    results['k3_ablation_kinetics'] = pd.DataFrame(rows)

    # ----------------------------------------------------------
    # EXPERIMENT K4: Radiation pulse (on for 60s, then off)
    # Shows ROS recovery dynamics after radiation stops
    # ----------------------------------------------------------
    print("  K4: Radiation pulse recovery...")
    df = run_simulation(radio_flux=25, duration=300,
                        use_dsup=True, use_mn=True, use_nrf2=True,
                        pulse_off_time=60)
    results['k4_pulse_recovery'] = df

    return results


if __name__ == '__main__':
    print("Running Phase 3: ODE Kinetic Model")
    print("=" * 60)

    results = run_all_kinetic_experiments()

    for name, df in results.items():
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {name.upper()}")
        print(f"{'='*60}")
        if len(df) <= 20:
            print(df.to_string(index=False))
        else:
            print(f"  Time series: {len(df)} points, {df['time_s'].iloc[-1]:.0f}s duration")
            print(f"  Peak superoxide: {df['superoxide_M'].max()*1e6:.4f} uM")
            print(f"  Peak H2O2:      {df['h2o2_M'].max()*1e6:.4f} uM")
            print(f"  Peak OH:        {df['oh_radical_M'].max()*1e9:.4f} nM")
            print(f"  Final DNA damage: {df['dna_damage'].iloc[-1]:.6f}")
            print(f"  Final GSH:      {df['gsh_M'].iloc[-1]*1e3:.2f} mM ({df['gsh_M'].iloc[-1]/GSH_TOTAL*100:.1f}%)")

    # Save results
    for name, df in results.items():
        df.to_csv(f'{name}.csv', index=False)

    print("\n\nKinetic results saved to current directory")
