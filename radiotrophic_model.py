"""
Radiotrophic Human Cell Metabolic Model
========================================
Computational feasibility analysis of melanin-based radiotrophic
metabolism in human cells through cross-species genetic engineering.

Authors: Adam Labban
Date: March 2026

Model: Custom constraint-based metabolic model (50 reactions, 50 metabolites)
Built with COBRApy. Includes glycolysis, TCA cycle, electron transport chain,
melanin synthesis, radiotrophic NADH generation, and four ROS defense systems
(native SOD/catalase/GPX + engineered Dsup/Mn-AOX/Nrf2).

Requirements:
    pip install cobra pandas matplotlib
"""

import cobra
import pandas as pd
import json
import os

# ============================================================
# MODEL CONSTRUCTION
# ============================================================

def build_model():
    """Build the radiotrophic human cell metabolic model."""
    
    model = cobra.Model('radiotrophic_human_cell')
    M = {}
    
    def m(mid, name, comp='c'):
        met = cobra.Metabolite(mid, name=name, compartment=comp)
        M[mid] = met
    
    # --- Cytosolic metabolites ---
    for mid, name in [
        ('glc_c','Glucose'), ('pyr_c','Pyruvate'), ('lac_c','Lactate'),
        ('atp_c','ATP'), ('adp_c','ADP'), ('nad_c','NAD+'), ('nadh_c','NADH'),
        ('h_c','H+'), ('pi_c','Phosphate'), ('h2o_c','Water'), ('co2_c','CO2'),
        ('o2_c','O2'), ('o2s_c','Superoxide'), ('h2o2_c','H2O2'),
        ('melanin_c','Melanin'), ('tyr_c','Tyrosine'), ('dopa_c','L-DOPA'),
        ('gthrd_c','GSH'), ('gthox_c','GSSG')
    ]:
        m(mid, name)
    
    # --- Mitochondrial metabolites ---
    for mid, name in [
        ('pyr_m','Pyruvate'), ('accoa_m','Acetyl-CoA'), ('coa_m','CoA'),
        ('oaa_m','Oxaloacetate'), ('cit_m','Citrate'), ('akg_m','Alpha-ketoglutarate'),
        ('succoa_m','Succinyl-CoA'), ('succ_m','Succinate'),
        ('fum_m','Fumarate'), ('mal_m','Malate'),
        ('nad_m','NAD+'), ('nadh_m','NADH'), ('fad_m','FAD'), ('fadh2_m','FADH2'),
        ('atp_m','ATP'), ('adp_m','ADP'), ('h_m','H+'), ('co2_m','CO2'),
        ('h2o_m','Water'), ('o2_m','O2'), ('pi_m','Phosphate'),
        ('o2s_m','Superoxide'), ('h2o2_m','H2O2')
    ]:
        m(mid, name, 'm')
    
    # --- Extracellular metabolites ---
    for mid, name in [
        ('glc_e','Glucose'), ('o2_e','O2'), ('co2_e','CO2'),
        ('h2o_e','Water'), ('lac_e','Lactate'), ('h_e','H+'),
        ('pi_e','Phosphate'), ('tyr_e','Tyrosine')
    ]:
        m(mid, name, 'e')
    
    # --- Helper to add reactions ---
    rxns = []
    def R(rid, name, md, bounds=(0, 1000)):
        r = cobra.Reaction(rid)
        r.name = name
        r.lower_bound, r.upper_bound = bounds
        r.add_metabolites({M[k]: v for k, v in md.items()})
        rxns.append(r)
    
    # ============================================================
    # TRANSPORT REACTIONS
    # ============================================================
    R('GLCt',  'Glucose transport',          {'glc_e':-1, 'glc_c':1})
    R('O2t',   'O2 to cytosol',             {'o2_e':-1, 'o2_c':1})
    R('O2tm',  'O2 to mitochondria',        {'o2_c':-1, 'o2_m':1}, (-1000,1000))
    R('CO2tm', 'CO2 from mitochondria',     {'co2_m':-1, 'co2_c':1}, (-1000,1000))
    R('CO2t',  'CO2 export',               {'co2_c':-1, 'co2_e':1})
    R('H2Ot',  'H2O transport',            {'h2o_e':-1, 'h2o_c':1}, (-1000,1000))
    R('H2Otm', 'H2O to mitochondria',      {'h2o_c':-1, 'h2o_m':1}, (-1000,1000))
    R('Ht',    'H+ export to extracellular', {'h_c':-1, 'h_e':1}, (-1000,1000))
    R('Htm',   'H+ to mitochondria',        {'h_c':-1, 'h_m':1}, (-1000,1000))
    R('LACt',  'Lactate export',            {'lac_c':-1, 'lac_e':1})
    R('PYRtm', 'Pyruvate to mitochondria',  {'pyr_c':-1, 'pyr_m':1})
    R('TYRt',  'Tyrosine transport',        {'tyr_e':-1, 'tyr_c':1})
    R('PIt',   'Phosphate transport',       {'pi_e':-1, 'pi_c':1})
    R('PItm',  'Phosphate to mitochondria', {'pi_c':-1, 'pi_m':1}, (-1000,1000))
    R('ATPtm', 'ATP/ADP translocase',       {'atp_m':-1, 'adp_c':-1, 'atp_c':1, 'adp_m':1}, (-1000,1000))
    R('NADHtm','NADH shuttle (malate-asp)', {'nadh_c':-1, 'nad_c':1, 'nadh_m':1, 'nad_m':-1})
    R('H2O2tm','H2O2 from mitochondria',    {'h2o2_m':-1, 'h2o2_c':1})
    R('O2Stm', 'Superoxide from mitochondria', {'o2s_m':-1, 'o2s_c':1})
    
    # ============================================================
    # CORE METABOLISM
    # ============================================================
    
    # Glycolysis (lumped): Glucose -> 2 Pyruvate + 2 ATP + 2 NADH
    R('GLYC', 'Glycolysis (lumped)', {
        'glc_c':-1, 'nad_c':-2, 'adp_c':-2, 'pi_c':-2,
        'pyr_c':2, 'nadh_c':2, 'atp_c':2, 'h2o_c':2, 'h_c':2
    })
    
    # Lactate dehydrogenase (reversible)
    R('LDH', 'Lactate dehydrogenase', {
        'pyr_c':-1, 'nadh_c':-1, 'h_c':-1, 'lac_c':1, 'nad_c':1
    }, (-1000, 1000))
    
    # Pyruvate dehydrogenase
    R('PDH', 'Pyruvate dehydrogenase', {
        'pyr_m':-1, 'nad_m':-1, 'coa_m':-1,
        'accoa_m':1, 'nadh_m':1, 'co2_m':1
    })
    
    # TCA Cycle
    R('CS',     'Citrate synthase',          {'accoa_m':-1, 'oaa_m':-1, 'h2o_m':-1, 'cit_m':1, 'coa_m':1})
    R('ACONIDH','Aconitase + Isocitrate DH', {'cit_m':-1, 'nad_m':-1, 'akg_m':1, 'nadh_m':1, 'co2_m':1})
    R('AKGDH',  'Alpha-KG dehydrogenase',    {'akg_m':-1, 'nad_m':-1, 'coa_m':-1, 'succoa_m':1, 'nadh_m':1, 'co2_m':1})
    R('SCS',    'Succinyl-CoA synthetase',   {'succoa_m':-1, 'adp_m':-1, 'pi_m':-1, 'succ_m':1, 'coa_m':1, 'atp_m':1})
    R('SDH',    'Succinate dehydrogenase',   {'succ_m':-1, 'fad_m':-1, 'fum_m':1, 'fadh2_m':1})
    R('FUM',    'Fumarase',                  {'fum_m':-1, 'h2o_m':-1, 'mal_m':1}, (-1000,1000))
    R('MDH',    'Malate dehydrogenase',      {'mal_m':-1, 'nad_m':-1, 'oaa_m':1, 'nadh_m':1, 'h_m':1}, (-1000,1000))
    
    # Electron Transport Chain (lumped with ROS byproduct)
    # NADH -> 2.5 ATP (P/O ratio), 1% electron leak to superoxide
    R('ETC_N', 'ETC Complex I-IV (NADH)', {
        'nadh_m':-1, 'o2_m':-0.5, 'adp_m':-2.5, 'pi_m':-2.5,
        'nad_m':1, 'h2o_m':0.5, 'atp_m':2.5, 'o2s_m':0.01
    })
    # FADH2 -> 1.5 ATP
    R('ETC_F', 'ETC Complex II-IV (FADH2)', {
        'fadh2_m':-1, 'o2_m':-0.5, 'adp_m':-1.5, 'pi_m':-1.5,
        'fad_m':1, 'h2o_m':0.5, 'atp_m':1.5, 'o2s_m':0.005
    })
    
    # ============================================================
    # MELANIN SYNTHESIS
    # ============================================================
    R('TYROX',  'Tyrosinase (Tyr -> DOPA)',  {'tyr_c':-1, 'o2_c':-0.5, 'dopa_c':1, 'h2o_c':0.5})
    R('MELSYN', 'Melanin synthesis (lumped)', {'dopa_c':-4, 'o2_c':-2, 'melanin_c':1, 'h2o_c':4, 'h2o2_c':1})
    
    # ============================================================
    # RADIOTROPHIC METABOLISM (novel engineered pathway)
    # Based on Dadachova et al. (2007) - Cryptococcus neoformans
    # Melanin absorbs ionizing radiation, energizes electrons,
    # drives NAD+ -> NADH with superoxide as byproduct
    # ============================================================
    R('RADIO', 'Melanin radiotrophic NADH generation', {
        'nad_c':-1, 'h_c':-1,
        'nadh_c':1, 'o2s_c':0.3   # ROS cost coefficient - TODO: validate from literature
    }, (0, 50))
    
    # ============================================================
    # ROS DEFENSE - NATIVE HUMAN
    # ============================================================
    R('SODc', 'Superoxide dismutase (cytosolic)',  {'o2s_c':-2, 'h_c':-2, 'h2o2_c':1, 'o2_c':1})
    R('SODm', 'Superoxide dismutase (mito)',       {'o2s_m':-2, 'h_m':-2, 'h2o2_m':1, 'o2_m':1})
    R('CATc', 'Catalase (cytosolic)',              {'h2o2_c':-2, 'h2o_c':2, 'o2_c':1})
    R('CATm', 'Catalase (mitochondrial)',          {'h2o2_m':-2, 'h2o_m':2, 'o2_m':1})
    R('GPX',  'Glutathione peroxidase',            {'h2o2_c':-1, 'gthrd_c':-2, 'h2o_c':2, 'gthox_c':1})
    R('GR',   'Glutathione reductase',             {'gthox_c':-1, 'nadh_c':-1, 'h_c':-1, 'gthrd_c':2, 'nad_c':1})
    
    # ============================================================
    # ROS DEFENSE - CROSS-SPECIES ENGINEERED
    # ============================================================
    
    # Tardigrade Dsup protein - directly shields DNA from hydroxyl radicals
    # Source: Hashimoto et al. (2016) Nature Communications
    # Demonstrated 40% reduction in radiation-induced DNA damage in human cells
    R('DSUP', 'Tardigrade Dsup DNA protection', {
        'o2s_c':-2, 'h_c':-2,
        'h2o2_c':0.5, 'o2_c':0.5, 'h2o_c':0.5  # More efficient than SOD
    })
    
    # Deinococcus radiodurans Mn-antioxidant complex
    # Source: Daly et al. (2004) Science
    # Manganese complexes protect proteins from oxidative damage
    R('MNAOX', 'Deinococcus Mn-antioxidant complex', {
        'h2o2_c':-2, 'h2o_c':2, 'o2_c':1
    })
    
    # Naked mole rat enhanced Nrf2 pathway
    # Source: Lewis et al. (2015) PNAS
    # Constitutively elevated Nrf2 drives superior antioxidant gene expression
    R('NRF2', 'Enhanced Nrf2 glutathione recycling', {
        'gthox_c':-1, 'nadh_c':-0.5, 'h_c':-0.5,
        'gthrd_c':2, 'nad_c':0.5  # More efficient than normal GR
    })
    
    # ============================================================
    # EXCHANGE REACTIONS
    # ============================================================
    R('EX_glc', 'Glucose uptake',    {'glc_e':-1}, (-10, 0))
    R('EX_o2',  'O2 uptake',         {'o2_e':-1},  (-20, 0))
    R('EX_co2', 'CO2 secretion',     {'co2_e':-1}, (0, 1000))
    R('EX_h2o', 'H2O exchange',      {'h2o_e':-1}, (-1000, 1000))
    R('EX_lac', 'Lactate secretion',  {'lac_e':-1}, (0, 1000))
    R('EX_h',   'H+ exchange',       {'h_e':-1},   (-1000, 1000))
    R('EX_pi',  'Phosphate uptake',  {'pi_e':-1},  (-100, 0))
    R('EX_tyr', 'Tyrosine uptake',   {'tyr_e':-1}, (-5, 0))
    
    # ============================================================
    # OBJECTIVE: ATP MAINTENANCE
    # ============================================================
    R('ATPM', 'ATP maintenance demand', {
        'atp_c':-1, 'h2o_c':-1, 'adp_c':1, 'pi_c':1, 'h_c':1
    }, (0, 1000))
    
    model.add_reactions(rxns)
    model.objective = 'ATPM'
    
    return model


def disable_engineered(model):
    """Disable all engineered pathways to simulate a normal human cell."""
    model.reactions.get_by_id('RADIO').upper_bound = 0
    model.reactions.get_by_id('DSUP').upper_bound = 0
    model.reactions.get_by_id('MNAOX').upper_bound = 0
    model.reactions.get_by_id('NRF2').upper_bound = 0


# ============================================================
# EXPERIMENTS
# ============================================================

def run_all_experiments():
    """Run all experiments and return results as DataFrames."""
    
    model = build_model()
    results = {}
    
    # ----------------------------------------------------------
    # EXPERIMENT 1: Glucose restriction
    # ----------------------------------------------------------
    rows = []
    for glc in [10, 8, 6, 5, 4, 3, 2, 1, 0.5, 0]:
        # Normal cell
        with model:
            model.reactions.get_by_id('EX_glc').lower_bound = -glc
            disable_engineered(model)
            s1 = model.optimize()
            atp_normal = s1.objective_value if s1.status == 'optimal' else 0
        
        # Radiotrophic cell
        with model:
            model.reactions.get_by_id('EX_glc').lower_bound = -glc
            s2 = model.optimize()
            atp_radio = s2.objective_value if s2.status == 'optimal' else 0
            radio_flux = s2.fluxes.get('RADIO', 0) if s2.status == 'optimal' else 0
            sod_flux = s2.fluxes.get('SODc', 0) if s2.status == 'optimal' else 0
            dsup_flux = s2.fluxes.get('DSUP', 0) if s2.status == 'optimal' else 0
        
        gain = atp_radio - atp_normal
        pct = (gain / atp_normal * 100) if atp_normal > 0 else (100 if atp_radio > 0 else 0)
        
        rows.append({
            'glucose': glc, 'atp_normal': round(atp_normal, 2),
            'atp_radio': round(atp_radio, 2), 'gain': round(gain, 2),
            'pct_boost': round(pct, 1), 'radio_flux': round(radio_flux, 2),
            'sod_flux': round(sod_flux, 2), 'dsup_flux': round(dsup_flux, 2)
        })
    
    results['glucose_restriction'] = pd.DataFrame(rows)
    
    # ----------------------------------------------------------
    # EXPERIMENT 2: Hypoxia (glucose = 5)
    # ----------------------------------------------------------
    rows = []
    for o2 in [20, 10, 5, 3, 2, 1, 0.5, 0]:
        with model:
            model.reactions.get_by_id('EX_glc').lower_bound = -5
            model.reactions.get_by_id('EX_o2').lower_bound = -o2
            disable_engineered(model)
            s1 = model.optimize()
            atp_normal = s1.objective_value if s1.status == 'optimal' else 0
        
        with model:
            model.reactions.get_by_id('EX_glc').lower_bound = -5
            model.reactions.get_by_id('EX_o2').lower_bound = -o2
            s2 = model.optimize()
            atp_radio = s2.objective_value if s2.status == 'optimal' else 0
            radio_flux = s2.fluxes.get('RADIO', 0) if s2.status == 'optimal' else 0
        
        pct = ((atp_radio - atp_normal) / atp_normal * 100) if atp_normal > 0 else 0
        rows.append({
            'o2': o2, 'atp_normal': round(atp_normal, 2),
            'atp_radio': round(atp_radio, 2),
            'pct_boost': round(pct, 1), 'radio_flux': round(radio_flux, 2)
        })
    
    results['hypoxia'] = pd.DataFrame(rows)
    
    # ----------------------------------------------------------
    # EXPERIMENT 3: Radiation dose-response (glucose = 5)
    # ----------------------------------------------------------
    rows = []
    for maxrad in [0, 2, 5, 10, 15, 20, 25, 30, 40, 50]:
        with model:
            model.reactions.get_by_id('EX_glc').lower_bound = -5
            model.reactions.get_by_id('RADIO').upper_bound = maxrad
            sol = model.optimize()
            if sol.status == 'optimal':
                rows.append({
                    'max_radio': maxrad,
                    'atp': round(sol.objective_value, 2),
                    'radio_used': round(sol.fluxes['RADIO'], 2),
                    'sod': round(sol.fluxes['SODc'], 2),
                    'dsup': round(sol.fluxes['DSUP'], 2),
                    'catalase': round(sol.fluxes['CATc'], 2),
                    'mn_aox': round(sol.fluxes['MNAOX'], 2),
                    'ros_generated': round(sol.fluxes['RADIO'] * 0.3, 2)
                })
    
    results['dose_response'] = pd.DataFrame(rows)
    
    # ----------------------------------------------------------
    # EXPERIMENT 4: Defense system ablation (glucose=5, radio=50)
    # ----------------------------------------------------------
    configs = [
        ("All defenses ON",      {'RADIO':50,'DSUP':1000,'MNAOX':1000,'NRF2':1000,'SODc':1000,'CATc':1000,'GPX':1000,'GR':1000}),
        ("No Dsup",              {'RADIO':50,'DSUP':0,   'MNAOX':1000,'NRF2':1000,'SODc':1000,'CATc':1000,'GPX':1000,'GR':1000}),
        ("No Mn-AOX",            {'RADIO':50,'DSUP':1000,'MNAOX':0,   'NRF2':1000,'SODc':1000,'CATc':1000,'GPX':1000,'GR':1000}),
        ("No Nrf2",              {'RADIO':50,'DSUP':1000,'MNAOX':1000,'NRF2':0,   'SODc':1000,'CATc':1000,'GPX':1000,'GR':1000}),
        ("No SOD",               {'RADIO':50,'DSUP':1000,'MNAOX':1000,'NRF2':1000,'SODc':0,   'CATc':1000,'GPX':1000,'GR':1000}),
        ("No Catalase",          {'RADIO':50,'DSUP':1000,'MNAOX':1000,'NRF2':1000,'SODc':1000,'CATc':0,   'GPX':1000,'GR':1000}),
        ("Only native defenses", {'RADIO':50,'DSUP':0,   'MNAOX':0,   'NRF2':0,   'SODc':1000,'CATc':1000,'GPX':1000,'GR':1000}),
        ("Only engineered",      {'RADIO':50,'DSUP':1000,'MNAOX':1000,'NRF2':1000,'SODc':0,   'CATc':0,   'GPX':0,   'GR':0}),
        ("No defenses",          {'RADIO':50,'DSUP':0,   'MNAOX':0,   'NRF2':0,   'SODc':0,   'CATc':0,   'GPX':0,   'GR':0}),
    ]
    
    rows = []
    for label, cfg in configs:
        with model:
            model.reactions.get_by_id('EX_glc').lower_bound = -5
            for rid, ub in cfg.items():
                model.reactions.get_by_id(rid).upper_bound = ub
            sol = model.optimize()
            atp = sol.objective_value if sol.status == 'optimal' else 0
            rf = sol.fluxes.get('RADIO', 0) if sol.status == 'optimal' else 0
            rows.append({'config': label, 'atp': round(atp, 2), 'radio_flux': round(rf, 2), 'status': sol.status})
    
    results['ablation'] = pd.DataFrame(rows)
    
    # ----------------------------------------------------------
    # EXPERIMENT 5: Combined stress
    # ----------------------------------------------------------
    rows = []
    for glc, o2 in [(5,20),(5,5),(2,10),(2,5),(1,5),(1,2),(0.5,1),(0,5),(0,1)]:
        with model:
            model.reactions.get_by_id('EX_glc').lower_bound = -glc
            model.reactions.get_by_id('EX_o2').lower_bound = -o2
            disable_engineered(model)
            s1 = model.optimize()
            a1 = s1.objective_value if s1.status == 'optimal' else 0
        with model:
            model.reactions.get_by_id('EX_glc').lower_bound = -glc
            model.reactions.get_by_id('EX_o2').lower_bound = -o2
            s2 = model.optimize()
            a2 = s2.objective_value if s2.status == 'optimal' else 0
        pct = ((a2 - a1) / a1 * 100) if a1 > 0 else (100 if a2 > 0 else 0)
        rows.append({'glucose': glc, 'o2': o2, 'atp_normal': round(a1,2), 'atp_radio': round(a2,2), 'pct_boost': round(pct,1)})
    
    results['combined_stress'] = pd.DataFrame(rows)
    
    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("Building radiotrophic cell model...")
    model = build_model()
    print(f"Model: {len(model.reactions)} reactions, {len(model.metabolites)} metabolites\n")
    
    print("Running all experiments...")
    results = run_all_experiments()
    
    for name, df in results.items():
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {name.upper()}")
        print(f"{'='*60}")
        print(df.to_string(index=False))
    
    # Save results
    os.makedirs('results', exist_ok=True)
    for name, df in results.items():
        df.to_csv(f'results/{name}.csv', index=False)
    
    # Save model
    cobra.io.save_json_model(model, 'radiotrophic_cell_model.json')
    
    print("\n\nResults saved to results/ directory")
    print("Model saved to radiotrophic_cell_model.json")
