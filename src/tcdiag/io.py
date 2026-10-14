"""Loading, cleaning and block-averaging of the Marine Engine Fault dataset (Zenodo 10.5281/zenodo.19857425).

Unit corrections (found by reproducing the dataset's own derived channels, see scripts/independent_checks.py):
  * 'Fuel Flow' is kg/h, not m3/h: Total Input Energy = 11.6 x Fuel Flow exactly, i.e. 41.76 MJ/kg x (kg/h) / 3600.
  * 'Total Input Energy' is kW, not W (Shaft Power / Total Input Energy = Effective Efficiency only in kW).
  * 'Exh. Gas Mass Flow' is not measured: it is a speed-density estimate, 0.0015408 x N x p_receiver / T_receiver
    (coefficient of variation 0.5 % over the reference run), so it carries no independent flow information.
Pressures: 'Charge Air Press.' is gauge in kgf/cm2 (receiver, after the cooler); filter loss and turbine back
pressure are Pa relative to ambient. Ambient pressure was not logged; P_AMB is assumed (sensitivity in the analysis).
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / 'data' / 'Marine_Engine_Fault_Data'
KGF_CM2 = 98.0665e3          # Pa
P_AMB = 101.325e3            # Pa, assumed
LHV = 41.76e6                # J/kg, implied by the dataset's Total Input Energy channel
SENTINEL = 999.0             # logger overflow value seen in the exhaust temperature channels

# short names used in the code
COLS = {
    'Engine Speed': 'N', 'Shaft Power': 'P', 'Fuel Flow': 'fuel_kgh',
    'Charge Air Press.': 'p_rec_g', 'Compressor Filter Loss': 'dp_f', 'Turbine Back Pressure': 'dp_ex',
    'Exh.Gas Temp. Turbine In': 'T4', 'Exh.Gas Temp. Turbine Out': 'T5',
    'Charge Air IC Air Temp. In': 'T2', 'Charge Air IC Air Temp. Out': 'T_rec',
    'Charge Air IC Cooling Water Temp. In': 'Tw_in', 'Charge Air IC Cooling Water Temp. Out': 'Tw_out',
    'Charge Air IC Cooling water flow': 'Vw_ac', 'Engine room Temp.': 'T1',
    'Exh. Gas Mass Flow': 'm_exh_ds', 'Max. In-Cylinder Press. No.1': 'pmax1',
    'Max. In-Cylinder Press. No.2': 'pmax2', 'Max. In-Cylinder Press. No.3': 'pmax3',
    'No.1 Exh.Gas Temp.': 'Tex1', 'No.2 Exh.Gas Temp.': 'Tex2', 'No.3 Exh.Gas Temp.': 'Tex3',
    'Cooling Water Temp. Engine In': 'Tjw_in', 'Cooling Water Temp. Engine Out I': 'Tjw_out1',
    'LO Temp. Engine In': 'Tlo_in', 'LO Temp. Engine Out': 'Tlo_out', 'Fuel Temp.': 'Tfuel',
    'LO Temp. TCH In': 'Tlo_tc_in', 'LO Temp. TCH Out': 'Tlo_tc_out',
}
TEMPS = ['T4', 'T5', 'T2', 'T_rec', 'Tw_in', 'Tw_out', 'T1', 'Tex1', 'Tex2', 'Tex3']

RUNS = {
    'AF_Clogging': ['AF_Clogging_40_Load', 'AF_Clogging_60_Load', 'AF_Clogging_75_Load', 'AF_Clogging_85_Load'],
    'AC_Fouling': ['AC_Fouling_40_Load', 'AC_Fouling_60_Load', 'AC_Fouling_75_Load', 'AC_Fouling_85_Load'],
    'Turbine_Degradation': ['Turbine_Degradation_40_Load', 'Turbine_Degradation_60_Load',
                            'Turbine_Degradation_85_Load'],
    'Pump_Cavitation': ['CW_Pump_Cavitation_60_Load', 'CW_Pump_Cavitation_85_Load'],
}
FAULT_OF_RUN = {r: f for f, rs in RUNS.items() for r in rs}


def read_raw(name):
    """One file with the full-name header row as columns. name: 'Reference_Data' or a scenario run name."""
    if name == 'Reference_Data':
        path = DATA / 'Reference_Data.csv'
    else:
        path = DATA / FAULT_OF_RUN[name] / f'{name}.csv'
    return pd.read_csv(path, header=0, skiprows=[1, 2], encoding='utf-8')


def clean(raw):
    """Short column names, SI-friendly units, sentinel values removed. Time in s from the start of the run."""
    df = raw.rename(columns=COLS)
    out = pd.DataFrame({'t': raw['Time'] - raw['Time'].iloc[0]} if 'Time' in raw else {'t': raw['Time_rel']})
    out['anomaly'] = raw['Anomaly State'] if 'Anomaly State' in raw else 0
    for c in COLS.values():
        out[c] = df[c] if c in df else np.nan
    for c in TEMPS:
        out.loc[out[c] >= SENTINEL - 0.5, c] = np.nan
    out['m_fuel'] = out['fuel_kgh'] / 3600.0                      # kg/s
    out['p_rec'] = P_AMB + out['p_rec_g'] * KGF_CM2                # Pa, absolute
    out['m_exh_ds'] = out['m_exh_ds']                              # kg/s (speed-density, dataset)
    out['m_air'] = out['m_exh_ds'] - out['m_fuel']
    out['m_w_ac'] = out['Vw_ac'] * 998.0 / 3600.0                  # kg/s
    return out


def load(name):
    return clean(read_raw(name))


def blocks(df, seconds=60.0):
    """Non-overlapping block means (NaN-aware) plus within-block std of speed and power."""
    b = (df['t'] // seconds).astype(int)
    g = df.groupby(b)
    m = g.mean(numeric_only=True)
    m['N_std'] = g['N'].std()
    m['P_std'] = g['P'].std()
    m['n'] = g.size()
    m['anomaly'] = g['anomaly'].max()
    return m[m['n'] >= 0.8 * seconds / 2.0].reset_index(drop=True)


def steady(b, window=5, dT4=4.0, dp=0.01, dP=2.5):
    """Flag blocks that sit on a settled operating point: over +-window blocks the turbine-inlet temperature,
    boost (kgf/cm2) and shaft power move by less than the given amounts. Excludes load steps and their thermal
    settling, which matters for a model calibrated on steady points."""
    r = lambda s: s.rolling(2 * window + 1, center=True, min_periods=window + 1)
    ok = ((r(b['T4']).max() - r(b['T4']).min()) < dT4) \
        & ((r(b['p_rec_g']).max() - r(b['p_rec_g']).min()) < dp) \
        & ((r(b['P']).max() - r(b['P']).min()) < dP)
    return ok.fillna(False).to_numpy()


def reference_sweeps(b):
    """The reference file holds two load sweeps; split at the large power drop between them."""
    P = b['P'].to_numpy()
    k = int(np.argmin(np.diff(P[len(P) // 3: 2 * len(P) // 3]))) + len(P) // 3 + 1
    return np.arange(len(P)) < k
