"""Independent checks on the raw data, written without the tcdiag package (pandas only, own column handling).

* reproduce the dataset's derived channels to find the true units and which flows are measured;
* count logger sentinel values;
* engine-level energy balance from the bench's own loss channels;
* charge-air cooler: air-side vs water-side heat;
* one compressor efficiency and one filter loss coefficient by hand.

Writes results/independent_checks.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'data' / 'Marine_Engine_Fault_Data'
out = {}
ref = pd.read_csv(D / 'Reference_Data.csv', header=0, skiprows=[1, 2])
on = ref[ref['Shaft Power'] > 60]

# derived channels
out['TotalInputEnergy_over_FuelFlow'] = dict(mean=float((on['Total Input Energy'] / on['Fuel Flow']).mean()),
                                             std=float((on['Total Input Energy'] / on['Fuel Flow']).std()))
out['ShaftPower_over_TotalInputEnergy_vs_EffectiveEfficiency'] = float(np.median(
    on['Shaft Power'] / on['Total Input Energy'] * 100 / on['Effective Efficiency']))
out['ShaftPower_vs_Torque_x_omega_max_rel'] = float(np.max(np.abs(
    on['Shaft Torque'] * on['Engine Speed'] * 2 * np.pi / 60 / 1e3 / on['Shaft Power'] - 1)))
p_rec_kpa = 101.325 + on['Charge Air Press.'] * 98.0665
sd = on['Exh. Gas Mass Flow'] / (on['Engine Speed'] * p_rec_kpa / (on['Charge Air IC Air Temp. Out'] + 273.15))
out['ExhMassFlow_speed_density_coefficient'] = dict(mean=float(sd.mean()), cv=float(sd.std() / sd.mean()))
tq = on['TCH Power'] / (on['Exh. Gas Mass Flow'] * (on['Exh.Gas Temp. Turbine In'] - on['Exh.Gas Temp. Turbine Out']))
out['TCHPower_over_mdot_dT_median'] = float(tq.median())
bsfc = on['Fuel Flow'] / on['Shaft Power']
out['bsfc_kg_per_kWh_if_fuel_in_kg_h'] = dict(median=float(bsfc.median()), p10=float(bsfc.quantile(0.1)),
                                              p90=float(bsfc.quantile(0.9)))

# sentinels
sent = {}
for f in sorted(D.glob('**/*.csv')):
    if f.name in ('dataset_index.csv', 'variable_dictionary.csv'):
        continue
    df = pd.read_csv(f, header=0, skiprows=[1, 2])
    n = int((df.select_dtypes('number') == 999).sum().sum())
    if n:
        sent[f.stem] = n
out['sentinel_999_counts'] = sent

# engine energy balance in kW: fuel = shaft + losses listed by the bench + exhaust + other
fuel = on['Total Input Energy']                       # kW (see above)
parts = ['Shaft Power', 'Mechanical Loss', 'Loss with cooling water', 'Loss with LO', 'Loss in Charge Air IC',
         'Loss with TCH LO', 'Exh.Gas Power', 'Other Loss']
tot = on[parts].sum(axis=1)
out['energy_balance_bench_channels'] = dict(
    closure_median=float((tot / fuel).median()), closure_p10=float((tot / fuel).quantile(0.1)),
    closure_p90=float((tot / fuel).quantile(0.9)),
    other_loss_share_median=float((on['Other Loss'] / fuel).median()),
    note='sum of the bench\'s own energy channels / fuel energy; Other Loss is zero in most samples, so it is '
         'not a balancing term and the gap is energy the bench does not attribute (radiation, unmeasured flows)')

# cooler: air-side vs water-side heat (block means of 120 s)
m_air = on['Exh. Gas Mass Flow'] - on['Fuel Flow'] / 3600
Qa = m_air * 1.0045 * (on['Charge Air IC Air Temp. In'] - on['Charge Air IC Air Temp. Out'])
Qw = on['Charge Air IC Cooling water flow'] * 998 / 3600 * 4.18 * (
    on['Charge Air IC Cooling Water Temp. Out'] - on['Charge Air IC Cooling Water Temp. In'])
blk = np.arange(len(on)) // 60
bq = pd.DataFrame({'Qa': Qa.to_numpy(), 'Qw': Qw.to_numpy(), 'P': on['Shaft Power'].to_numpy(), 'b': blk}).groupby('b').mean()
ratio = {}
for lo, hi in [(60, 100), (100, 150), (150, 200), (200, 260)]:
    s = bq[(bq.P >= lo) & (bq.P < hi)]
    ratio[f'{lo}-{hi} kW'] = float((s.Qw / s.Qa).median())
out['cooler_water_over_air_heat'] = ratio

# hand calculation at one steady reference point (the 200th 2-s sample after the 3rd hour)
r = on.iloc[5400]
p1 = 101325 - r['Compressor Filter Loss']
p2 = 101325 + r['Charge Air Press.'] * 98066.5
T1 = r['Engine room Temp.'] + 273.15
T2 = r['Charge Air IC Air Temp. In'] + 273.15
eta = T1 * ((p2 / p1) ** (0.4 / 1.4) - 1) / (T2 - T1)
ma = r['Exh. Gas Mass Flow'] - r['Fuel Flow'] / 3600
Kf = r['Compressor Filter Loss'] * (p1 / (287 * T1)) / ma ** 2
out['hand_point'] = dict(P_kW=float(r['Shaft Power']), pi_c=float(p2 / p1), eta_c=float(eta), K_f=float(Kf))
(ROOT / 'results' / 'independent_checks.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
