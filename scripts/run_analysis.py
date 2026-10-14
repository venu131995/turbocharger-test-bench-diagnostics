"""Full analysis on the Marine Engine Fault test-bench data.

1. Physics health indicators on healthy and faulty data; how well each carries over between test days.
2. Detection and isolation of the three turbocharger-side faults: physics indicators vs PCA (T^2 / SPE),
   over a range of alarm limits, and at matched false-alarm rate.
3. Mean-value engine/turbocharger model calibrated on healthy reference data: validation on unseen healthy data
   and prediction of the measured response to each fault.

Writes results/*.csv, results/summary.json and figures/fig2-fig6.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tcdiag import detect, indicators as ind, io, mvem      # noqa: E402

RES, FIG = ROOT / 'results', ROOT / 'figures'
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
FAULTS = ['AF_Clogging', 'Turbine_Degradation', 'AC_Fouling']
LABEL = {'AF_Clogging': 'air-filter clogging', 'Turbine_Degradation': 'turbine degradation',
         'AC_Fouling': 'charge-air cooler fouling', 'Pump_Cavitation': 'cooling-water pump cavitation'}
COL = {'AF_Clogging': 'C1', 'Turbine_Degradation': 'C3', 'AC_Fouling': 'C2', 'Pump_Cavitation': 'C4'}
SEV = {'AF_Clogging': 'dp_f', 'Turbine_Degradation': 'dp_ex', 'AC_Fouling': 'T_rec'}
t0 = time.time()
summary = {}

# ---------------------------------------------------------------- data
ref = io.blocks(io.load('Reference_Data'))
st = io.steady(ref)
sw1 = io.reference_sweeps(ref)
runs = [r for f in io.RUNS for r in io.RUNS[f]]
B = {r: io.blocks(io.load(r)) for r in runs}
summary['data'] = dict(reference_blocks=int(len(ref)), reference_steady_blocks=int(st.sum()),
                       runs=len(runs), block_seconds=60)

# ---------------------------------------------------------------- 1. indicators and day-to-day transfer
Iref = detect.physics_frame(ref)
names = ['K_f', 'K_ex', 'UA', 'eta_c', 'WR', 'f_ex']
fits = {}
for k in ['K_f', 'UA', 'eta_c', 'WR', 'f_ex']:
    y = Iref[k][st].to_numpy()
    x = ref['P'][st].to_numpy()
    ok = np.isfinite(y)
    fits[k] = np.polyfit(x[ok], y[ok], 2)
kex_h = np.concatenate([detect.physics_frame(B[r])['K_ex'][B[r]['anomaly'] == 0].dropna().to_numpy() for r in runs])
K_EX = float(np.median(kex_h))
K_F = float(np.median(Iref['K_f'][st]))
rows = []
for r in runs:
    b, I = B[r], detect.physics_frame(B[r])
    h = b['anomaly'] == 0
    row = {'run': r}
    for k in names:
        base = (np.polyval(fits[k], b['P'][h]) if k in fits else K_EX)
        row[k] = float(np.nanmedian(I[k][h] / base))
    rows.append(row)
transfer = pd.DataFrame(rows).set_index('run')
transfer.to_csv(RES / 'indicator_transfer.csv', float_format='%.4f')
summary['transfer_between_days'] = {k: dict(median=float(transfer[k].median()),
                                            min=float(transfer[k].min()), max=float(transfer[k].max()))
                                    for k in names}

fig, ax = plt.subplots(2, 3, figsize=(11, 6.2))
ylab = {'K_f': 'filter loss coeff. K_f (1/m^4)', 'K_ex': 'back-pressure coeff. K_ex (1/m^4)',
        'UA': 'cooler UA (W/K)', 'eta_c': 'compressor efficiency (apparent)', 'WR': 'work ratio W_c / W_t,gas',
        'f_ex': 'exhaust energy fraction'}
for a, k in zip(ax.flat, names):
    a.scatter(ref['P'][st], Iref[k][st], s=7, c='k', label='reference, healthy (steady)')
    for r in runs:
        b, I = B[r], detect.physics_frame(B[r])
        h = (b['anomaly'] == 0).to_numpy()
        a.scatter(b['P'][h], I[k][h], s=3, c='0.6', alpha=0.5)
        a.scatter(b['P'][~h], I[k][~h], s=3, c=COL[io.FAULT_OF_RUN[r]], alpha=0.5)
    a.set_xlabel('shaft power (kW)')
    a.set_ylabel(ylab[k])
ax.flat[1].set_ylim(0, 6e4)
ax.flat[2].set_ylim(0, 2.5e3)
ax.flat[3].set_ylim(0.55, 1.0)
ax.flat[4].set_ylim(0, 1.0)
ax.flat[5].set_ylim(0.2, 0.36)
hs = [plt.Line2D([], [], ls='', marker='o', c=c, ms=4) for c in ['k', '0.6'] + [COL[f] for f in COL]]
fig.legend(hs, ['reference', 'before fault'] + [LABEL[f] for f in COL], loc='lower center', ncol=6, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(FIG / 'fig2_indicators.png')
plt.close(fig)

# ---------------------------------------------------------------- 2. detection and isolation
target = [r for r in runs if io.FAULT_OF_RUN[r] in FAULTS]


def evaluate(method, **kw):
    for k, v in kw.items():
        setattr(detect, k, v)
    out = {}
    for r in runs:
        o = method(B[r])
        b = B[r]
        sev = None
        if o['t_det'] is not None and io.FAULT_OF_RUN[r] in SEV:
            c = SEV[io.FAULT_OF_RUN[r]]
            h = b['anomaly'] == 0
            sev = float(b[c].iloc[o['t_det']] - b[c][h].mean())
        out[r] = dict(false_alarm=o['false_alarm'], delay=None if o['t_det'] is None else int(o['t_det'] - o['onset']),
                      isolated=o['isolated'], severity=sev, top=o.get('top_channel'))
    return out


def agg(res):
    fa = np.mean([res[r]['false_alarm'] for r in runs])
    det = [r for r in target if res[r]['delay'] is not None]
    iso = sum(res[r]['isolated'] == io.FAULT_OF_RUN[r] for r in target)
    d = [res[r]['delay'] for r in det]
    return dict(false_alarm=float(fa), detected=len(det), isolated=int(iso),
                median_delay_min=float(np.median(d)) if d else None)


sweep = []
for h in [3, 4.5, 6, 9, 12]:
    sweep.append(dict(method='physics', setting=h, **agg(evaluate(detect.physics_monitor, H_MIN=h))))
detect.H_MIN = 6.0
for s in [1.5, 3, 6, 12, 24, 48, 96]:
    sweep.append(dict(method='pca', setting=s, **agg(evaluate(detect.pca_monitor, SAFETY=s))))
sweep = pd.DataFrame(sweep)
sweep.to_csv(RES / 'detection_sweep.csv', index=False, float_format='%.4f')

# matched operating points: physics at H_MIN = 6; PCA at the smallest SAFETY whose false-alarm rate is within
# 10 % (relative) of the physics one
phys = evaluate(detect.physics_monitor, H_MIN=6.0)
fa_p = agg(phys)['false_alarm']
cand = sweep[(sweep.method == 'pca') & (sweep.false_alarm <= 1.1 * fa_p + 1e-9)]
s_match = float(cand.setting.min()) if len(cand) else 96.0
pca = evaluate(detect.pca_monitor, SAFETY=s_match)
pca_loose = evaluate(detect.pca_monitor, SAFETY=24.0)
detect.SAFETY = 1.5
tab = []
for r in runs:
    tab.append(dict(run=r, fault=io.FAULT_OF_RUN[r],
                    phys_delay=phys[r]['delay'], phys_isolated=phys[r]['isolated'], phys_severity=phys[r]['severity'],
                    phys_false_alarm=phys[r]['false_alarm'],
                    pca_delay=pca[r]['delay'], pca_isolated=pca[r]['isolated'], pca_top=pca[r]['top'],
                    pca_severity=pca[r]['severity'], pca_false_alarm=pca[r]['false_alarm'],
                    pca24_delay=pca_loose[r]['delay'], pca24_isolated=pca_loose[r]['isolated'],
                    pca24_false_alarm=pca_loose[r]['false_alarm']))
tab = pd.DataFrame(tab)
tab.to_csv(RES / 'detection.csv', index=False, float_format='%.4f')
summary['detection'] = dict(physics=agg(phys), physics_H_MIN=6.0, pca_matched=agg(pca), pca_matched_safety=s_match,
                            pca_safety24=agg(pca_loose))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for m, c, lab in [('physics', 'C0', 'physics indicators'), ('pca', 'C7', 'PCA on raw channels')]:
    s = sweep[sweep.method == m]
    ax[0].plot(100 * s.false_alarm, s.median_delay_min, 'o-', c=c, label=lab)
    ax[1].plot(100 * s.false_alarm, s.isolated, 'o-', c=c, label=lab)
    for _, q in s.iterrows():
        ax[0].annotate(f'{q.setting:g}', (100 * q.false_alarm, q.median_delay_min), fontsize=7,
                       xytext=(3, 3), textcoords='offset points')
ax[0].set_xscale('symlog', linthresh=1)
ax[1].set_xscale('symlog', linthresh=1)
ax[0].set_xlabel('false alarms on held-out healthy blocks (%)')
ax[0].set_ylabel('median detection delay (min)')
ax[1].set_xlabel('false alarms on held-out healthy blocks (%)')
ax[1].set_ylabel('faults correctly isolated (of 11)')
ax[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / 'fig4_detection_tradeoff.png')
plt.close(fig)

# example monitors, one run per fault
ex = ['AF_Clogging_60_Load', 'Turbine_Degradation_60_Load', 'AC_Fouling_85_Load']
fig, ax = plt.subplots(3, 3, figsize=(12, 7.5), sharex='col')
detect.H_MIN = 6.0
for j, r in enumerate(ex):
    b = B[r]
    o = detect.physics_monitor(b)
    detect.SAFETY = s_match
    q = detect.pca_monitor(b)
    detect.SAFETY = 1.5
    t = np.arange(len(b))
    c = SEV[io.FAULT_OF_RUN[r]]
    ax[0, j].plot(t, b[c], c=COL[io.FAULT_OF_RUN[r]])
    ax[0, j].set_ylabel({'dp_f': 'filter loss (Pa)', 'dp_ex': 'turbine back pressure (Pa)',
                         'T_rec': 'charge-air temperature (C)'}[c])
    ax[0, j].set_title(LABEL[io.FAULT_OF_RUN[r]] + f', {r.split("_")[-2]} % load')
    for k, cc in [('K_f', 'C1'), ('K_ex', 'C3'), ('UA', 'C2')]:
        if k in o['zs']:
            ax[1, j].plot(t, detect.SIGN[k] * o['zs'][k], c=cc, label=k)
            ax[1, j].axhline(o['lim'][k], c=cc, ls=':', lw=0.8)
    ax[1, j].set_ylabel('signed EWMA z')
    ax[1, j].set_ylim(-10, 60)
    ax[1, j].legend(frameon=False, fontsize=7, loc='upper left')
    ax[2, j].semilogy(t, q['Q'] / q['lQ'], c='C7', label='SPE / limit')
    ax[2, j].semilogy(t, q['T2'] / q['lT2'], c='k', lw=0.8, label='T^2 / limit')
    ax[2, j].axhline(1, c='C7', ls=':')
    ax[2, j].set_ylabel('PCA statistic / limit')
    ax[2, j].set_xlabel('time (min)')
    ax[2, j].legend(frameon=False, fontsize=7, loc='upper left')
    for a in ax[:, j]:
        a.axvline(o['onset'], c='r', lw=0.8)
        a.axvspan(o['train'][0], o['train'][-1], color='0.9', zorder=0)
        if o['t_det'] is not None:
            a.axvline(o['t_det'], c='C0', ls='--', lw=0.8)
fig.tight_layout()
fig.savefig(FIG / 'fig3_monitoring_examples.png')
plt.close(fig)

# ---------------------------------------------------------------- 3. mean-value model
cal = {}
for et in [0.6, 0.7, 0.8]:
    p, sol, cov = mvem.calibrate(ref, st & sw1, K_EX, eta_t=et)
    p.K_f = K_F
    errs = {}
    for name, m in [('calibration (sweep 1)', st & sw1), ('validation (sweep 2)', st & ~sw1)]:
        idx = np.flatnonzero(m)
        pr = mvem.predict(p, ref, idx)
        errs[name] = {k: dict(bias=float(np.nanmean(pr[k].to_numpy() - ref[k].to_numpy()[idx])),
                              rmse=float(np.sqrt(np.nanmean((pr[k].to_numpy() - ref[k].to_numpy()[idx]) ** 2))))
                      for k in ['p_rec_g', 'T2', 'T4', 'T5']}
    cal[et] = dict(A_t_mm2=p.A_t * 1e6, eta_t_slope_per_kW=p.eta_t_slope, errors=errs, params=p)
pj, solj, covj = mvem.calibrate(ref, st & sw1, K_EX, joint=True)
summary['model'] = {f'eta_t={et}': {k: v for k, v in cal[et].items() if k != 'params'} for et in cal}
summary['model']['joint_fit'] = dict(A_t_mm2=pj.A_t * 1e6, eta_t=pj.eta_t,
                                     corr_At_eta=float(covj[0, 2] / np.sqrt(covj[0, 0] * covj[2, 2])))
summary['model']['closure_points_dropped'] = cal[0.7]['params'].dropped

# fault responses: model calibrated on all steady reference points, eta_t = 0.6 / 0.7 / 0.8
resp = []
pred_store = {}
for et in [0.6, 0.7, 0.8]:
    p, _, _ = mvem.calibrate(ref, st, K_EX, eta_t=et)
    p.K_f = K_F
    for f in FAULTS:
        for r in io.RUNS[f]:
            b = B[r]
            pr = mvem.predict(p, b)
            if et == 0.7:
                pred_store[r] = pr
            h = np.flatnonzero(b['anomaly'] == 0)
            e = np.flatnonzero(b['anomaly'] == 1)[-10:]
            row = dict(run=r, fault=f, eta_t=et)
            for k in ['p_rec_g', 'T2', 'T4', 'T5']:
                row[f'd{k}_meas'] = b[k].iloc[e].mean() - b[k].iloc[h].mean()
                row[f'd{k}_model'] = pr[k].iloc[e].mean() - pr[k].iloc[h].mean()
                row[f'{k}_err_before'] = (pr[k].iloc[h] - b[k].iloc[h]).mean()
            for k in ['dp_f', 'dp_ex', 'T_rec']:
                row[f'd{k}'] = b[k].iloc[e].mean() - b[k].iloc[h].mean()
            resp.append(row)
resp = pd.DataFrame(resp)
resp.to_csv(RES / 'fault_response.csv', index=False, float_format='%.4f')
r7 = resp[resp.eta_t == 0.7]
summary['fault_response'] = {
    f: dict(dboost_meas=r7[r7.fault == f]['dp_rec_g_meas'].round(4).tolist(),
            dboost_model=r7[r7.fault == f]['dp_rec_g_model'].round(4).tolist(),
            dT4_meas=r7[r7.fault == f]['dT4_meas'].round(2).tolist(),
            dT4_model=r7[r7.fault == f]['dT4_model'].round(2).tolist())
    for f in FAULTS}
summary['fault_response']['eta_t_spread_dT4_model_K'] = float(
    resp.groupby('run')['dT4_model'].agg(lambda s: s.max() - s.min()).max())
summary['fault_response']['eta_t_spread_dboost_model'] = float(
    resp.groupby('run')['dp_rec_g_model'].agg(lambda s: s.max() - s.min()).max())

# model validation figure
p7 = cal[0.7]['params']
idx = np.flatnonzero(st)
pr = mvem.predict(p7, ref, idx)
fig, ax = plt.subplots(1, 4, figsize=(13, 3.3))
for a, k, lab in zip(ax, ['p_rec_g', 'T2', 'T4', 'T5'], ['boost (bar g)', 'compressor outlet T (C)',
                                                         'turbine inlet T (C)', 'turbine outlet T (C)']):
    m1 = sw1[idx]
    a.scatter(ref[k].to_numpy()[idx][m1], pr[k][m1], s=9, c='k', label='calibration (sweep 1)')
    a.scatter(ref[k].to_numpy()[idx][~m1], pr[k][~m1], s=9, c='C0', label='validation (sweep 2)')
    lo, hi = np.nanmin(ref[k].to_numpy()[idx]), np.nanmax(ref[k].to_numpy()[idx])
    a.plot([lo, hi], [lo, hi], c='0.5', lw=0.8)
    a.set_xlabel('measured ' + lab)
    a.set_ylabel('model')
ax[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / 'fig5_model_validation.png')
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for a, k, lab in zip(ax, ['dp_rec_g', 'dT4', 'dT5'], ['change in boost (bar)', 'change in turbine inlet T (K)',
                                                       'change in turbine outlet T (K)']):
    x = np.arange(len(r7))
    a.bar(x - 0.2, r7[f'{k}_meas'], 0.4, color=[COL[f] for f in r7.fault], label='measured')
    lo = resp.groupby('run', sort=False)[f'{k}_model'].min().reindex(r7.run).to_numpy()
    hi = resp.groupby('run', sort=False)[f'{k}_model'].max().reindex(r7.run).to_numpy()
    a.bar(x + 0.2, r7[f'{k}_model'], 0.4, color='w', edgecolor=[COL[f] for f in r7.fault], hatch='///',
          label='model (eta_t 0.7; bar = 0.6-0.8)')
    a.errorbar(x + 0.2, r7[f'{k}_model'], yerr=[r7[f'{k}_model'] - lo, hi - r7[f'{k}_model']], fmt='none',
               c='k', lw=0.8)
    a.set_xticks(x, [r.replace('Turbine_Degradation', 'TD').replace('AF_Clogging', 'AF')
                     .replace('AC_Fouling', 'AC').replace('_Load', '%').replace('_', ' ') for r in r7.run],
                 rotation=60, fontsize=7)
    a.set_ylabel(lab)
    a.axhline(0, c='k', lw=0.5)
ax[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / 'fig6_fault_response.png')
plt.close(fig)

summary['K_ex_healthy_median'] = K_EX
summary['K_f_healthy_median'] = K_F
summary['runtime_s'] = time.time() - t0
(RES / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps({k: summary[k] for k in ('detection', 'transfer_between_days')}, indent=1, default=float))
print(r7[['run', 'dp_rec_g_meas', 'dp_rec_g_model', 'dT4_meas', 'dT4_model', 'dT5_meas', 'dT5_model']].round(3)
      .to_string(index=False))
print(f'done in {time.time() - t0:.0f} s')
