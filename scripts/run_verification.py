"""Verification of the building blocks and of the calibration / detection chain on synthetic data with known truth.

Writes results/verification.json and figures/fig1_verification.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tcdiag import detect, io, mvem, thermo as th      # noqa: E402

RES, FIG = ROOT / 'results', ROOT / 'figures'
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
out = {}

# 1. orifice law: choked limit and subsonic isentropic flow, written out independently
g, R, A, p0, T0 = th.G_EXH, th.R_EXH, 1e-3, 3e5, 800.0
m_choked = A * p0 * np.sqrt(g / (R * T0)) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
pr = 0.9
M = np.sqrt(2 / (g - 1) * (pr ** (-(g - 1) / g) - 1))
T = T0 / (1 + 0.5 * (g - 1) * M ** 2)
m_sub = A * pr * p0 / (R * T) * M * np.sqrt(g * R * T)
out['orifice'] = dict(choked_rel_err=float(abs(th.orifice_mass_flow(A, p0, T0, 1e5) / m_choked - 1)),
                      subsonic_rel_err=float(abs(th.orifice_mass_flow(A, p0, T0, pr * p0) / m_sub - 1)))

# 2. effectiveness-NTU inversion round trip, including the cr -> 0 limit
ntu = np.linspace(0.05, 4, 80)[:, None]
cr = np.array([0.0, 0.05, 0.3, 0.7, 0.99])[None, :]
eps = th.eps_crossflow_cmin_mixed(ntu, cr)
out['eps_ntu_roundtrip_max_err'] = float(np.max(np.abs(th.ntu_crossflow_cmin_mixed(eps, cr) - ntu)))

# 3. compressor efficiency round trip
T1, pis, etas = 300.0, np.linspace(1.05, 2.5, 30), np.linspace(0.5, 0.9, 30)
T2 = th.compressor_outlet_T(T1, pis, etas)
out['compressor_eta_roundtrip_max_err'] = float(np.max(np.abs(th.compressor_eta(T1, T2, pis) - etas)))

# 4. model: balances at solved operating points, and recovery of known parameters from synthetic boost data
ref = io.blocks(io.load('Reference_Data'))
st = io.steady(ref)
true, _, _ = mvem.calibrate(ref, st, 1.5e4, eta_t=0.7)
idx = np.flatnonzero(st)
bal, mass = [], []
for i in idx:
    s = mvem.solve(true, *mvem.inputs(ref, i))
    bal.append(abs(s['W_av'] - s['W_c']) / s['W_c'])
    mass.append(abs(s['m_exh'] - s['m_air'] - ref['m_fuel'].iloc[i]) / s['m_exh'])
out['model_balance'] = dict(turbo_power_rel_max=float(max(bal)), mass_rel_max=float(max(mass)), points=len(idx))

rng = np.random.default_rng(1)
syn = ref.copy()
clean = mvem.predict(true, ref, idx)
rec = []
for noise in [0.0, 0.005, 0.02]:
    syn.loc[idx, 'p_rec_g'] = clean['p_rec_g'].to_numpy() + noise * rng.standard_normal(len(idx))
    # closures are taken from the real data, so only the boost is synthetic: refit A_t and slope from it
    p, sol, cov = mvem.calibrate(syn, st, 1.5e4, eta_t=0.7)
    rec.append(dict(noise_bar=noise, A_t_true_mm2=true.A_t * 1e6, A_t_fit_mm2=p.A_t * 1e6,
                    A_t_sd_mm2=float(np.sqrt(cov[0, 0]) * 1e3),
                    slope_true=true.eta_t_slope, slope_fit=p.eta_t_slope,
                    slope_sd=float(np.sqrt(cov[1, 1]) * 1e-3)))
out['synthetic_recovery'] = rec

# 5. detector on synthetic faults: healthy indicator noise plus a ramp in one indicator
rng = np.random.default_rng(2)
hits = []
for k in detect.PHYS:
    for rep in range(20):
        n, onset = 150, 60
        b = {'z': rng.standard_normal((n, 3))}
        j = detect.PHYS.index(k)
        b['z'][onset:, j] += detect.SIGN[k] * np.linspace(0, 15, n - onset)
        zs = {kk: detect.ewma(b['z'][:, i]) for i, kk in enumerate(detect.PHYS)}
        train = np.arange(30)
        lim = {kk: max(detect.H_MIN, detect.PHYS_SAFETY * np.quantile(detect.SIGN[kk] * zs[kk][train], 0.995))
               for kk in zs}
        alarm = np.any([detect.SIGN[kk] * zs[kk] > lim[kk] for kk in zs], axis=0)
        t = detect.first_persistent(alarm, 30)
        iso = None
        if t is not None:
            c = {kk: detect.SIGN[kk] * zs[kk][t + 2] / lim[kk] for kk in zs if detect.SIGN[kk] * zs[kk][t + 2] > lim[kk]}
            iso = max(c, key=c.get) if c else None
        hits.append(dict(ind=k, early=t is not None and t < onset, correct=iso == k))
out['synthetic_detector'] = dict(trials=len(hits), false_before_onset=sum(h['early'] for h in hits),
                                 correctly_isolated=sum(h['correct'] for h in hits))

(RES / 'verification.json').write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(out, indent=1, default=float))

fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for i, c in enumerate(cr[0]):
    ax[0].plot(ntu[:, 0], eps[:, i], label=f'C_r = {c:g}')
ax[0].set_xlabel('NTU')
ax[0].set_ylabel('effectiveness')
ax[0].legend(frameon=False, fontsize=7)
ax[0].set_title('cross-flow, air side mixed')
ax[1].semilogy(np.array(bal) + 1e-17, '.', label='turbocharger power balance')
ax[1].semilogy(np.array(mass) + 1e-17, '.', label='mass balance')
ax[1].set_xlabel('solved reference operating point')
ax[1].set_ylabel('relative residual')
ax[1].legend(frameon=False, fontsize=7)
nz = [r['noise_bar'] for r in rec]
ax[2].errorbar(nz, [r['A_t_fit_mm2'] for r in rec], yerr=[2 * r['A_t_sd_mm2'] for r in rec], fmt='o', capsize=3)
ax[2].axhline(true.A_t * 1e6, c='k', ls=':', label='true')
ax[2].set_xlabel('noise added to boost (bar)')
ax[2].set_ylabel('recovered turbine area (mm^2)')
ax[2].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / 'fig1_verification.png')
