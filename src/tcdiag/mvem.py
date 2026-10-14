"""Mean-value engine and turbocharger model, calibrated on healthy test-bench data.

Inputs per operating point (all measured): speed N, fuel flow, compressor inlet temperature T1 (engine room),
receiver temperature T_rec (held by the cooler's water controller), filter loss dp_f and turbine back pressure dp_ex.
Unknown: the boost pressure p_rec, found by balancing the turbocharger:

  air flow      m_exh = C_SD N p_rec / T_rec  (speed-density, as in the dataset), m_air = m_exh - m_fuel
  compressor    pi = p_rec / (p_amb - dp_f), T2 = T1 (1 + (pi^k - 1) / eta_c(m_air)),  W_c = m_air cp (T2 - T1)
  cylinders     T4 = T_rec + f_ex m_fuel LHV / (m_exh cp_g),  f_ex linear in (P, N, phi = m_fuel / m_air)
  turbine       nozzle of area A_t (Project 1 orifice law): m_exh = A_t p4 psi(p5/p4) / sqrt(R T4) -> p4
                shaft power  eta_t m_exh cp_g T4 (1 - (p5/p4)^k_g),   p5 = p_amb + dp_ex
                (eta_t here includes the bearing losses: turbine x mechanical efficiency)
  balance       shaft power = W_c
  outlet        T5 = T4 - W_c / (WR(P) m_exh cp_g); WR = measured compressor power / turbine gas enthalpy
                drop, i.e. the gas also loses heat through the uninsulated turbine casing (WR 0.3 at low load,
                0.75 at high load on this bench)

eta_c(m_air), f_ex and WR(P) are low-order fits to the healthy data (two passes, points beyond 4 robust standard
deviations dropped). Without a turbine-inlet pressure measurement, A_t and eta_t are not separately identifiable
(their estimates are 99 % correlated when fitted together), so eta_t is fixed and A_t is calibrated on the measured
boost; the analysis repeats everything for eta_t = 0.6 / 0.7 / 0.8. The model is then used, with no refitting, to
predict how boost and temperatures respond to the measured fault inputs.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, least_squares

from . import indicators as ind
from .io import LHV, P_AMB
from .thermo import CP_AIR, CP_EXH, G_EXH, K_AIR, R_EXH, T0C, orifice_mass_flow, turbine_isentropic_work

C_SD = 0.0015408e-3          # kg/s per (rpm Pa / K): the dataset's speed-density coefficient (p in Pa)


@dataclass
class Params:
    eta_c: np.ndarray = field(default_factory=lambda: np.array([0.78, 0.0, 0.0]))   # poly in (m_air - 0.25)
    f_ex: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.0, 0.0, 0.0]))  # 1, P, N, phi
    wr: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.0, 0.0]))       # poly in (P - 150) kW
    A_t: float = 2.0e-3
    eta_t: float = 0.7
    eta_t_slope: float = 0.0   # per kW of shaft power, about P = 150 kW
    K_ex: float = 1.5e4      # back-pressure coefficient used where dp_ex was not logged (healthy median)
    K_f: float = 3.2e3       # filter loss coefficient used where dp_f was not logged (healthy median)


def _poly(c, x):
    return np.polyval(c[::-1], x)


def _robust_lstsq(X, y, k=4.0):
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    for _ in range(2):
        c, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        r = y - X @ c
        mad = 1.4826 * np.median(np.abs(r[ok] - np.median(r[ok])))
        ok = ok & (np.abs(r) < k * mad)
    c, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return c, int(np.sum(~ok))


def _fex_x(P, N, phi):
    return np.c_[np.ones_like(np.atleast_1d(P)), np.atleast_1d(P), np.atleast_1d(N), np.atleast_1d(phi)]


def fit_closures(b, I, mask):
    """Fit eta_c, f_ex and WR closures by robust least squares on healthy steady blocks."""
    p = Params()
    m = b['m_air'][mask].to_numpy()
    phi = (b['m_fuel'] / b['m_air'])[mask].to_numpy()
    P = b['P'][mask].to_numpy()
    N = b['N'][mask].to_numpy()
    V = lambda x, n: np.vander(x, n + 1, increasing=True)
    p.eta_c, n1 = _robust_lstsq(V(m - 0.25, 2), I['eta_c'][mask].to_numpy())
    p.f_ex, n2 = _robust_lstsq(_fex_x(P, N, phi), I['f_ex'][mask].to_numpy())
    p.wr, n3 = _robust_lstsq(V(P - 150.0, 2), I['WR'][mask].to_numpy())
    p.dropped = dict(eta_c=n1, f_ex=n2, wr=n3)
    return p


def _p4(m_exh, T4, p5, A_t):
    f = lambda p4: orifice_mass_flow(A_t, p4, T4, p5) - m_exh
    hi = p5 * 1.01
    while f(hi) < 0:
        hi *= 1.5
        if hi > 50 * p5:
            return np.nan
    return brentq(f, p5 * (1 + 1e-9), hi, xtol=1e-6)


def point(p, N, m_fuel, T1, T_rec, dp_f, dp_ex, P, p_rec, p_amb=P_AMB):
    """All model states for a given boost pressure. Temperatures in K, pressures in Pa."""
    m_exh = C_SD * N * p_rec / T_rec
    m_air = m_exh - m_fuel
    if not np.isfinite(dp_f):
        dp_f = p.K_f * m_air ** 2 * R_EXH * T1 / p_amb
    pi = p_rec / (p_amb - dp_f)
    eta_c = _poly(p.eta_c, m_air - 0.25)
    T2 = T1 * (1.0 + (pi ** K_AIR - 1.0) / eta_c)
    W_c = m_air * CP_AIR * (T2 - T1)
    T4 = T_rec + float((_fex_x(P, N, m_fuel / m_air) @ p.f_ex)[0]) * m_fuel * LHV / (m_exh * CP_EXH)
    # where dp_ex was not logged: healthy K_ex, with T5 approximated by T4 - 60 K in the density
    p5 = p_amb + (dp_ex if np.isfinite(dp_ex) else p.K_ex * m_exh ** 2 * R_EXH * (T4 - 60.0) / p_amb)
    wr = _poly(p.wr, P - 150.0)
    W_gas = W_c / wr
    p4 = _p4(m_exh, T4, p5, p.A_t)
    W_av = (p.eta_t + p.eta_t_slope * (P - 150.0)) * turbine_isentropic_work(m_exh, T4, p4, p5)
    T5 = T4 - W_gas / (m_exh * CP_EXH)
    return dict(p_rec=p_rec, m_air=m_air, m_exh=m_exh, T2=T2, T4=T4, T5=T5, p4=p4, p5=p5, W_c=W_c,
                W_gas=W_gas, W_av=W_av, res=W_av - W_c)


def solve(p, N, m_fuel, T1, T_rec, dp_f, dp_ex, P, p_amb=P_AMB):
    """Matched boost pressure: turbine power available = power the compressor needs. The balance can have more
    than one root (at very low boost the T4 closure leaves its range); the operating point is the highest
    stable root, where the surplus turbine power changes from positive to negative as boost rises."""
    g = lambda pr: point(p, N, m_fuel, T1, T_rec, dp_f, dp_ex, P, pr, p_amb)['res']
    grid = p_amb * np.linspace(1.001, 3.0, 61)
    vals = np.array([g(x) for x in grid])
    k = np.flatnonzero((vals[:-1] > 0) & (vals[1:] <= 0))
    if len(k) == 0:
        return None
    i = k[-1]
    pr = brentq(g, grid[i], grid[i + 1], xtol=1.0)
    return point(p, N, m_fuel, T1, T_rec, dp_f, dp_ex, P, pr, p_amb)


def inputs(b, i):
    r = b.iloc[i]
    return (r['N'], r['m_fuel'], r['T1'] + T0C, r['T_rec'] + T0C, r['dp_f'], r['dp_ex'], r['P'])


def predict(p, b, idx=None):
    idx = range(len(b)) if idx is None else idx
    rows = []
    for i in idx:
        s = solve(p, *inputs(b, i))
        rows.append({k: (s[k] if s else np.nan) for k in ('p_rec', 'm_air', 'T2', 'T4', 'T5', 'p4', 'p5')})
    import pandas as pd
    out = pd.DataFrame(rows, index=list(idx))
    for k in ('T2', 'T4', 'T5'):
        out[k] = out[k] - T0C
    out['p_rec_g'] = (out['p_rec'] - P_AMB) / 98.0665e3
    return out


def calibrate(b, mask, K_ex, eta_t=0.7, joint=False):
    """Closures by regression, then A_t and the load slope of eta_t (and its level too if joint) by least squares
    on the measured boost; eta_t is the level at P = 150 kW.
    Returns params, the scipy solution and the parameter covariance."""
    I = ind.compute(b)
    p = fit_closures(b, I, mask)
    p.K_ex = K_ex
    p.eta_t = eta_t
    idx = np.flatnonzero(mask)
    meas = b['p_rec_g'].to_numpy()[idx]

    def r(x):
        p.A_t = x[0] * 1e-3
        p.eta_t_slope = x[1] * 1e-3
        if joint:
            p.eta_t = x[2]
        pr = predict(p, b, idx)['p_rec_g'].to_numpy()
        return np.nan_to_num(pr - meas, nan=1.0)

    x0, lo, hi = [1.5, 0.0], [0.1, -3.0], [20.0, 3.0]
    if joint:
        x0, lo, hi = x0 + [0.7], lo + [0.3], hi + [1.0]
    sol = least_squares(r, x0=x0, bounds=(lo, hi))
    r(sol.x)
    J = sol.jac
    cov = np.linalg.pinv(J.T @ J) * np.sum(sol.fun ** 2) / max(len(idx) - len(x0), 1)
    return p, sol, cov
