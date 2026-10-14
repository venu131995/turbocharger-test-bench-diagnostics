import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tcdiag import detect, indicators as ind, io, mvem, thermo as th      # noqa: E402


def test_orifice_choked_matches_closed_form():
    g, R, A, p0, T0 = th.G_EXH, th.R_EXH, 1e-3, 3e5, 800.0
    exact = A * p0 * np.sqrt(g / (R * T0)) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
    assert th.orifice_mass_flow(A, p0, T0, 1e5) == pytest.approx(exact, rel=1e-12)
    assert th.orifice_mass_flow(A, p0, T0, p0) == 0.0


def test_eps_ntu_roundtrip_and_limits():
    ntu = np.linspace(0.1, 3, 20)[:, None]
    cr = np.array([0.0, 0.2, 0.9])[None, :]
    eps = th.eps_crossflow_cmin_mixed(ntu, cr)
    assert np.allclose(th.ntu_crossflow_cmin_mixed(eps, cr), ntu, atol=1e-10)
    assert np.allclose(eps[:, 0], 1 - np.exp(-ntu[:, 0]))
    assert np.all(np.diff(eps, axis=1) <= 0)          # more water-side resistance -> lower effectiveness


def test_compressor_roundtrip():
    T2 = th.compressor_outlet_T(300.0, 1.8, 0.75)
    assert th.compressor_eta(300.0, T2, 1.8) == pytest.approx(0.75, rel=1e-12)


def test_indicators_hand_point():
    b = pd.DataFrame(dict(T1=[30.0], T2=[80.0], T_rec=[35.0], T4=[520.0], T5=[460.0], Tw_in=[20.0],
                          m_air=[0.25], m_exh_ds=[0.26], m_fuel=[0.01], dp_f=[200.0], dp_ex=[2000.0],
                          p_rec=[101325 + 0.45 * 98066.5]))
    I = ind.compute(b)
    p1, T1 = 101325 - 200.0, 303.15
    assert I['K_f'][0] == pytest.approx(200.0 * p1 / (287 * T1) / 0.25 ** 2)
    pi = b['p_rec'][0] / p1
    assert I['eta_c'][0] == pytest.approx(T1 * (pi ** (0.4 / 1.4) - 1) / 50.0)
    assert I['eps'][0] == pytest.approx(45.0 / 60.0)


def test_cleaning_units_and_sentinels():
    raw = io.read_raw('Reference_Data')
    d = io.clean(raw)
    assert ((d[io.TEMPS] < 998) | d[io.TEMPS].isna()).all().all()
    on = raw['Shaft Power'] > 60
    assert np.allclose(raw['Total Input Energy'][on], 11.6 * raw['Fuel Flow'][on], rtol=1e-5)
    assert np.allclose(d['m_air'] + d['m_fuel'], d['m_exh_ds'])


def test_model_balances_at_solution():
    ref = io.blocks(io.load('Reference_Data'))
    st = io.steady(ref)
    p = mvem.Params()
    p = mvem.fit_closures(ref, ind.compute(ref), st)
    p.A_t, p.eta_t, p.eta_t_slope = 1.6e-3, 0.7, -1.5e-3
    i = int(np.flatnonzero(st)[10])
    s = mvem.solve(p, *mvem.inputs(ref, i))
    assert abs(s['W_av'] - s['W_c']) / s['W_c'] < 1e-4
    assert s['m_exh'] - s['m_air'] == pytest.approx(ref['m_fuel'].iloc[i], rel=1e-12)
    assert s['p4'] > s['p5'] and s['T5'] < s['T4']


def test_higher_back_pressure_lowers_boost_and_heats_exhaust():
    ref = io.blocks(io.load('Reference_Data'))
    st = io.steady(ref)
    p = mvem.fit_closures(ref, ind.compute(ref), st)
    p.A_t, p.eta_t, p.eta_t_slope = 1.6e-3, 0.7, 0.0
    args = list(mvem.inputs(ref, int(np.flatnonzero(st)[40])))
    args[5] = 2000.0
    a = mvem.solve(p, *args)
    args[5] = 5000.0
    b = mvem.solve(p, *args)
    assert b['p_rec'] < a['p_rec'] and b['T4'] > a['T4']


def test_physics_monitor_flags_and_isolates_air_filter_run():
    b = io.blocks(io.load('AF_Clogging_85_Load'))
    o = detect.physics_monitor(b)
    assert o['t_det'] is not None and o['t_det'] >= o['onset']
    assert o['isolated'] == 'AF_Clogging'
    assert o['false_alarm'] == 0.0
