"""Physics-based health indicators computed from measured channels (block means).

Each indicator removes the first-order effect of the operating point by a physical law, so that what is left
moves with the component's condition rather than with load:

  K_f   air-filter loss coefficient     dp_f rho_1 / m_air^2           (quadratic loss law)
  K_ex  exhaust back-pressure coeff.    dp_ex rho_5 / m_exh^2
  eta_c compressor isentropic efficiency (apparent: inlet temperature = engine-room temperature)
  NTU   charge-air cooler, air side     -ln(1 - eps), eps = (T2 - T_rec)/(T2 - Tw_in)
  WR    turbocharger work ratio         compressor power / turbine gas-side enthalpy drop
  f_ex  exhaust energy fraction         m_exh cp (T4 - T_rec) / (m_fuel LHV)

The remaining (weaker) load dependence is removed by a healthy baseline fitted on reference data (baseline.py).
"""
import numpy as np
import pandas as pd

from .io import LHV, P_AMB
from .thermo import CP_AIR, CP_EXH, R_AIR, R_EXH, T0C, compressor_eta

NAMES = ['K_f', 'K_ex', 'eta_c', 'NTU', 'WR', 'f_ex']


def compute(b, p_amb=P_AMB):
    """b: block-mean frame from io.blocks. Returns a frame with the six indicators plus useful intermediates."""
    T1 = b['T1'] + T0C
    T2 = b['T2'] + T0C
    Trec = b['T_rec'] + T0C
    T4 = b['T4'] + T0C
    T5 = b['T5'] + T0C
    m_air = b['m_air']
    m_exh = b['m_exh_ds']
    p1 = p_amb - b['dp_f']
    p_rec = b['p_rec'] - P_AMB + p_amb
    p5 = p_amb + b['dp_ex']
    pi_c = p_rec / p1
    out = pd.DataFrame(index=b.index)
    out['K_f'] = b['dp_f'] * (p1 / (R_AIR * T1)) / m_air ** 2
    out['K_ex'] = b['dp_ex'] * (p5 / (R_EXH * T5)) / m_exh ** 2
    out['eta_c'] = compressor_eta(T1, T2, pi_c)
    eps = (T2 - Trec) / (T2 - (b['Tw_in'] + T0C))
    out['NTU'] = -np.log(np.clip(1.0 - eps, 1e-6, None))
    W_c = m_air * CP_AIR * (T2 - T1)
    W_t_gas = m_exh * CP_EXH * (T4 - T5)
    out['WR'] = W_c / W_t_gas
    out['f_ex'] = m_exh * CP_EXH * (T4 - Trec) / (b['m_fuel'] * LHV)
    out['eps'] = eps
    out['pi_c'] = pi_c
    out['W_c'] = W_c
    out['W_t_gas'] = W_t_gas
    return out
