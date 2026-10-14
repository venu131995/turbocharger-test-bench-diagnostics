"""Gas relations. Air and exhaust gas are two calorically perfect gases; the exhaust values match the one-gas model
of Project 1 (gamma 1.35, R 287 J/kg/K). The orifice law is the same quasi-steady nozzle used there for valves and
the turbine (Project 1, src/gx/gas.py)."""
import numpy as np

R_AIR = 287.0
G_AIR = 1.40
CP_AIR = R_AIR * G_AIR / (G_AIR - 1.0)        # 1004.5 J/kg/K
R_EXH = 287.0
G_EXH = 1.35
CP_EXH = R_EXH * G_EXH / (G_EXH - 1.0)        # 1107 J/kg/K
K_AIR = (G_AIR - 1.0) / G_AIR
K_EXH = (G_EXH - 1.0) / G_EXH
T0C = 273.15


def orifice_mass_flow(A_eff, p0, T0, p_down, g=G_EXH, R=R_EXH):
    """Quasi-steady isentropic flow through a restriction, choking at the critical pressure ratio. Arrays allowed."""
    p0 = np.asarray(p0, float)
    pr = np.clip(np.asarray(p_down, float) / p0, 0.0, 1.0)
    pr_crit = (2.0 / (g + 1.0)) ** (g / (g - 1.0))
    pr = np.maximum(pr, pr_crit)
    psi = np.sqrt(np.maximum(2 * g / (g - 1) * (pr ** (2 / g) - pr ** ((g + 1) / g)), 0.0))
    return A_eff * p0 / np.sqrt(R * T0) * psi


def compressor_outlet_T(T1, pi, eta):
    """Total outlet temperature for a compressor of isentropic efficiency eta and pressure ratio pi."""
    return T1 * (1.0 + (pi ** K_AIR - 1.0) / eta)


def compressor_eta(T1, T2, pi):
    """Isentropic efficiency from inlet/outlet temperatures and pressure ratio."""
    return T1 * (pi ** K_AIR - 1.0) / (T2 - T1)


def turbine_isentropic_work(m, T4, p4, p5):
    """Isentropic enthalpy drop rate (W) for an expansion p4 -> p5 at inlet temperature T4."""
    return m * CP_EXH * T4 * (1.0 - (p5 / p4) ** K_EXH)


def ntu_crossflow_cmin_mixed(eps, cr):
    """NTU of a cross-flow exchanger with the minimum-capacity stream mixed (air side here), inverted analytically:
    eps = (1 - exp(-cr (1 - exp(-NTU)))) / cr. cr -> 0 gives NTU = -ln(1 - eps)."""
    eps = np.asarray(eps, float)
    cr = np.asarray(cr, float)
    small = cr < 1e-6
    inner = np.where(small, eps, -np.log(np.clip(1.0 - cr * eps, 1e-12, None)) / np.where(small, 1.0, cr))
    return -np.log(np.clip(1.0 - inner, 1e-12, None))


def eps_crossflow_cmin_mixed(ntu, cr):
    ntu = np.asarray(ntu, float)
    cr = np.asarray(cr, float)
    small = cr < 1e-6
    a = 1.0 - np.exp(-ntu)
    return np.where(small, a, (1.0 - np.exp(-cr * a)) / np.where(small, 1.0, cr))
