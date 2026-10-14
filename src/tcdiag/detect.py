"""Fault detection and isolation: physics indicators vs a PCA monitor on the raw channels.

Both methods get the same information: each run is monitored against its own first stretch of healthy operation
(the first TRAIN_FRAC of its pre-anomaly blocks), as a test engineer would compare against the start of a test.
The rest of the pre-anomaly blocks measure false alarms; the anomaly blocks measure detection delay.

Physics: standardised residual z = (I - mean_train) / std_train for each indicator, smoothed by an EWMA. The tests
are one-sided in the only direction the physical fault can move them (a clogged filter or a restricted exhaust can
only raise a loss coefficient, fouling can only lower UA). An indicator alarms when its signed EWMA z exceeds its
limit for PERSIST consecutive blocks; the alarming indicator with the largest z names the fault
(K_f -> air filter, K_ex -> turbine/exhaust side, UA -> charge-air cooler).

PCA: standardised raw channels, components up to 90 % variance, Hotelling T^2 and SPE (Q) with the same EWMA and
persistence. For both methods every limit = SAFETY x the 99.5 % quantile of that statistic over the training blocks
(at least H_MIN for the physics z), so the two are tuned by the same rule. Isolation by the channel with the
largest contribution to SPE (or T^2 if only T^2 alarms).
"""
import numpy as np
import pandas as pd

from . import indicators as ind
from .thermo import CP_AIR, ntu_crossflow_cmin_mixed

TRAIN_FRAC = 0.6
LAMBDA = 0.3          # EWMA weight per 60 s block
H_MIN = 3.0           # floor on the physics limit (training standard deviations)
PERSIST = 3           # consecutive blocks
SAFETY = 1.5          # PCA limit factor on the training quantile
PHYS_SAFETY = 1.5     # physics limit factor on the training quantile (same rule, own setting)

PHYS = ['K_f', 'K_ex', 'UA']                         # isolating indicators
SUPPORT = ['eta_c', 'WR', 'f_ex']                    # reported, not used for isolation
SIGN = {'K_f': 1.0, 'K_ex': 1.0, 'UA': -1.0}
FAULT_OF_IND = {'K_f': 'AF_Clogging', 'K_ex': 'Turbine_Degradation', 'UA': 'AC_Fouling'}
RAW = ['N', 'P', 'fuel_kgh', 'p_rec_g', 'dp_f', 'dp_ex', 'T1', 'T2', 'T_rec', 'Tw_in', 'Vw_ac', 'T4', 'T5',
       'Tex1', 'Tex2', 'Tex3', 'pmax1', 'pmax2', 'pmax3', 'Tjw_in', 'Tjw_out1', 'Tlo_in', 'Tlo_out']
FAULT_OF_RAW = {'dp_f': 'AF_Clogging', 'dp_ex': 'Turbine_Degradation', 'T_rec': 'AC_Fouling',
                'Vw_ac': 'AC_Fouling', 'T4': 'Turbine_Degradation', 'T5': 'Turbine_Degradation'}


def physics_frame(b):
    I = ind.compute(b)
    Ca = b['m_air'] * CP_AIR
    Cw = b['m_w_ac'].clip(lower=1e-3) * 4180.0
    cr = np.clip(Ca / Cw, 0.0, 0.999)
    I['UA'] = ntu_crossflow_cmin_mixed(I['eps'], cr) * Ca
    return I


def ewma(x, lam=LAMBDA):
    out = np.empty_like(x)
    s = 0.0
    for i, v in enumerate(x):
        s = lam * v + (1 - lam) * s
        out[i] = s
    return out


def first_persistent(alarm, start, persist=PERSIST):
    run = 0
    for i in range(start, len(alarm)):
        run = run + 1 if alarm[i] else 0
        if run >= persist:
            return i - persist + 1
    return None


def split(b):
    healthy = np.flatnonzero(b['anomaly'].to_numpy() == 0)
    n_train = int(TRAIN_FRAC * len(healthy))
    train = healthy[:n_train]
    test_h = healthy[n_train:]
    onset = int(np.flatnonzero(b['anomaly'].to_numpy() == 1)[0])
    return train, test_h, onset


def physics_monitor(b):
    I = physics_frame(b)
    train, test_h, onset = split(b)
    z, zs = {}, {}
    for k in PHYS + SUPPORT:
        x = I[k].to_numpy(float)
        if not np.isfinite(x[train]).sum() > 5:
            continue
        mu, sd = np.nanmean(x[train]), np.nanstd(x[train])
        zk = np.nan_to_num((x - mu) / sd)
        z[k] = zk
        zs[k] = ewma(zk)
    lim = {k: max(H_MIN, PHYS_SAFETY * np.quantile(SIGN[k] * zs[k][train], 0.995)) for k in PHYS if k in zs}
    alarm = {k: SIGN[k] * zs[k] > lim[k] for k in lim}
    any_alarm = np.any(np.vstack(list(alarm.values())), axis=0)
    t_det = first_persistent(any_alarm, test_h[0] if len(test_h) else onset)
    fa = float(np.mean(any_alarm[test_h])) if len(test_h) else np.nan
    t_det_fault = first_persistent(any_alarm, onset)
    iso = None
    if t_det_fault is not None:
        k_at = t_det_fault + PERSIST - 1
        cand = {k: SIGN[k] * zs[k][k_at] / lim[k] for k in alarm if alarm[k][k_at]}
        iso = FAULT_OF_IND[max(cand, key=cand.get)] if cand else None
    return dict(I=I, z=z, zs=zs, lim=lim, alarm=any_alarm, onset=onset, train=train, test_h=test_h,
                false_alarm=fa, first_alarm_after_train=t_det, t_det=t_det_fault, isolated=iso)


def pca_monitor(b, var=0.90):
    train, test_h, onset = split(b)
    cols = [c for c in RAW if np.isfinite(b[c].to_numpy(float)).all() and b[c].iloc[train].std() > 0]
    X = b[cols].to_numpy(float)
    mu, sd = X[train].mean(0), X[train].std(0)
    Z = (X - mu) / sd
    U, S, Vt = np.linalg.svd(Z[train], full_matrices=False)
    lam = S ** 2 / (len(train) - 1)
    k = int(np.searchsorted(np.cumsum(lam) / lam.sum(), var) + 1)
    P = Vt[:k].T
    T = Z @ P
    T2 = np.sum(T ** 2 / lam[:k], axis=1)
    E = Z - T @ P.T
    Q = np.sum(E ** 2, axis=1)
    sT2, sQ = ewma(T2), ewma(Q)
    lT2 = SAFETY * np.quantile(sT2[train], 0.995)
    lQ = SAFETY * np.quantile(sQ[train], 0.995)
    alarm = (sT2 > lT2) | (sQ > lQ)
    fa = float(np.mean(alarm[test_h])) if len(test_h) else np.nan
    t_det = first_persistent(alarm, onset)
    iso, top = None, None
    if t_det is not None:
        i = t_det + PERSIST - 1
        contrib = E[i] ** 2 if sQ[i] > lQ else (T[i] / np.sqrt(lam[:k])) @ P.T * Z[i]
        top = cols[int(np.argmax(np.abs(contrib)))]
        iso = FAULT_OF_RAW.get(top, 'unclear')
    return dict(cols=cols, k=k, T2=sT2, Q=sQ, lT2=lT2, lQ=lQ, alarm=alarm, onset=onset, false_alarm=fa,
                t_det=t_det, isolated=iso, top_channel=top)
