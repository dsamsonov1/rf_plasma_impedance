from rfd_conf import cf
import numpy as np
import matplotlib.pyplot as plt
from rfd_utils import *

def plot_Te():
    en_range = np.arange(1, 7.1, 0.1)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    minor_ticks = np.arange(0, 101, 4)
    ax.set_xticks(minor_ticks, minor=True)
    plt.plot(en_range, cf["Vp"] * cf["ng"] * Kiz(en_range), label='Kiz side')
    plt.plot(en_range, (cf["Ae"] + cf["Ag"]) * u_Bohm(en_range), label='Bohm side')
    plt.axvline(cf["Te"], color='cyan', linestyle=':')
    plt.grid(which='minor', linestyle='--')
    plt.grid(which='major', linestyle='--')
    plt.minorticks_on()
    plt.yscale('log')
    plt.xlabel('Temperature [eV]')
    plt.ylabel('Particle balance [m^3/s]')
    plt.legend()
    plt.show()

def plot_K():
    en_range = np.arange(1, 7.1, 0.1)

    plt.figure(figsize=(12, 5))
    plt.plot(en_range, (Kel(en_range) * cf["eps_el"]) / Kiz(en_range), linestyle=':', label='elastic')
    plt.plot(en_range, cf["eps_iz"] * np.ones(en_range.size), linestyle=':', label='ionization')
    plt.plot(en_range, (Kex(en_range) * cf["eps_ex"]) / Kiz(en_range), linestyle=':', label='excitation')
    plt.plot(en_range, eps_c(en_range), label='total')
    plt.yscale('log')
    plt.xlabel('Temperature [eV]')
    plt.ylabel('Ionization cost [eV]')
    plt.legend()
    plt.grid()
    plt.show()


def plot_UI(a_analysis, a_Rp):

    ##### Извлекаем интересующие токи и напряжения

    time_raw = np.array(a_analysis.time)

    Vpl_raw = getU('5', '0', a_analysis)  # Vpl
    Ipl_raw = getU('8', '9', a_analysis) / a_Rp  # Ipl
    V_R_rf_raw = getU('2', '1', a_analysis)  # Vrf
    Irf_raw = V_R_rf_raw / cf["val_R_rf"]  # Irf
    Vl_raw = getU('3', '0', a_analysis)  # Vl
    Il_raw = getU('4', '5', a_analysis) / cf["val_R_m"]  # Il
    Vs1_raw = getU('5', '7', a_analysis)  # Vs1
    Vs2_raw = getU('9', '10', a_analysis)  # Vs2
    V_R_rf_raw = getU('2', '1', a_analysis)  # Vrf
    VRm_raw = getU('4', '5', a_analysis)  # VRm
    VRstray_raw = getU('6', '0', a_analysis)  # VRm

    # Строим ток и напряжение на плазме (2 периода)

    first_steady_period = 400  # Номер периода, с которого считаем, что установившийся режим наступил

    time_2_last_periods = extract_N_periods(time_raw, 1, 2, 'rev')
    Vpl_2_last_periods = extract_N_periods(Vpl_raw, 1, 2, 'rev')
    Ipl_2_last_periods = extract_N_periods(Ipl_raw, 1, 2, 'rev')

    fig, ax1 = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

    ax1[0].set_xlabel('time (s)')
    ax1[0].set_ylabel('Vpl [V]', color='tab:red')
    ax1[0].plot(time_2_last_periods, Vpl_2_last_periods, color='tab:red',
                label='Vpl')  # Показываем два _предпоследних_ периода, т.к.
    ax1[0].tick_params(axis='y', labelcolor='tab:red')
    ax2 = ax1[0].twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Ipl [A]', color='tab:blue')  # we already handled the x-label with ax1
    ax2.plot(time_2_last_periods, Ipl_2_last_periods, color='tab:blue', label='Ipl')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1[0].legend()
    ax1[1].plot(time_raw / cf["Tf"], Vpl_raw)  # Обзорный график для определения установившегося режима
    ax1[1].axvline(first_steady_period, color='cyan', linestyle=':', label='steady state')
    ax1[1].legend()
    _ = ax1[1].set_xlabel('Periods count')
    plt.show()