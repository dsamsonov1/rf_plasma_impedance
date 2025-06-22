import matplotlib

from rfd_conf import cf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def plot_UI():

    ##### Извлекаем интересующие токи и напряжения

    time_raw = np.array(cf['analysis'].time)

    Vpl_raw = getU('5', '0', cf['analysis'])  # Vpl
    Ipl_raw = getU('8', '9', cf['analysis']) / cf['Rp']  # Ipl
    V_R_rf_raw = getU('2', '1', cf['analysis'])  # Vrf
    Irf_raw = V_R_rf_raw / cf["val_R_rf"]  # Irf
    Vl_raw = getU('3', '0', cf['analysis'])  # Vl
    Il_raw = getU('4', '5', cf['analysis']) / cf["val_R_m"]  # Il
    Vs1_raw = getU('5', '7', cf['analysis'])  # Vs1
    Vs2_raw = getU('9', '10', cf['analysis'])  # Vs2
    V_R_rf_raw = getU('2', '1', cf['analysis'])  # Vrf
    VRm_raw = getU('4', '5', cf['analysis'])  # VRm
    VRstray_raw = getU('6', '0', cf['analysis'])  # VRm

    # Строим ток и напряжение на плазме (2 периода)

    #TODO разобраться с отображением и учетом в коде steady линии на графике
    first_steady_period = 400  # Номер периода, с которого считаем, что установившийся режим наступил

    time_2_last_periods = extract_N_periods(time_raw, 1, 2, 'rev')
    Vpl_2_last_periods = extract_N_periods(Vpl_raw, 1, 2, 'rev')
    Ipl_2_last_periods = extract_N_periods(Ipl_raw, 1, 2, 'rev')

    Vs1_2_last_periods = extract_N_periods(Vs1_raw, 1, 2)
    Vs2_2_last_periods = extract_N_periods(Vs2_raw, 1, 2)

    # Графики Upl, Ipl на одной картинке
    fig = plt.figure(figsize=(9, 10))
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Vpl [V]', color='tab:red')
#    ax1.spines['bottom'].set_position('zero')
    ax1.plot(time_2_last_periods, Vpl_2_last_periods, color='tab:red',
                label='Vpl')  # Показываем два _предпоследних_ периода, т.к.
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.spines['bottom'].set_position('zero')
    ax2.set_ylabel('Ipl [A]', color='tab:blue')  # we already handled the x-label with ax1
    ax2.plot(time_2_last_periods, Ipl_2_last_periods, color='tab:blue', label='Ipl')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend()
    plt.show()
    
    
    # Графики переходного процесса для контроля сходимости
    fig = plt.figure(figsize=(9, 10))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(time_raw / cf["Tf"], Vpl_raw, label='Vp', alpha=0.5)  # Обзорный график для определения установившегося режима
    ax1.plot(time_raw / cf["Tf"], Vs1_raw, label='Vs1', alpha=0.5)  # Обзорный график для определения установившегося режима
    ax1.plot(time_raw / cf["Tf"], Vs2_raw, label='Vs2', alpha=0.5)  # Обзорный график для определения установившегося режима
    ax1.set_ylabel('U [V]')
    ax1.axvline(first_steady_period, color='cyan', linestyle=':', label='steady state')
    ax1.legend()
    _ = ax1.set_xlabel('Periods count')
    plt.show()
    
    # Графики Vs1, Vs2 на одной картинке
    fig = plt.figure(figsize=(9, 10))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(time_2_last_periods, Vs1_2_last_periods, label='Vs1')
    ax1.plot(time_2_last_periods, Vs2_2_last_periods, label='Vs2')
    ax1.set_ylabel('U [V]')
    ax1.spines['bottom'].set_position('zero')
    ax1.legend()
    plt.show()

    # Спектры

    (_, Ipl_abs, Ipl_phase, freqs, freqsMHz, reduced_freqs, Ipl_reduced_abs, Ipl_reduced_angle, waste_freqs,
     Ipl_waste_abs, Ipl_waste_angle, true_freqs, Ipl_true_abs, Ipl_true_angle) = get_spectra(Ipl_raw)
    (_, Vpl_abs, Vpl_phase, _, _, _, Vpl_reduced_abs, Vpl_reduced_angle, _, Vpl_waste_abs, Vpl_waste_angle, _,
     Vpl_true_abs, Vpl_true_angle) = get_spectra(Vpl_raw)

    (_, _, _, _, _, _, _, _, _, _, _, _, Il_true_abs, Il_true_angle) = get_spectra(Irf_raw)
    (_, _, _, _, _, _, _, _, _, _, _, _, Vl_true_abs, Vl_true_angle) = get_spectra(Vl_raw)

    if cf["verbose_plots"]:
        # Обзорные спектры

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))

        Ipl_norm = np.linalg.norm(Ipl_abs)
        Vpl_norm = np.linalg.norm(Vpl_abs)
        Zpl_norm = Vpl_norm / Ipl_norm

        Ipl_normalized = Ipl_abs / Ipl_norm
        Vpl_normalized = Vpl_abs / Vpl_norm

        axs[0].semilogy(freqsMHz, Ipl_normalized, label='Ipl')
        axs[0].semilogy(freqsMHz, Vpl_normalized, label='Vpl')
        axs[0].set_xlabel('Frequency [MHz]')
        axs[0].set_title('Survey plasma amplitude spectrum')
        axs[0].legend()
        axs[1].plot(freqsMHz, Ipl_phase, label='Ipl')
        axs[1].plot(freqsMHz, Vpl_phase, label='Vpl')
        axs[1].set_xlabel('Frequency [MHz]')
        axs[1].set_title('Survey plasma phase spectrum')
        axs[1].set_ylim(bottom=-np.pi, top=np.pi)
        axs[1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ['-π', '-π/2', '0', 'π/2', 'π'])
        axs[1].grid(axis='y', linestyle=':')
        axs[1].legend()
        plt.show()

    if cf["verbose_plots"]:
        # Укрупненные спектры (первые nHarm гармоник)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))

        axs[0].stem(true_freqs, Ipl_true_abs, linefmt='orange', markerfmt='C1o', label='Ipl', basefmt=' ')
        axs[0].stem(true_freqs, Vpl_true_abs, linefmt='orange', markerfmt='C2o', label='Vpl', basefmt=' ')
        axs[0].stem(waste_freqs, Ipl_waste_abs, linefmt=':', basefmt=' ')
        axs[0].stem(waste_freqs, Vpl_waste_abs, linefmt=':', basefmt=' ')
        axs[0].set_yscale('log')
        axs[0].set_xlabel('Frequency [MHz]')
        axs[0].set_ylabel('Amplitude [a.u.]')
        axs[0].set_title('Plasma amplitude spectrum')
        axs[0].legend()
        
        axs[1].stem(true_freqs, Vpl_true_angle, linefmt='orange', markerfmt='C1o', label='Vpl', basefmt=' ')
        axs[1].stem(true_freqs, Ipl_true_angle, linefmt='orange', markerfmt='C2o', label='Ipl', basefmt=' ')
        axs[1].stem(waste_freqs, Ipl_waste_angle, linefmt=':', basefmt=' ')
        axs[1].stem(waste_freqs, Vpl_waste_angle, linefmt=':', basefmt=' ')
        axs[1].set_ylabel('Phase [rad]')
        axs[1].set_xlabel('Frequency [MHz]')
        axs[1].set_title('Plasma phase spectrum')
        axs[1].set_ylim(bottom=-1.1 * np.pi, top=1.1 * np.pi)
        axs[1].set_yticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                            ['-π', '-3π/4', '-π/2', '-π/4', '0', 'π/4', 'π/2', '3π/4', 'π'])
        axs[1].grid(axis='y', linestyle=':')
        axs[1].legend()

    # Спектры Upl, Ipl на одной картинке
    barWidth = 6
    fig = plt.figure(figsize=(9, 10))
    axs = fig.add_subplot(1, 1, 1)
    br2 = [x + barWidth for x in true_freqs]
    bars = axs.bar(true_freqs, Vpl_true_abs, width=barWidth, label='Vpl [V]')
    # Iterate and add text labels
    for bar in bars:
        height = bar.get_height()
        axs.text(bar.get_x() + bar.get_width() / 2., height, '%.2f' % height, ha='center', va='bottom')

    bars = axs.bar(br2, Ipl_true_abs, width=barWidth, label='Ipl [A]')
    # Iterate and add text labels
    for bar in bars:
        height = bar.get_height()
        axs.text(bar.get_x() + bar.get_width() / 2., height, '%.2f' % height, ha='center', va='bottom')
    axs.set_yscale('log')
    axs.set_ylabel('Amplitude [a.u.]')
    axs.set_xlabel('Frequency [MHz]')
    axs.legend()
    _ = axs.set_xticks([x + 0.5 * barWidth for x in true_freqs], np.round(true_freqs, 2))
    plt.show()

'''    
    # Спектр импеданса
    fig = plt.figure(figsize=(9, 10))
    axs = fig.add_subplot(1, 1, 1)
    axs.bar(true_freqs, Vpl_true_abs / Ipl_true_abs, width=barWidth)
    _ = axs.set_xticks(np.round(true_freqs, 2))
    axs.set_yscale('log')
    axs.set_ylabel('Abs impedance [Ohm]')
    _ = axs.set_xlabel('Frequency [MHz]')
    plt.show()
'''

def plot_sweepResult(a_df):
    fig = plt.figure(figsize=(9, 18))
    gs = gridspec.GridSpec(5, 3, hspace=0)
    #axs = gs.subplots(sharex=True)
    a_df.plot(ax=fig.add_subplot(gs[0]), x='p0 [Pa]', y=['Pp [W]', 'PRm [W]', 'PRstray [W]'], marker='x')
    a_df.plot(ax=fig.add_subplot(gs[1]), x='p0 [Pa]', y=['Ubias [V]', 'Urf [V]', 'Vs1 [V]', 'Vs2 [V]'], marker='x')
    a_df.plot(ax=fig.add_subplot(gs[2]), x='p0 [Pa]', y='ne [m^-3]', marker='x')
    a_df.plot(ax=fig.add_subplot(gs[3]), x='p0 [Pa]', y='Te [eV]', marker='x')
    a_df.plot(ax=fig.add_subplot(gs[4]), x='p0 [Pa]', y=['Iion1 [A]', 'Iion2 [A]'], marker='x')
    a_df.plot(ax=fig.add_subplot(gs[5]), x='p0 [Pa]', y=['C1 [pF]', 'C2 [pF]'], marker='x')
    a_df.plot(ax=fig.add_subplot(gs[6]), x='p0 [Pa]', y=['Re(Zl) [Ohm]'], marker='x')
    a_df.plot(ax=fig.add_subplot(gs[7]), x='p0 [Pa]', y=['Im(Zl) [Ohm]'], marker='x')
    a_df.plot(ax=fig.add_subplot(gs[8]), x='p0 [Pa]', y=['Re(Zp) [Ohm]'], marker='x')
    a_df.plot(ax=fig.add_subplot(gs[9]), x='p0 [Pa]', y=['Im(Zp) [Ohm]'], marker='x')
    a_df.plot(ax=fig.add_subplot(gs[10]), x='p0 [Pa]', y=['jIon1 [uA/cm^2]', 'jIon2 [uA/cm^2]'], marker='x')
    fig.suptitle(f'{cf['name']}: Ar f0={cf["f0"]/1e6:.2f} [MHz] L={cf["l_B"]/1e-2:.2f} [cm] Ae={cf["Ae"]*1e4:.2f} [cm^2] Ag={cf["Ag"]*1e4:.2f} [cm^2] P0={cf["P0"]:.1f} [W]')
    plt.tight_layout()
    plt.show()
    fig.savefig(f'{cf['out_path']}/{cf['next_aaaa']:04d}_{cf['name']}_{cf['current_date']}_sweep.png')
