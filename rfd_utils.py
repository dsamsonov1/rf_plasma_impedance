import math
import numpy as np
from PySpice.Spice.Netlist import Circuit
from scipy.fft import rfft, rfftfreq
from scipy.integrate import trapezoid
from rfd_conf import *
from scipy import optimize
import pandas as pd
import os
import re
from datetime import datetime
import sys
import matplotlib.pyplot as plt

##############
# Определение вспомогательных функций
##############

def Kex_novec(a_Te):
    res = 5.02e-15 * np.exp(-12.64 / a_Te)  # 3p1 11.5 eV
    return res


Kex = np.vectorize(Kex_novec, otypes=[float])


def Kel_novec(a_Te):
    lnkel = -31.3879 + 1.6090 * np.log(a_Te) + 0.0618 * np.log(a_Te) ** 2 - 0.1171 * np.log(a_Te) ** 3
    # print(f'lnkel = {lnkel}')
    res = np.exp(lnkel)
    #    res = lnkel
    return res


Kel = np.vectorize(Kel_novec, otypes=[float])


def Kiz_novect(a_Te):
    res = 2.34e-14 * a_Te ** 0.59 * np.exp(-17.44 / a_Te)  # Хорошо совпадает с [Schmidt]
    #    res = 2.9e-14*Te**0.50*np.exp(-17.8/Te)
    #    res = 7.93e-13*np.exp(-18.9/Te)      # Плохо совпадает с [Schmidt]
    return res


Kiz = np.vectorize(Kiz_novect, otypes=[float])


def u_Bohm_novect(a_Te):
    return np.sqrt(ct["qe"] * a_Te / ct["Mi"])


u_Bohm = np.vectorize(u_Bohm_novect, otypes=[float])


def get_mean(a_T, a_V, a_N):
    # print(f'VERBOZE: a_T={a_T} a_V={a_V} a_N={a_N} Tf={Tf} trap={trapezoid(x=a_T, y=a_V)}')
    return 1 / (a_N * cf["Tf"]) * trapezoid(x=a_T, y=a_V)


#    return abs(1/(a_N*Tf) * trapezoid(x=a_T, y=a_V))

def extract_N_periods(array, start, periods_count, direction='rev'):
    if direction == 'fwd':
        return array[start * cf["sim_periods_div"]:(start + periods_count) * cf["sim_periods_div"]]
    elif direction == 'rev':
        return array[-(start + periods_count) * cf["sim_periods_div"]:-start * cf["sim_periods_div"]]


def getU(a_n1, a_n2, a_an):
    if a_n2 == '0':
        return np.array(a_an[a_n1])
    else:
        return np.array(a_an[a_n1]) - np.array(a_an[a_n2])


def ang_normalize_novec(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


ang_normalize = np.vectorize(ang_normalize_novec, otypes=[float])


def get_spectra(a_Data, a_nHarm=10, num_periods_for_spectra=10):
    data_cut = extract_N_periods(a_Data, cf["first_steady_period"], num_periods_for_spectra, 'fwd')
    _spectra = rfft(data_cut) / (num_periods_for_spectra * cf["sim_periods_div"])
    _freqs = rfftfreq(data_cut.size, d=cf["Tf"] / cf["sim_periods_div"])

    freqsMHz = _freqs / 1e6

    full_abs = np.abs(_spectra)
    full_angle = np.angle(_spectra)

    # Индекс в массиве результатов fft, чтобы вырезать (немного больше) N гармоник
    idxHarm = (a_nHarm + 1) * num_periods_for_spectra
    
    reduced_spectra = _spectra[0:idxHarm]

    reduced_freqs = freqsMHz[0:idxHarm]
    reduced_abs = np.abs(reduced_spectra)[0:idxHarm]
    reduced_angle = np.angle(reduced_spectra)[0:idxHarm]

    idxrange = np.arange(start=num_periods_for_spectra,
                         stop=(a_nHarm + 1) * num_periods_for_spectra,
                         step=num_periods_for_spectra)

    waste_freqs = np.delete(reduced_freqs, idxrange)  # "Мусорные" гармоники, лежащие между кратными f0
    waste_abs = np.delete(reduced_abs, idxrange)
    waste_angle = np.delete(reduced_angle, idxrange)

    true_freqs = reduced_freqs[idxrange]  # Гармоники с частотами, кратными f0
    true_abs = reduced_abs[idxrange]
    true_angle = reduced_angle[idxrange]

    return (_spectra, full_abs, full_angle, _freqs, freqsMHz, reduced_freqs, reduced_abs, reduced_angle, waste_freqs,
        waste_abs, waste_angle, true_freqs, true_abs, true_angle)


# Пытаемся найти Te как корень баланса частиц Vp*ng*Kiz(Te)-(Ae + Ag)*u_Bohm(Te) на отрезке [1-7] eV
def dfr(a_Te):
    return cf["Vp"] * cf["ng"] * Kiz_novect(a_Te) - (cf["Ae"] + cf["Ag"]) * u_Bohm_novect(a_Te)


def eps_c_novec(a_Te):
    return cf["eps_iz"] + (Kel(a_Te) * cf["eps_el"] + Kex(a_Te) * cf["eps_ex"]) / Kiz(a_Te)


eps_c = np.vectorize(eps_c_novec, otypes=[float])

##########
# 7. Определяем и считаем цепь
##########

def calcCircuit():
    
    def check_steady_state(time, output, threshold):
        """
        Check if the output voltage has reached steady state by comparing
        the amplitude variation in the last period.
        """
        # Find indices for the last complete period
        time_period = cf['Tf']*cf["num_periods_sim"]  # 50 Hz signal
        mask_current = (time >= time_period*period) & (time <= time_period*(period+1))
        mask_previous = (time >= time_period*(period-1)) & (time <= time_period*period)
        
        if not np.any(mask_current):
            return False  # Not enough data to check
        
        current_period_voltage = np.mean(output[mask_current])
        previous_period_voltage = np.mean(output[mask_previous])
        
        relative_variation = np.abs(1 - (current_period_voltage / previous_period_voltage))
        
        if cf['verbose_circuit']:
            print(f"var: {relative_variation:.2f} (1: {previous_period_voltage:.2f}V, 2: {current_period_voltage:.2f}V)", end = ' ')
        
        if cf['verbose_circ_plots']:
            plt.figure(figsize=(12, 5))
            plt.plot(time[mask_current], output[mask_current])
            plt.plot(time[mask_previous], output[mask_previous])
    
            plt.text(0.02, 0.98, f'step #{period}', 
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     horizontalalignment='left',
                     bbox=dict(facecolor='white', alpha=0.5))
        
            plt.show()
        
        return relative_variation < threshold
    

    if cf["verbose_circuit"]:
        print(f' - VERBOSE: Lp={cf["Lp"] * 1e9:.2f} [nH] Rp={cf["Rp"]:.2f} [Ohm] alpha={cf["alpha"]:.2e}')
        print(f' - VERBOSE: Be_e={cf["Ie01"]:.3f} Bi_e={cf["Iion1"]:.2e} Cs1={np.sqrt(cf["CCs1"]):.2e}/sqrt(Vs_e(t))')
        print(f' - VERBOSE: Be_g={cf["Ie02"]:.3f} Bi_g={cf["Iion2"]:.2e} Cs2={np.sqrt(cf["CCs2"]):.2e}/sqrt(Vs_g(t))')

    circuit = Circuit('RF discharge impedance')
    circuit.SinusoidalVoltageSource('V0', 1, 0, amplitude=cf["Vm"], frequency=cf["f0"])
    # Фаза результатов сдвинута относительно [Schmidt], т.к. там cos, а тут sin
    # TODO: найти способ обойти ограничение PySpice на задание именно COS источника
    # (сам ngspice это позволяет)

    circuit.R('Rrf', 1, 2, cf["val_R_rf"])
    circuit.C('Cm1', 2, 0, cf["val_C_m1"])
    circuit.C('Cm2', 2, 3, cf["val_C_m2"])
    circuit.L('Lm2', 3, 4, cf["val_L_m2"])
    #    circuit.L('LL', 5, 0, 10e-9)
    circuit.R('Rm', 4, 5, cf["val_R_m"])
    circuit.C('Cstray', 5, 6, cf["val_C_stray"])
    circuit.R('Rstray', 6, 0, cf["val_R_stray"])
    circuit.BehavioralSource('Be_e', 5, 7, current_expression=f'v(7,5) > 0 ? {cf["Ie01"]}*exp({cf["alpha"]}*v(7,5)) : 1e-15')
    circuit.CurrentSource('Bi_e', 7, 5, cf["Iion1"])
    circuit.BehavioralCapacitor('Cs1', 7, 5, capacitance_expression=f'C=\'sqrt({cf["CCs1"]}/abs(v(7,5)))\'')
    circuit.L('L_p', 7, 8, cf["Lp"])
    circuit.R('R_p', 8, 9, cf["Rp"])
    circuit.BehavioralSource('Be_g', 10, 9, current_expression=f'v(9,10) > 0 ? {cf["Ie02"]}*exp({cf["alpha"]}*v(9,10)) : 1e-15')
    circuit.CurrentSource('Bi_g', 9, 10, cf["Iion2"])
    circuit.BehavioralCapacitor('Cs2', 9, 10, capacitance_expression=f'C=\'sqrt({cf["CCs2"]}/abs(v(9,10)))\'')
    circuit.VoltageSource('Viz', 10, 0, 0)
#    circuit.LosslessTransmissionLine('TL', 11, 0, 0, 0, impedance=50, frequency=cf["f0"], normalized_length=0.35)
#    circuit.SingleLossyTransmissionLine('TL', 5, 0, 0, 0, model='ymod', length=1, raw_spice='\n.MODEL ymod txl R=0.1 L=8.972e-9 G=0 C=0.468e-12 length=22.12')
#    circuit.LossyTransmission('TL', 5, 0, 0, 0, model='LOSSYMOD', raw_spice='\n.model LOSSYMOD ltra rel=2 r=0.1 g=0 l=8.972e-9 c=0.468e-12 len=12 nosteplimit compactrel=1.0e-3 compactabs=1.0e-14')
#    circuit.LossyTransmission('TL', 5, 0, 0, 0, model='LOSSYMOD', raw_spice='\n.model LOSSYMOD ltra rel=1 r=1 l=8.972e-9 c=0.468e-12 len=5.53m compactrel=1.0e-2 compactabs=1.0e-8')

    if cf["cooling"]:
        circuit.R('R_rl', 5, 11, 0.01)
        circuit.L('Lmx', 11, 0, 5e-6)

    simulator = circuit.simulator()
#    simulator = circuit.simulator(simulator='ngspice-shared')

    # Надо задать какие-то около-, но ненулевые НУ,
    # т.к. иначе pyspice ломается на предупреждениях от Ngspice
    simulator._initial_condition = {'v(5)': 1e-10, 'v(9)': 1e-10}

    # print(simulator) # Можно напечатать .IC для проверки
    if cf['verbose_circuit']:
        print(circuit) # Можно напечатать получившийся netlist для проверки

    # Сформировать строку с ngspice netlist для отчета
    cf['sim_circ'] = str(circuit)

    # Run simulation in segments, checking for steady state
    current_time = 0
    period = 0
    all_time = np.array([])
    all_input = np.array([])
    all_output = np.array([])
    steady_state_reached = False
    max_periods = 100

    # Simulation parameters
    end_time = max_periods*cf["tmax_sim"]
    check_interval = 10*cf['Tf']  # Interval between steady-state checks
    steady_state_threshold = 0.005  # 1% change considered steady
    
    
    while current_time < end_time or period < max_periods:
        next_time = min(current_time + cf["tmax_sim"], end_time)
#        print(f"Circ #{period}: {current_time/cf['Tf']:.1f}-{next_time/cf['Tf']:.1f}", end=' ')
        
        # Run transient simulation for this segment
        #    print(f'Ts {cf["Tf"] / 100:.2e} Te {cf["tmax_sim"]:.2e}')
        analysis = simulator.transient(step_time=cf["Tf"] / cf['sim_periods_div'], end_time=next_time)
        
        time_segment = np.array(analysis.time)
        output_segment = np.array(analysis['5'])-np.array(analysis['7'])  # Voltage at 'out' node
        
        # Check for steady state (after first segment)
        if current_time > 0:
            steady_state_reached = check_steady_state(np.array(analysis.time), np.array(analysis['5'])-np.array(analysis['7']), 
                                                    steady_state_threshold)
            if steady_state_reached:
#                print("+", end='\n')
                cf['analysis'] = analysis
                break
        
        current_time = next_time
        
        period = period+1
        
        if period >= max_periods:
            sys.exit("CIRCUIT STEADY STATE NOT REACHED. PERIODS LIMIT REACHED. STOP.")


def plot_UI2(a_iter=0):

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
    first_steady_period = cf["num_periods_sim"]-cf["num_periods_for_integration"]  # Номер периода, с которого считаем, что установившийся режим наступил

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
    
    if a_iter > 0:
        plt.text(0.02, 0.98, f'ne iteration #{a_iter}', 
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 horizontalalignment='left',
                 bbox=dict(facecolor='white', alpha=0.5))
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

    if a_iter > 0:
        plt.text(0.02, 0.98, f'ne iteration #{a_iter}', 
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 horizontalalignment='left',
                 bbox=dict(facecolor='white', alpha=0.5))
    
    plt.show()
    

    # Графики Vs1, Vs2 на одной картинке
    fig = plt.figure(figsize=(9, 10))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(time_2_last_periods, Vs1_2_last_periods, label='Vs1')
    ax1.plot(time_2_last_periods, Vs2_2_last_periods, label='Vs2')
    ax1.set_ylabel('U [V]')
    ax1.spines['bottom'].set_position('zero')
    ax1.legend()

    if a_iter > 0:
        plt.text(0.02, 0.98, f'ne iteration #{a_iter}', 
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 horizontalalignment='left',
                 bbox=dict(facecolor='white', alpha=0.5))
    
    plt.show()


    # Спектры

    (_, Ipl_abs, Ipl_phase, freqs, freqsMHz, reduced_freqs, Ipl_reduced_abs, Ipl_reduced_angle, waste_freqs,
     Ipl_waste_abs, Ipl_waste_angle, true_freqs, Ipl_true_abs, Ipl_true_angle) = get_spectra(Ipl_raw)
    (_, Vpl_abs, Vpl_phase, _, _, _, Vpl_reduced_abs, Vpl_reduced_angle, _, Vpl_waste_abs, Vpl_waste_angle, _,
     Vpl_true_abs, Vpl_true_angle) = get_spectra(Vpl_raw)

    (_, _, _, _, _, _, _, _, _, _, _, _, Il_true_abs, Il_true_angle) = get_spectra(Irf_raw)
    (_, _, _, _, _, _, _, _, _, _, _, _, Vl_true_abs, Vl_true_angle) = get_spectra(Vl_raw)

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

    if a_iter > 0:
        plt.text(0.02, 0.98, f'ne iteration #{a_iter}', 
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 horizontalalignment='left',
                 bbox=dict(facecolor='white', alpha=0.5))
    
    plt.show()


##########
# 8. Делаем итерации по ne
##########

def calc_dischargePoint():
    
    ne_new = -2 * cf["eps_ne"]
    iter_no = 0
    
    while np.abs(cf["ne"] - ne_new) > cf["eps_ne"]:
    
        if iter_no > 0:
            cf["ne"] = ne_new
    
        if iter_no < cf["max_iter_ne"]:
            iter_no = iter_no + 1
    
            print(f'  -- ne #{iter_no: =2}:', end=' ')
    
            redefineCircuitParameters()
            calcCircuit()
            
            if cf["verbose_plots"]:
                plot_UI2(iter_no)
                
            calcPlasmaQuantities(postprocess=False)
    
            ##############
            # Определение ne
            ##############
    
            Pguess = cf["ne"] * cf["Vp"] * cf["ng"] * Kiz(cf["Te"]) * ct["qe"] * \
                     (eps_c(cf["Te"]) + cf["eps_e"] + cf["fE"] * np.abs(cf["_Vs1"]) + cf["fG"] * np.abs(cf["_Vs2"]) + cf["Te"] / 2)
    
            ne_new = cf["Ppl"] / (cf["Vp"] * cf["ng"] * Kiz(cf["Te"]) * ct["qe"] * \
                            (eps_c(cf["Te"]) + cf["eps_e"] + cf["fE"] * np.abs(cf["_Vs1"]) + cf["fG"] * np.abs(cf["_Vs2"]) + cf["Te"] / 2))
    
            # "регуляризация" для улучшения сходимости итераций ne (suggested by G. Marchiy)
            ne_new = cf["ne"] + (ne_new - cf["ne"]) * cf["beta"]
    
            print(f"Ppl={cf['Ppl']:.2f} [W], Pguess={Pguess:.2f} [W], ne={cf['ne']:.2e} [m^-3], ne_new={ne_new:.2e} [m^-3] dne={ne_new - cf['ne']:.2e}", end=' ')
    
            if np.abs(ne_new - cf["ne"]) < cf["eps_ne"]:
                print(f'<- ne CONVERGED', end='\n')
                return cf['gamma']
    
            else:
                print(f'--', end='\n')
    
        else:
            sys.exit("ne NOT CONVERGED. ne ITERATIONS LIMIT REACHED. STOP.")


##############
# Определение вспомогательных функций
##############

def calcImpedance_1Harm(a_u1, a_U2, a_I1, a_I2, a_Rval):
    Vl_raw = getU(a_u1, a_U2, cf["analysis"])  # Vl
    Il_raw = getU(a_I1, a_I2, cf["analysis"]) / a_Rval  # Il

    Vl_2_last_periods = extract_N_periods(Vl_raw, 1, 2)
    Il_2_last_periods = extract_N_periods(Il_raw, 1, 2)

    spectraVl = rfft(Vl_2_last_periods) / (2 * cf["sim_periods_div"])
    spectraIl = rfft(Il_2_last_periods) / (2 * cf["sim_periods_div"])
    freqsl = rfftfreq(Vl_2_last_periods.size, d=cf["Tf"] / cf["sim_periods_div"])

    UU = abs(spectraVl)[2]
    II = abs(spectraIl)[2]

    Ua = np.angle(spectraVl)[2]
    Ia = np.angle(spectraIl)[2]

    RL = UU / II * np.cos(Ua - Ia)
    #    XL=UU/II*np.sin(Ua-Ia)+2*np.pi*config["f0"]*1500e-9
    XL = UU / II * np.sin(Ua - Ia)

    return RL, XL


def calcVoltage_Harm(a_u1, a_U2, a_harm, a_realflag=True):
    Vl_raw = getU(a_u1, a_U2, cf["analysis"])  # Vl

    Vl_2_last_periods = extract_N_periods(Vl_raw, 1, 2)

    spectraVl = rfft(Vl_2_last_periods) / (2 * cf["sim_periods_div"])
    # freqsl = rfftfreq(Vl_2_last_periods.size, d=cf["Tf"] / cf["sim_periods_div"])

    if a_realflag:
        return np.real(spectraVl[a_harm])
    else:
        return np.abs(spectraVl[a_harm])


# Расчет элементов согласующего Г-образного четырехполюсника (см. сербов)
def calcMatchingNetwork(a_RL, a_XL, a_w, a_Z0):
    #    #top=1 LC=2 см. сербов
    #    X1 = -a_XL + np.sqrt(a_RL * (a_Z0 - a_RL))
    #    X2 = -(a_Z0 * a_RL) / (a_XL + X1)
    #    L1 = X1 / a_w
    #    C2 = -1 / (a_w * X2)
    #    return (L1, C2)

    # top=1 LC=4 см. сербов
    X1 = -a_XL + np.sqrt(a_RL * (a_Z0 - a_RL))
    X2 = -(a_Z0 * a_RL) / (a_XL + X1)
    C1 = -1 / (a_w * X1)
    C2 = -1 / (a_w * X2)

    return (C1, C2)


def cmn2(a_RL, a_XL, a_w):
    C1 = np.sqrt((50 / a_RL - 1) / (a_w * 50) ** 2)
    C2 = 0

    return (C1, C2)

def calcPower_Mean(a_U1, a_U2, a_I1, a_I2, a_Rval, total=True):

    V_raw = getU(a_U1, a_U2, cf["analysis"])  # Vpl
    I_raw = getU(a_I1, a_I2, cf["analysis"]) / a_Rval  # Ipl

    t_integration = extract_N_periods(np.array(cf["analysis"].time), 1, cf["num_periods_for_integration"])
    # data[-(num_periods_for_integration+1)*sim_periods_div:-1*sim_periods_div]

    V_integration = extract_N_periods(V_raw, 1, cf["num_periods_for_integration"])
    I_integration = extract_N_periods(I_raw, 1, cf["num_periods_for_integration"])
    P_integration = np.multiply(V_integration, I_integration)

    return get_mean(t_integration, P_integration, cf["num_periods_for_integration"])

def calcVoltage_Mean(a_U1, a_U2):
    
    t_integration = extract_N_periods(np.array(cf["analysis"].time), 1, cf["num_periods_for_integration"])
    V_raw = getU(a_U1, a_U2, cf["analysis"])  # Vs1
    V_integration = extract_N_periods(V_raw, 1, cf["num_periods_for_integration"])
    return get_mean(t_integration, V_integration, cf["num_periods_for_integration"])


def calcPlasmaQuantities(postprocess=False):
    
    cf["Ppl"] = calcPower_Mean('5', '0', '8', '9', cf['Rp'])
    
    if cf["Ppl"] < 0:
        sys.exit("Negative plasma power. STOP.")
        
    cf["_Vs1"] = calcVoltage_Mean('5', '7')
    cf["_Vs2"] = calcVoltage_Mean('9', '10')
    (Rii, Xii) = calcImpedance_1Harm('2', '0', '1', '2', cf["val_R_rf"])
    Zi = complex(Rii, Xii)
    cf["gamma"] = abs((Zi-cf["val_R_rf"])/(Zi+cf["val_R_rf"]))    

    if postprocess:
        P_R_m = calcPower_Mean('4', '5', '4', '5', cf['val_R_m'])
        P_R_stray = calcPower_Mean('6', '0', '6', '0', cf['val_R_stray'])
        (Rpl, Xpl) = calcImpedance_1Harm('5', '0', '8', '9', cf['Rp'])
        (Rll, Xll) = calcImpedance_1Harm('5', '0', '4', '5', cf["val_R_m"])
        Ubias = calcVoltage_Harm('5', '0', 0, a_realflag=True)
        Urf = calcVoltage_Harm('5', '0', 2, a_realflag=False)

        if cf["verbose_plots"]:
            print(f"  - VERBOZE: Vs_e={np.abs(cf['_Vs1']):.2f} [V] Vs_g={np.abs(cf['_Vs2']):.2f} [V]")

        cf["pd"] = pd.DataFrame({'p0 [Pa]': [cf["p0"]], 'f0 [MHz]': [cf["f0"]/1e6], 'ne [m^-3]': [cf["ne"]], 'Te [eV]': [cf["Te"]],
                                 'C1 [pF]': [cf["val_C_m1"]/1e-12], 'C2 [pF]': [cf["val_C_m2"]/1e-12],
                                 'L1 [nH]': [cf["val_L_m2"]/1e-9], 'P0 [W]': [cf["P0"]],
                                 'Vp [m^3]': [cf["Vp"]], 'ng [m^-3]': [cf["ng"]], 'Kiz []': [Kiz(cf["Te"])], 'eps_c []': [eps_c(cf["Te"])],
                                 'eps_e []': [cf["eps_e"]], 'fE []': [cf["fE"]], 'fG []': [cf["fG"]], 'Ae [m^2]': [cf["Ae"]], 'Ag [m^2]': [cf["Ag"]],
                                 'L_bulk [m]': [cf["l_B"]], 'T0 [K]': [cf["T0"]], 'Rrf [Ω]': [cf['val_R_rf']], 'Rm [Ω]': [cf['val_R_m']],
                                 'Cstray [pF]': [cf['val_C_stray']], 'Rstray [Ω]': [cf['val_R_stray']], "Ie01 [A]": [cf["Ie01"]],
                                 "alpha []": [cf["alpha"]], "Iion1 [A]": [cf["Iion1"]], "CCs1 []": [cf["CCs1"]], "Lp [nH]": [cf["Lp"]/1e-9],
                                 'Rp [Ω]': [cf['Rp']], 'Ie02 [A]': [cf['Ie02']], 'Iion2 [A]': [cf['Iion2']], 'CCs2 []': [cf['CCs2']],
                                 'Vm [V]': [cf['Vm']], 'jIon1 [uA/cm^2]': [cf['Iion1']*1e6/(cf['Ae']*1e4)], 'jIon2 [uA/cm^2]': [cf['Iion2']*1e6/(cf['Ag']*1e4)],
                                 'Re(Zi) [Ohm]': [Rii], 'Im(Zi) [Ohm]': [Xii], 'Re(Zl) [Ohm]': [Rll], 'Im(Zl) [Ohm]': [Xll], 'Re(Zp) [Ohm]': [Rpl], 'Im(Zp) [Ohm]': [Xpl],
                                 'Pp [W]': [cf['Ppl']], 'PRm [W]': [P_R_m], 'PRstray [W]': [P_R_stray], 'Ptot [W]': [cf['Ppl'] + P_R_m + P_R_stray],
                                 'Ubias [V]': [Ubias], 'Urf [V]': [Urf], 'Vs1 [V]': [cf['_Vs1']], 'Vs2 [V]': [cf['_Vs2']],
                                 'G [1]': [cf['gamma']], 'G^2 [1]': [cf['gamma']**2]})


def printSimulationResults():
    calcPlasmaQuantities(postprocess=True)
    print(f"Ppl={cf['pd']['Pp [W]'].values[0]:.2f} [W], PRm={cf['pd']['PRm [W]'].values[0]:.2f} [W], PRstray={cf['pd']['PRstray [W]'].values[0]:.2f} [W], TOTAL={cf['pd']['Ptot [W]'].values[0]:.2f} [W]")
    print(f"Zi=({cf['pd']['Re(Zi) [Ohm]'].values[0]:.2f}, {cf['pd']['Im(Zi) [Ohm]'].values[0]:.2f}) [Ohm], Zl=({cf['pd']['Re(Zl) [Ohm]'].values[0]:.2f}, {cf['pd']['Im(Zl) [Ohm]'].values[0]:.2f}) [Ohm], Zp=({cf['pd']['Re(Zp) [Ohm]'].values[0]:.2f}, {cf['pd']['Im(Zp) [Ohm]'].values[0]:.2f}) [Ohm]")
    print(f"G={cf['pd']['G [1]'].values[0]:.2f}, G^2={cf['pd']['G^2 [1]'].values[0]:.2f}")
    print(f'=== SIMULATION COMPLETE ===\n')


def redefineRuntimeParams():

    cf["ne"] = cf["ne_init"] 

    #################
    # 3. Вычисляемые величины №1
    #################

    cf["ng"] = cf["p0"] / (ct["k_B"] * cf["T0"])  # Концентрация буферного газа [м^-3]
    cf["Vp"] = cf["Ae"] * cf["l_B"]  # Объем плазменного столба
    cf["fE"] = cf["Ae"] / (cf["Ae"] + cf["Ag"])  # Весовой коэф. слоя E
    cf["fG"] = cf["Ag"] / (cf["Ae"] + cf["Ag"])  # Весовой коэф. слоя G
    cf["Tf"] = 1 / cf["f0"]  # Период ВЧ поля [с]

    #################
    # 4. Настройки анализа Ngspice и алгоритма
    #################

#   Теперь берется из конф. файла модели и sweep
#    cf["num_periods_sim"] = 500  # Количество периодов ВЧ поля, которое надо просчитать
    cf["sim_periods_div"] = 100  # Количество точек результата на период и шаг по времени расчета цепи
    cf["tmax_sim"] = cf["Tf"] * cf["num_periods_sim"]  # Сколько времени просчитывать в Ngspice
    cf["tmin_sim"] = cf["Tf"] * (cf["num_periods_sim"] - 5)  # От какого времени делать вывод
    cf["timestep_output"] = cf["Tf"] / cf["sim_periods_div"]  # Шаг, с которым будет вывод
    # Минимально необходимое количество периодов на интегрирование ДУ цепи, при котором результат интеграла сходится.
    # Это и будет критерий наступления установившегося режима
    cf["num_periods_for_integration"] = 10
    cf["first_steady_period"] = cf["num_periods_sim"] - cf["num_periods_for_integration"]
#    cf['ngspice_sim_step_period_frac'] = 500

    ##############
    # 5. Определение Te
    ##############

    sol = optimize.root_scalar(dfr, bracket=[1, 7], x0=3, x1=5, xtol=1e-3, method='secant')
    cf["Te"] = sol.root
    print(f"Te={cf['Te']:.2f} [eV], f0={cf['f0']/1e6:.2f} [MHz], initial ne={cf['ne']:.2e}")

    ##############
    # 6. Определение цены ионизации газа
    ##############

    cf["eps_ex"] = 11.5
    cf["eps_el"] = 3 * ct["me"] * cf["Te"] / ct["Mi"]
    cf["eps_iz"] = 15.76  # Энергия ионизации Ar [eV]
    cf["eps_e"] = 2 * cf["Te"]

    # Строим график Te для ручной проверки
#    if cf["verbose_plots"]:
#        plot_Te()
#        plot_K()

    cf["P0"] = (cf["Vm"] / (2 * np.sqrt(2))) ** 2 / cf["val_R_rf"]
    
    cf["val_C_m1"] = cf["C_m1_init"] 
    cf["val_C_m2"] = cf["C_m2_init"]
    
def redefineCircuitParameters():
    
    #################
    # 8. Вычисляемые величины №2
    #################

    Km = Kel(cf["Te"]) + Kiz(cf["Te"]) + Kex(cf["Te"])  # Коэффициент электрон-нейтральных столкновений [м^3*с^-1]
    nu_el_netr = Km * cf["ng"]  # Частота электрон-нейтральных столкновений [c^-1]
    v_midd_e = math.sqrt(8 * ct["qe"] * cf["Te"] / (math.pi * ct["me"]))  # Средняя тепловая скорость электронов
    nu_eff = nu_el_netr + (v_midd_e / cf["l_B"])  # Эффективная частота электрон-нейтральных столкновений [c^-1]

    cf["Lp"] = cf["l_B"] * ct["me"] / (ct["qe"] ** 2 * cf["ne"] * cf["Ae"])  # Индуктивность bulk плазмы [Гн]
    cf["Rp"] = nu_eff * cf["Lp"]  # Сопротивление bulk плазмы [Ом]
    cf["alpha"] = -1 / cf["Te"]  # Коэффициент показателя экспоненты электронного тока
    cf["Iion1"] = ct["qe"] * cf["ne"] * u_Bohm(cf["Te"]) * cf["Ae"]  # Полный ионный ток на горячий электрод [А]
    cf["Iion2"] = ct["qe"] * cf["ne"] * u_Bohm(cf["Te"]) * cf["Ag"]  # Полный ионный ток на заземленный электрод [А]
    cf["Ie01"] = ct["qe"] * cf["ne"] * v_midd_e * cf["Ae"]  # Амплитуда электронного тока у горячего электрода [А]
    cf["Ie02"] = ct["qe"] * cf["ne"] * v_midd_e * cf["Ag"]  # Амплитуда электронного тока у заземленного электрода [А]
    cf["CCs1"] = (ct["qe"] * cf["ne"] * ct["eps_0"] * cf["Ae"] ** 2) / 2  # Коэффициент при емкости слоя горячего электрода
    cf["CCs2"] = (ct["qe"] * cf["ne"] * ct["eps_0"] * cf["Ag"] ** 2) / 2  # Коэффициент при емкости слоя заземленного электрода

    
def sample_deviatedC(initial_values, percentages, N, linear=False, seed=None, symmetric=False):

    if symmetric:
        rangeC1 = (initial_values[0] * (1 - percentages[0]/100), initial_values[0] * (1 + percentages[0]/100))
        rangeC2 = (initial_values[1] * (1 - percentages[1]/100), initial_values[1] * (1 + percentages[1]/100))
    else:
        rangeC1 = (initial_values[0] * (1 - percentages[0]/100), initial_values[0])
        rangeC2 = (initial_values[1] * (1 - percentages[1]/100), initial_values[1])

    if linear:
        samplesC1 = np.linspace(rangeC1[0], rangeC1[1], N)
        samplesC2 = np.linspace(rangeC2[0], rangeC2[1], N)
    else:
        if seed is not None:
            np.random.seed(seed)
        samplesC1 = np.random.uniform(low=rangeC1[0], high=rangeC1[1], size=N)
        samplesC2 = np.random.uniform(low=rangeC2[0], high=rangeC2[1], size=N)
        
    return samplesC1, samplesC2

def is_valid_date(date_str):
    """
    Check if the provided date string is a valid date in the format YYYY-MM-DD.
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def get_next_available_aaaa(base_directory, bb, cc):
    """
    Checks for subdirectories matching the pattern BB_CC_AAAA_YYYY-MM-DD
    and returns the next available AAAA. If no existing matches are found, returns 0.
    Raises an error if the next AAAA is unavailable (i.e., exceeds 9999).
    
    :param base_directory: The directory to check for subdirectories.
    :param bb: The arbitrary string identifier to match in the directory names.
    :return: The next available AAAA as an integer.
    :raises ValueError: If the next AAAA exceeds 9999.
    """
    # Define the regex pattern to match the required directory format
    regex = re.compile(rf"^{re.escape(bb)}_{re.escape(cc)}_(\d{{4}})_(\d{{4}})-(\d{{2}})-(\d{{2}})$")

    existing_a = []

    # Traverse through subdirectories in the given base directory
    for root, dirs, files in os.walk(base_directory):
        for dir_name in dirs:
            match = regex.match(dir_name)
            if match:
                aaaa = int(match.group(1))  # Extract AAAA as an integer
                date_str = f"{match.group(2)}-{match.group(3)}-{match.group(4)}"  # YYYY-MM-DD
                if is_valid_date(date_str):
                    existing_a.append(aaaa)

    # If no existing AAAA found, return 0
    if not existing_a:
        return 0

    # Find the next available AAAA
    max_aaaa = max(existing_a)
    next_aaaa = max_aaaa + 1
    
    # Ensure next_aaaa is a 4-digit number (i.e., less than 10000)
    if next_aaaa < 10000:
        return next_aaaa
    else:
        raise ValueError("No available AAAA left. Maximum limit reached.")


def create_subdirectory(base_directory, aaaa, bb, cc):
    """
    Creates a subdirectory with the name format AAAA-BB_YYYY-MM-DD.
    
    :param base_directory: The directory in which to create the subdirectory.
    :param aaaa: A 4-digit integer (leading zeros) as a string.
    :param bb: An arbitrary string identifier.
    :raises ValueError: If AAAA is not a valid 4-digit integer or if the directory creation fails.
    """
    # Validate AAAA
    if not (isinstance(aaaa, int) and 0 <= aaaa < 10000):
        raise ValueError("AAAA must be a 4-digit integer (0-9999).")
    
    # Format AAAA with leading zeros
    formatted_aaaa = f"{aaaa:04d}"
    
    # Get the current date in YYYY-MM-DD format
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Create the directory name
    directory_name = f"{bb}_{cc}_{formatted_aaaa}_{current_date}"
    
    # Create the full path for the new subdirectory
    full_path = os.path.join(base_directory, directory_name)
    
    # Create the subdirectory
    try:
        os.makedirs(full_path)
        print(f"Subdirectory created: {full_path}")
        return full_path, current_date
    except Exception as e:
        raise ValueError(f"Failed to create subdirectory: {e}")
        
        
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout  # Save a reference to the original stdout
        self.log_file = open(filename, 'a')  # Open the log file in append mode

    def write(self, message):
        self.terminal.write(message)  # Write to the terminal (console)
        self.log_file.write(message)   # Write to the log file

    def flush(self):
        pass  # This is needed for Python 3 compatibility