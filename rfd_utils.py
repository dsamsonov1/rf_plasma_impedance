import math

import numpy as np
from PySpice.Spice.Netlist import Circuit
from scipy.fft import rfft, rfftfreq
from scipy.integrate import trapezoid
from rfd_conf import *
from scipy import optimize
import pandas as pd


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
    data_cut = extract_N_periods(a_Data, first_steady_period, num_periods_for_spectra, 'fwd')
    _spectra = rfft(data_cut) / (num_periods_for_spectra * sim_periods_div)
    _freqs = rfftfreq(data_cut.size, d=Tf / sim_periods_div)

    freqsMHz = _freqs / 1e6

    full_abs = np.abs(_spectra)
    full_angle = np.angle(_spectra)

    idxHarm = (
                      a_nHarm + 1) * num_periods_for_spectra  # Индекс в массиве результатов fft, чтобы вырезать (немного больше) N гармоник
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

    return (
        _spectra, full_abs, full_angle, _freqs, freqsMHz, reduced_freqs, reduced_abs, reduced_angle, waste_freqs,
        waste_abs,
        waste_angle, true_freqs, true_abs, true_angle)


# Пытаемся найти Te как корень баланса частиц Vp*ng*Kiz(Te)-(Ae + Ag)*u_Bohm(Te) на отрезке [1-7] eV
def dfr(a_Te):
    return cf["Vp"] * cf["ng"] * Kiz_novect(a_Te) - (cf["Ae"] + cf["Ag"]) * u_Bohm_novect(a_Te)


def eps_c_novec(a_Te):
    return cf["eps_iz"] + (Kel(a_Te) * cf["eps_el"] + Kex(a_Te) * cf["eps_ex"]) / Kiz(a_Te)


eps_c = np.vectorize(eps_c_novec, otypes=[float])


def calcCircuit():

    if cf["verbose_plots"]:
        print(f' - VERBOSE: Lp={cf["Lp"] * 1e9:.2f} [nH] Rp={cf["Rp"]:.2f} [Ohm] alpha={cf["alpha"]:.2e}')
        print(
            f'  - VERBOSE: Be_e={cf["Ie01"]:.3f} Bi_e={cf["Iion1"]:.2e} Cs1={np.sqrt(cf["CCs1"]):.2e}/sqrt(Vs_e(t))')
        print(
            f'  - VERBOSE: Be_g={cf["Ie02"]:.3f} Bi_g={cf["Iion2"]:.2e} Cs2={np.sqrt(cf["CCs2"]):.2e}/sqrt(Vs_g(t))')

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
    circuit.BehavioralSource('Be_e', 5, 7, current_expression=f'v(7,5) > 0 ? {cf["Ie01"]}*exp({cf["alpha"]}*v(7,5)) : 1e-12')
    circuit.CurrentSource('Bi_e', 7, 5, cf["Iion1"])
    circuit.BehavioralCapacitor('Cs1', 7, 5, capacitance_expression=f'C=\'sqrt({cf["CCs1"]}/abs(v(7,5)))\'')
    circuit.L('L_p', 7, 8, cf["Lp"])
    circuit.R('R_p', 8, 9, cf["Rp"])
    circuit.BehavioralSource('Be_g', 10, 9, current_expression=f'v(9,10) > 0 ? {cf["Ie02"]}*exp({cf["alpha"]}*v(9,10)) : 1e-12')
    circuit.CurrentSource('Bi_g', 9, 10, cf["Iion2"])
    circuit.BehavioralCapacitor('Cs2', 9, 10, capacitance_expression=f'C=\'sqrt({cf["CCs2"]}/abs(v(9,10)))\'')
    circuit.VoltageSource('Viz', 10, 0, 0)
#    circuit.LosslessTransmissionLine('TL', 5, 0, 0, 0, impedance=50, frequency=cf["f0"], normalized_length=0.25)
#    circuit.SingleLossyTransmissionLine('TL', 5, 0, 0, 0, model='ymod', length=1, raw_spice='\n.MODEL ymod txl R=0.1 L=8.972e-9 G=0 C=0.468e-12 length=22.12')
    simulator = circuit.simulator()

    # Надо задать какие-то около-, но ненулевые НУ,
    # т.к. иначе pyspice ломается на предупреждениях от Ngspice
    simulator._initial_condition = {'v(5)': 1e-10, 'v(9)': 1e-10}

    # print(simulator) # Можно напечатать .IC для проверки
    # print(circuit) # Можно напечатать получившийся netlist для проверки

    return simulator.transient(step_time=cf["Tf"] / 100, end_time=cf["tmax_sim"]), cf["Rp"]


##############
# Определение вспомогательных функций
##############

def calcImpedance_1Harm(a_analysis, a_u1, a_U2, a_I1, a_I2, a_Rval):
    Vl_raw = getU(a_u1, a_U2, a_analysis)  # Vl
    Il_raw = getU(a_I1, a_I2, a_analysis) / a_Rval  # Il

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


def calcVoltage_Harm(a_analysis, a_u1, a_U2, a_harm):
    Vl_raw = getU(a_u1, a_U2, a_analysis)  # Vl

    Vl_2_last_periods = extract_N_periods(Vl_raw, 1, 2)

    spectraVl = rfft(Vl_2_last_periods) / (2 * cf["sim_periods_div"])
    # freqsl = rfftfreq(Vl_2_last_periods.size, d=cf["Tf"] / cf["sim_periods_div"])

    return np.real(spectraVl[a_harm])


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


def calcPowerBalance(a_analysis, a_Rp):
    time_raw = np.array(a_analysis.time)

    Vpl_raw = getU('5', '0', a_analysis)  # Vpl
    Ipl_raw = getU('8', '9', a_analysis) / a_Rp  # Ipl

    t_integration = extract_N_periods(time_raw, 1, cf["num_periods_for_integration"])
    # data[-(num_periods_for_integration+1)*sim_periods_div:-1*sim_periods_div]

    Vpl_integration = extract_N_periods(Vpl_raw, 1, cf["num_periods_for_integration"])
    Ipl_integration = extract_N_periods(Ipl_raw, 1, cf["num_periods_for_integration"])
    Ppl_integration = np.multiply(Vpl_integration, Ipl_integration)

    Ppl = get_mean(t_integration, Ppl_integration, cf["num_periods_for_integration"])

    VRm_raw = getU('4', '5', a_analysis)  # VRm
    VRstray_raw = getU('6', '0', a_analysis)  # VRm

    VRm_integration = extract_N_periods(VRm_raw, 1, cf["num_periods_for_integration"])
    P_R_m = get_mean(t_integration, np.multiply(VRm_integration, VRm_integration),
                     cf["num_periods_for_integration"]) / cf["val_R_m"]

    VRstray_integration = extract_N_periods(VRstray_raw, 1, cf["num_periods_for_integration"])
    P_R_stray = get_mean(t_integration, np.multiply(VRstray_integration, VRstray_integration),
                         cf["num_periods_for_integration"]) / cf["val_R_stray"]

    print(
        f'Ppl={Ppl:.2f} [W], PRm={P_R_m:.2f} [W], PRstray = {P_R_stray:.2f} [W], TOTAL={Ppl + P_R_m + P_R_stray:.2f} [W]')

    Vpl_2_last_periods = extract_N_periods(Vpl_raw, 1, 2)
    Ipl_2_last_periods = extract_N_periods(Ipl_raw, 1, 2)

    spectraVpl = rfft(Vpl_2_last_periods) / (2 * cf["sim_periods_div"])
    spectraIpl = rfft(Ipl_2_last_periods) / (2 * cf["sim_periods_div"])
    freqsl = rfftfreq(Vpl_2_last_periods.size, d=cf["Tf"] / cf["sim_periods_div"])

    UUpl = abs(spectraVpl)[2]
    IIpl = abs(spectraIpl)[2]

    Uapl = np.angle(spectraVpl)[2]
    Iapl = np.angle(spectraIpl)[2]

    Rpl = UUpl / IIpl * np.cos(Uapl - Iapl)
    #    XL=UU/II*np.sin(Ua-Ia)+2*np.pi*config["f0"]*1500e-9
    Xpl = UUpl / IIpl * np.sin(Uapl - Iapl)

    # Импеданс на входе C-C звена для проверки согласования (похоже это не работает правильно
    # т.к. 50 Ом в этой точке будет если это располовиненная цепь - разобраться
    (Rii, Xii) = calcImpedance_1Harm(a_analysis, '2', '0', '4', '5', cf["val_R_m"])

    # Импеданс на входе C-C звена для проверки согласования (похоже это не работает правильно
    # т.к. 50 Ом в этой точке будет если это располовиненная цепь - разобраться
    (Rll, Xll) = calcImpedance_1Harm(a_analysis, '5', '0', '4', '5', cf["val_R_m"])

    print(f'Zi=({Rii:.2f}, {Xii:.2f}) [Ohm], Zl=({Rll:.2f}, {Xll:.2f}) [Ohm], Zp=({Rpl:.2f}, {Xpl:.2f}) [Ohm]')

    return pd.DataFrame({'Re(Zi) [Ohm]': [Rii], 'Im(Zi) [Ohm]': [Xii],
                         'Re(Zl) [Ohm]': [Rll], 'Im(Zl) [Ohm]': [Xll],
                         'Re(Zp) [Ohm]': [Rpl], 'Im(Zp) [Ohm]': [Xpl],
                         'Pp [W]': [Ppl], 'PRm [W]': P_R_m, 'PRstray [W]': P_R_stray,
                         'Ptot [W]': Ppl + P_R_m + P_R_stray})


def calcPlasmaQuantities(a_analysis, a_Rp):
    time_raw = np.array(a_analysis.time)
    Vpl_raw = getU('5', '0', a_analysis)  # Vpl
    Ipl_raw = getU('8', '9', a_analysis) / a_Rp  # Ipl
    Vs1_raw = getU('5', '7', a_analysis)  # Vs1
    Vs2_raw = getU('9', '10', a_analysis)  # Vs2

    t_integration = extract_N_periods(time_raw, 1, cf["num_periods_for_integration"])

    Vpl_integration = extract_N_periods(Vpl_raw, 1, cf["num_periods_for_integration"])
    Ipl_integration = extract_N_periods(Ipl_raw, 1, cf["num_periods_for_integration"])

    Ppl_integration = np.multiply(Vpl_integration, Ipl_integration)

    Ppl = get_mean(t_integration, Ppl_integration, cf["num_periods_for_integration"])

    Vs1_integration = extract_N_periods(Vs1_raw, 1, cf["num_periods_for_integration"])
    Vs2_integration = extract_N_periods(Vs2_raw, 1, cf["num_periods_for_integration"])

    _Vs1 = get_mean(t_integration, Vs1_integration, cf["num_periods_for_integration"])
    _Vs2 = get_mean(t_integration, Vs2_integration, cf["num_periods_for_integration"])

    if cf["verbose_plots"]:
        print(f'  - VERBOZE: Vs_e={np.abs(_Vs1):.2f} [V] Vs_g={np.abs(_Vs2):.2f} [V]')

    return Ppl, _Vs1, _Vs2


def printSimulationResults(a_analysis, a_out_Rp):
    print(f'=== SIMULATION COMPLETE ===\n')

    calcPowerBalance(a_analysis, a_out_Rp)


def redefineRuntimeParams():
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

    cf["num_periods_sim"] = 500  # Количество периодов ВЧ поля, которое надо просчитать
    cf["sim_periods_div"] = 100  # Количество точек результата на период
    cf["tmax_sim"] = cf["Tf"] * cf["num_periods_sim"]  # Сколько времени просчитывать в Ngspice
    cf["tmin_sim"] = cf["Tf"] * (cf["num_periods_sim"] - 5)  # От какого времени делать вывод
    cf["timestep_output"] = cf["Tf"] / cf["sim_periods_div"]  # Шаг, с которым будет вывод
    # Минимально необходимое количество периодов на интегрирование ДУ цепи, при котором результат интеграла сходится.
    # Это и будет критерий наступления установившегося режима
    cf["num_periods_for_integration"] = 50

    ##############
    # 5. Определение Te
    ##############

    sol = optimize.root_scalar(dfr, bracket=[1, 7], x0=3, x1=5, xtol=1e-3, method='secant')
    cf["Te"] = sol.root
    print(f'Te={cf["Te"]:.2f} [eV] ne={cf["ne"]:.2e}')

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
