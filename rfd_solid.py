from scipy import optimize
from rfd_utils import *
from rfd_plots import *
from rfd_conf import *

print(f'Config: {cf["name"]}\n{cf["comment"]}')

#################
# 3. Вычисляемые величины №1
#################

cf["ng"] = cf["p0"] / (ct["k_B"] * cf["T0"])    # Концентрация буферного газа [м^-3]
cf["Vp"] = cf["Ae"] * cf["l_B"]                 # Объем плазменного столба
cf["fE"] = cf["Ae"] / (cf["Ae"] + cf["Ag"])     # Весовой коэф. слоя E
cf["fG"] = cf["Ag"] / (cf["Ae"] + cf["Ag"])     # Весовой коэф. слоя G
cf["Tf"] = 1 / cf["f0"]                         # Период ВЧ поля [с]

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
# 6. Определение Te
##############

sol = optimize.root_scalar(dfr, bracket=[1, 7], x0=3, x1=5, xtol=1e-3, method='secant')
cf["Te"] = sol.root
print(f'Te={cf["Te"]:.2f} [eV] ne={cf["ne"]:.2e}')

##############
# 7. Определение цены ионизации газа
##############

cf["eps_ex"] = 11.5
cf["eps_el"] = 3 * ct["me"] * cf["Te"] / ct["Mi"]
cf["eps_iz"] = 15.76  # Энергия ионизации Ar [eV]
cf["eps_e"] = 2 * cf["Te"]

# Строим график Te для ручной проверки
if cf["verbose_plots"]:
    plot_Te()

if cf["verbose_plots"]:
    plot_K()

##########
# 8. Определяем и считаем цепь
##########

matching_flag = cf["matching_flag"]

val_C_m2_prev = 0
val_C_m1_prev = 0
max_iter_ne = cf["max_iter_ne"]

miter = 0
matching_cond = True
# print(f'mc={np.abs(val_C_m1 - val_C_m1_prev) > 5e-12 and np.abs(val_C_m2 - val_C_m2_prev) > 5e-12}', end=' ')

print(f'RF excitation: P0={(cf["Vm"] / (2 * np.sqrt(2))) ** 2 / 50:.2f} [W], f0={cf["f0"] / 1e6:.2f} [MHz], p={cf["p0"]} [Pa] Ar\n')
print(f'Constant parameters: Vp={cf["Vp"]:.2e} ng={cf["ng"]:.2e} Kiz={Kiz(cf["Te"]):.2e} eps_c={eps_c(cf["Te"]):.2e} eps_e={cf["eps_e"]:.2e} fE={cf["fE"]:.2e} fG={cf["fG"]:.2e}\n\n')

print(f'=== SIMULATION STARTS ===\n')

while matching_cond:

    miter = miter + 1
    print(f'-- matching iteration #{miter: =2} starts: C1={cf["val_C_m1"] * 1e12:.2f} [pF], C2={cf["val_C_m2"] * 1e12:.2f} [pF]')

    ne_new = 0
    iter_no = 0

    while np.abs(cf["ne"] - ne_new) > cf["eps_ne"]:

        if iter_no > 0:
            cf["ne"] = ne_new

        if iter_no < max_iter_ne:
            iter_no = iter_no + 1

            print(f'  -- ne iteration #{iter_no: =2} starts --\n', end=' ')

            (analysis, out_Rp) = calcCircuit(cf["Te"], cf["ne"], cf["val_C_m1"], cf["val_C_m2"])

            time_raw = np.array(analysis.time)
            Vpl_raw = getU('5', '0', analysis)  # Vpl
            Ipl_raw = getU('8', '9', analysis) / out_Rp  # Ipl
            Vs1_raw = getU('5', '7', analysis)  # Vs1
            Vs2_raw = getU('9', '10', analysis)  # Vs2

            t_integration = extract_N_periods(time_raw, 1, cf["num_periods_for_integration"], 'rev')

            Vpl_integration = extract_N_periods(Vpl_raw, 1, cf["num_periods_for_integration"], 'rev')
            Ipl_integration = extract_N_periods(Ipl_raw, 1, cf["num_periods_for_integration"], 'rev')
            Ppl_integration = np.multiply(Vpl_integration, Ipl_integration)

            Ppl = get_mean(t_integration, Ppl_integration, cf["num_periods_for_integration"])

            Vs1_integration = extract_N_periods(Vs1_raw, 1, cf["num_periods_for_integration"], 'rev')
            Vs2_integration = extract_N_periods(Vs2_raw, 1, cf["num_periods_for_integration"], 'rev')

            _Vs1 = get_mean(t_integration, Vs1_integration, cf["num_periods_for_integration"])
            _Vs2 = get_mean(t_integration, Vs2_integration, cf["num_periods_for_integration"])
            if cf["verbose_plots"]:
                print(f'  - VERBOZE: Vs_e={np.abs(_Vs1):.2f} [V] Vs_g={np.abs(_Vs2):.2f} [V]')

            ##############
            # Определение ne
            ##############

            Pguess = cf["ne"] * cf["Vp"] * cf["ng"] * Kiz(cf["Te"]) * ct["qe"] * \
                    (eps_c(cf["Te"]) + cf["eps_e"] + cf["fE"] * np.abs(_Vs1) + cf["fG"] * np.abs(_Vs2) + cf["Te"] / 2)
#            Pguess = cf["ne"] * cf["Vp"] * cf["ng"] * Kiz(cf["Te"]) * \
#                      (eps_c(cf["Te"]) + cf["eps_e"] + cf["fE"] * ct["qe"] * np.abs(_Vs1) + cf["fG"] * ct["qe"] * np.abs(_Vs2) + cf["Te"] / 2)

            ne_new = Ppl / (
                    cf["Vp"] * cf["ng"] * Kiz(cf["Te"]) * (
                        eps_c(cf["Te"]) + cf["eps_e"] + cf["fE"] * np.abs(_Vs1) + cf["fG"] * np.abs(_Vs2) + cf["Te"] / 2) * ct["qe"])

#            ne_new = Ppl / (
#                    cf["Vp"] * cf["ng"] * Kiz(cf["Te"]) * \
#                    (eps_c(cf["Te"]) + cf["eps_e"] + cf["fE"] * ct["qe"] * np.abs(_Vs1) + cf["fG"] * ct["qe"] * np.abs(_Vs2) + cf["Te"] / 2))

            print(f'  - OUT: Ppl={Ppl:.2f} [W] Pguess={Pguess:.2f} [W] ne={cf["ne"]:.2e} [cm^-3], ne_new={ne_new:.2e} [cm^-3]', end=' ')

            if np.abs(ne_new - cf["ne"]) < cf["eps_ne"]:
                print(f'|ne_new - ne|={np.abs(ne_new - cf["ne"]):.2e} <- ne CONVERGED')

                ##############
                # Определение импеданса нагрузки для согласования
                ##############

                val_C_m2_prev = cf["val_C_m2"]
                val_C_m1_prev = cf["val_C_m1"]

                (RL, XL) = calcLoadImpedance_1Harm(analysis)
                (RRL, XXL) = calcLoadImpedance2_1Harm(analysis)
                print(f'  - OUT: Z_l=({RL:.2f}, {XL:.2f}) [Ohm] ---> {np.abs(complex(RL, XL)):.2f}*exp(j*{np.degrees(np.angle(complex(RL, XL))):.2f}deg)')
                print(f'  - OUT: Z_l (noind)=({RRL:.2f}, {XXL:.2f}) [Ohm]')

                if matching_flag:
                    (cf["val_C_m2"], cf["val_C_m1"]) = calcMatchingNetwork(RL, XL, 2 * np.pi * cf["f0"], 50)
                    print(f'- dCm1={np.abs(cf["val_C_m1"] - val_C_m1_prev) * 1e12:.2f} [pF], dCm2={np.abs(cf["val_C_m2"] - val_C_m2_prev) * 1e12:.2f} [pF]', end=' ')

                    matching_cond = np.abs(cf["val_C_m1"] - val_C_m1_prev) > 1e-12 or np.abs(cf["val_C_m2"] - val_C_m2_prev) > 1e-12

                    if matching_cond:
                        print(f'<- NEW MATCHING VALUES: C1={cf["val_C_m1"] * 1e12:.2f} C2={cf["val_C_m2"] * 1e12:.2f}\n')
                    else:
                        print(f'<- MATCHING CONVERGED: C1={val_C_m1_prev * 1e12:.2f} C2={val_C_m2_prev * 1e12:.2f}\n')
                else:
                    matching_cond = False
            else:
                print(f'|ne_new - ne|={np.abs(ne_new - cf["ne"]):.2e}')

            print(f'  -- iteration #{iter_no: =2} complete --\n')

        else:
            print(f'ne NOT CONVERGED. ITERATIONS LIMIT REACHED\n')
            matching_cond = False

if iter_no < max_iter_ne:
    printSimulationResults(analysis, out_Rp)
