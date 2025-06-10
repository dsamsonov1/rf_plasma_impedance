from scipy import optimize
from rfd_plots import *

print(f'Config: {cf["name"]}\n{cf["comment"]}')

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
if cf["verbose_plots"]:
    plot_Te()
    plot_K()

##########
# 7. Определяем и считаем цепь
##########

matching_flag = cf["matching_flag"]

val_C_m2_prev = 0
val_C_m1_prev = 0

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

        if iter_no < cf["max_iter_ne"]:
            iter_no = iter_no + 1

            print(f'  -- ne iteration #{iter_no: =2} starts --\n', end=' ')

            (analysis, out_Rp) = calcCircuit(cf["Te"], cf["ne"], cf["val_C_m1"], cf["val_C_m2"])
            (Ppl, _Vs1, _Vs2) = calcPlasmaQuantities(analysis, out_Rp)

            ##############
            # Определение ne
            ##############

            Pguess = cf["ne"] * cf["Vp"] * cf["ng"] * Kiz(cf["Te"]) * ct["qe"] * \
                    (eps_c(cf["Te"]) + cf["eps_e"] + cf["fE"] * np.abs(_Vs1) + cf["fG"] * np.abs(_Vs2) + cf["Te"] / 2)
#            Pguess = cf["ne"] * cf["Vp"] * cf["ng"] * Kiz(cf["Te"]) * \
#                      (eps_c(cf["Te"]) + cf["eps_e"] + cf["fE"] * ct["qe"] * np.abs(_Vs1) + cf["fG"] * ct["qe"] * np.abs(_Vs2) + cf["Te"] / 2)

            ne_new = Ppl / (
                    cf["Vp"] * cf["ng"] * Kiz(cf["Te"]) * ct["qe"] * (
                        eps_c(cf["Te"]) + cf["eps_e"] + cf["fE"] * np.abs(_Vs1) + cf["fG"] * np.abs(_Vs2) + cf["Te"] / 2))

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

if iter_no < cf["max_iter_ne"]:
    printSimulationResults(analysis, out_Rp)
