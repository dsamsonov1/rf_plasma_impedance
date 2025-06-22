import sys
import os
from rfd_plots import *
from rfd_utils import *
from rfd_conf import *
import pandas as pd
from datetime import datetime
import matplotlib.gridspec as gridspec

print(f'Config: {cf["name"]}\n{cf["comment"]}')


##########
# 7. Определяем и считаем цепь
##########

def calc_discharge():

    matching_flag = cf["matching_flag"]

    miter = 0
    matching_cond = True
    # print(f'mc={np.abs(val_C_m1 - val_C_m1_prev) > 5e-12 and np.abs(val_C_m2 - val_C_m2_prev) > 5e-12}', end=' ')

    print('=== NEW TASK STARTS ===\n')
    print(f'RF excitation: P0={(cf["Vm"] / (2 * np.sqrt(2))) ** 2 / 50:.2f} [W], f0={cf["f0"] / 1e6:.2f} [MHz], p={cf["p0"]} [Pa] Ar\n')
    print(f'Constant parameters: Vp={cf["Vp"]:.2e} ng={cf["ng"]:.2e} Kiz={Kiz(cf["Te"]):.2e} eps_c={eps_c(cf["Te"]):.2e} eps_e={cf["eps_e"]:.2e} fE={cf["fE"]:.2e} fG={cf["fG"]:.2e}\n\n')

    print('=== SIMULATION STARTS ===\n')

    while matching_cond:

        miter = miter + 1
        print(
            f'-- matching iteration #{miter: =2} starts: C1={cf["val_C_m1"] * 1e12:.2f} [pF], C2={cf["val_C_m2"] * 1e12:.2f} [pF]')

        #    ne_new = 0
        ne_new = -2 * cf["eps_ne"]
        iter_no = 0

        while np.abs(cf["ne"] - ne_new) > cf["eps_ne"]:

            if iter_no > 0:
                cf["ne"] = ne_new

            if iter_no < cf["max_iter_ne"]:
                iter_no = iter_no + 1

                print(f'  -- ne #{iter_no: =2}:', end=' ')

                (analysis, out_Rp) = calcCircuit()
                (Ppl, _Vs1, _Vs2) = calcPlasmaQuantities(analysis, out_Rp)

                ##############
                # Определение ne
                ##############

                Pguess = cf["ne"] * cf["Vp"] * cf["ng"] * Kiz(cf["Te"]) * ct["qe"] * \
                         (eps_c(cf["Te"]) + cf["eps_e"] + cf["fE"] * np.abs(_Vs1) + cf["fG"] * np.abs(_Vs2) + cf["Te"] / 2)

                ne_new = Ppl / (cf["Vp"] * cf["ng"] * Kiz(cf["Te"]) * ct["qe"] * \
                                (eps_c(cf["Te"]) + cf["eps_e"] + cf["fE"] * np.abs(_Vs1) + cf["fG"] * np.abs(_Vs2) + cf["Te"] / 2))

                # "регуляризация" для улучшения сходимости итераций ne (suggested by G. Marchiy)
                ne_new = cf["ne"] + (ne_new - cf["ne"]) * cf["beta"]

                print(f'Ppl={Ppl:.2f} [W], Pguess={Pguess:.2f} [W], ne={cf["ne"]:.2e} [m^-3], ne_new={ne_new:.2e} [m^-3]', end=' ')
                #plot_UI(analysis, out_Rp)

                if np.abs(ne_new - cf["ne"]) < cf["eps_ne"]:
                    print(f'dne={np.abs(ne_new - cf["ne"]):.2e} <- ne CONVERGED')

                    ##############
                    # Определение импеданса нагрузки для согласования
                    ##############

                    val_C_m2_prev = cf["val_C_m2"]
                    val_C_m1_prev = cf["val_C_m1"]

                    # Импеданс на выходе C-C звена для расчета согласования
                    (Rmm, Xmm) = calcImpedance_1Harm(analysis, '3', '0', '4', '5', cf["val_R_m"])

                    print(f'   - OUT: Z_m=({Rmm:.2f}, {Xmm:.2f}) [Ohm] -> {np.abs(complex(Rmm, Xmm)):.2f}*exp(j*{np.degrees(np.angle(complex(Rmm, Xmm))):.2f}deg)')

                    plot_UI(analysis, out_Rp)

                    if matching_flag:
                        (cf["val_C_m2"], cf["val_C_m1"]) = calcMatchingNetwork(Rmm, Xmm, 2 * np.pi * cf["f0"], 50)

                        matching_cond = np.abs(cf["val_C_m1"] - val_C_m1_prev) > 1e-12 or np.abs(cf["val_C_m2"] - val_C_m2_prev) > 1e-12
                        
                        beta2 = 0.01
                        cf["val_C_m1"] = val_C_m1_prev + beta2 * (cf["val_C_m1"] - val_C_m1_prev)
                        cf["val_C_m2"] = val_C_m2_prev + beta2 * (cf["val_C_m2"] - val_C_m2_prev)

                        print(f'-- dCm1={np.abs(cf["val_C_m1"] - val_C_m1_prev) * 1e12:.2f} [pF], dCm2={np.abs(cf["val_C_m2"] - val_C_m2_prev) * 1e12:.2f} [pF]', end=' ')

                        if matching_cond:
                            print(f'<- NEW MATCHING VALUES: C1={cf["val_C_m1"] * 1e12:.2f} C2={cf["val_C_m2"] * 1e12:.2f}\n')
                            
                            if cf["val_C_m1"] <= 0 or cf["val_C_m2"] <= 0:
                                sys.exit("Wrong C1 or C2 value. STOP.")
                        else:
                            print(
                                f'<- MATCHING CONVERGED: C1={val_C_m1_prev * 1e12:.2f} C2={val_C_m2_prev * 1e12:.2f}\n')
                    else:
                        matching_cond = False
                else:
                    print(f'dne={np.abs(ne_new - cf["ne"]):.2e} --', end='\n')

            else:
                # print(f'ne NOT CONVERGED. ITERATIONS LIMIT REACHED\n')
                # matching_cond = False
                sys.exit("ne NOT CONVERGED. ITERATIONS LIMIT REACHED. STOP.")

    if iter_no < cf["max_iter_ne"]:
        printSimulationResults(analysis, out_Rp)
        plot_UI(analysis, out_Rp)

    pd1 = calcPowerBalance(analysis, out_Rp)
    _, Vs1, Vs2 = calcPlasmaQuantities(analysis, out_Rp)
    Ubias = calcVoltage_Harm(analysis, '5', '0', 0, True)
    Urf = calcVoltage_Harm(analysis, '5', '0', 2, False)

    print(f'Ubias = {Ubias:.2f} [V]', end='\n')

    print('=== TASK FINISHED ===\n\n')

    pd2 = pd.DataFrame({'p0 [Pa]': [cf["p0"]], 'f0 [MHz]': [cf["f0"]/1e6], 'ne [m^-3]': [cf["ne"]], 'Te [eV]': [cf["Te"]],
                        'C1 [pF]': [cf["val_C_m1"]/1e-12], 'C2 [pF]': [cf["val_C_m2"]/1e-12],
                        'L1 [nH]': [cf["val_L_m2"]/1e-9], 'P0 [W]': [cf["P0"]],
                        'Vp [m^3]': [cf["Vp"]], 'ng [m^-3]': [cf["ng"]], 'Kiz []': [Kiz(cf["Te"])], 'eps_c []': [eps_c(cf["Te"])],
                        'eps_e []': [cf["eps_e"]], 'fE []': [cf["fE"]], 'fG []': [cf["fG"]], 'Vs1 [V]': [Vs1], 'Vs2 [V]': [Vs2],
                        'Ubias [V]': [Ubias], 'Urf [V]': [Urf], 'Ae [m^2]': [cf["Ae"]], 'Ag [m^2]': [cf["Ag"]],
                        'L_bulk [m]': [cf["l_B"]], 'T0 [K]': [cf["T0"]], 'Rrf [Ω]': [cf['val_R_rf']], 'Rm [Ω]': [cf['val_R_m']],
                        'Cstray [pF]': [cf['val_C_stray']], 'Rstray [Ω]': [cf['val_R_stray']], "Ie01 [A]": [cf["Ie01"]],
                        "alpha []": [cf["alpha"]], "Iion1 [A]": [cf["Iion1"]], "CCs1 []": [cf["CCs1"]], "Lp [nH]": [cf["Lp"]/1e-9],
                        "Rp [Ω]": [cf["Rp"]], "Ie02 [A]": [cf["Ie02"]], "Iion2 [A]": [cf["Iion2"]], "CCs2 []": [cf["CCs2"]],
                        "Vm [V]": [cf["Vm"]], 'jIon1 [uA/cm^2]': [cf['Iion1']*1e6/(cf['Ae']*1e4)], 'jIon2 [uA/cm^2]': [cf['Iion2']*1e6/(cf['Ag']*1e4)]})

    return pd.concat([pd1, pd2], axis=1)

sweep_freq = False
sweep_pressure = False

df = pd.DataFrame()

cf['next_aaaa'] = get_next_available_aaaa('out/', cf['name'])
cf['out_path'], cf['current_date'] = create_subdirectory('out/', cf['next_aaaa'], cf['name'])

# Redirect stdout to the Logger
sys.stdout = Logger(f'{cf['out_path']}/output.log')

if sweep_freq:
    #    freqs = [13.56e6, 27e6, 40e6, 60e6, 80e6]
    #    freqs = [13.56e6, 27e6]
    #    inds = [1500e-9, 150e-9]
    freqs = [20e6]
    inds = [550e-9] 
    #freqs = [13.56e6, 27e6]
    #inds = [1200e-9, 750e-9]

    for i in range(len(freqs)):
        cf["f0"] = freqs[i]
        cf["val_L_m2"] = inds[i]
        redefineRuntimeParams()
        df = pd.concat([df, calc_discharge()], ignore_index=True)

elif sweep_pressure:

    press = [1, 2.5, 5, 7.5, 10]

    for p in press:
        cf["p0"] = p
        redefineRuntimeParams()
        df = pd.concat([df, calc_discharge()], ignore_index=True)

else:
    redefineRuntimeParams()
    df = pd.concat([df, calc_discharge()], ignore_index=True)

df.to_excel(f'{cf['out_path']}/{cf['next_aaaa']:04d}_{cf['name']}_{cf['current_date']}.xlsx', index=False)


fig = plt.figure(figsize=(9, 18))
gs = gridspec.GridSpec(5, 3, hspace=0)
#axs = gs.subplots(sharex=True)
df.plot(ax=fig.add_subplot(gs[0]), x='p0 [Pa]', y=['Pp [W]', 'PRm [W]', 'PRstray [W]'], marker='x')
df.plot(ax=fig.add_subplot(gs[1]), x='p0 [Pa]', y=['Ubias [V]', 'Urf [V]', 'Vs1 [V]', 'Vs2 [V]'], marker='x')
df.plot(ax=fig.add_subplot(gs[2]), x='p0 [Pa]', y='ne [m^-3]', marker='x')
df.plot(ax=fig.add_subplot(gs[3]), x='p0 [Pa]', y='Te [eV]', marker='x')
df.plot(ax=fig.add_subplot(gs[4]), x='p0 [Pa]', y=['Iion1 [A]', 'Iion2 [A]'], marker='x')
df.plot(ax=fig.add_subplot(gs[5]), x='p0 [Pa]', y=['C1 [pF]', 'C2 [pF]'], marker='x')
df.plot(ax=fig.add_subplot(gs[6]), x='p0 [Pa]', y=['Re(Zl) [Ohm]'], marker='x')
df.plot(ax=fig.add_subplot(gs[7]), x='p0 [Pa]', y=['Im(Zl) [Ohm]'], marker='x')
df.plot(ax=fig.add_subplot(gs[8]), x='p0 [Pa]', y=['Re(Zp) [Ohm]'], marker='x')
df.plot(ax=fig.add_subplot(gs[9]), x='p0 [Pa]', y=['Im(Zp) [Ohm]'], marker='x')
df.plot(ax=fig.add_subplot(gs[10]), x='p0 [Pa]', y=['jIon1 [uA/cm^2]', 'jIon2 [uA/cm^2]'], marker='x')
fig.suptitle(f'{cf['name']}: Ar f0={cf["f0"]/1e6:.2f} [MHz] L={cf["l_B"]/1e-2:.2f} [cm] Ae={cf["Ae"]*1e4:.2f} [cm^2] Ag={cf["Ag"]*1e4:.2f} [cm^2] P0={cf["P0"]:.1f} [W]')
plt.tight_layout()
plt.show()
fig.savefig(f'{cf['out_path']}/{cf['next_aaaa']:04d}_{cf['name']}_{cf['current_date']}_sweep.png')

#TODO Копировать исходный файл конфигурации модели в каталог с результатами расчета
#TODO Сделать PDF отчет, включающий графики каждой итерации для контроля качества модели
#TODO Сделать расчет скорости и селективности травления через оценку IEDF
#TODO Приделать контроль установившегося режима в цепи (breakpoint в ngspice?)