import sys
import os
from rfd_plots import *
from rfd_utils import *
from rfd_conf import *
import pandas as pd
from datetime import datetime
import shutil


from scipy.optimize import minimize, basinhopping

print(f'Config: {cf["name"]}\n{cf["comment"]}')

def optimizeF():
    
    def targetFuncF(x):
        
        print(f'F={x[0]/1e6:.3f} MHz')
        cf["f0"] = x[0]
        return calc_dischargePoint()

    f0_init = cf["f0"]
    # Initial guess
    a_x0 = [f0_init]
    
    a_bounds = [(0.5*f0_init, 1.5*f0_init)]

    #Local optimization
    result = minimize(targetFuncF, x0=a_x0, bounds=a_bounds, method="Nelder-Mead", tol=1e2)
    print(result)

def optimizeC():
    
    def targetFuncC(x):
        
        print(f'C1={x[0]*1e12:.2f} pF; C2={x[1]*1e12:.2f} pF')
        cf["val_C_m1"] = x[0]
        cf["val_C_m2"] = x[1]
        return calc_dischargePoint()
    
    # Initial guess
    a_x0 = [100e-12, 1000e-12]
    a_bounds = [(1e-12, 5000e-12), (1e-12, 5000e-12)]
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": [(1e-12, 2500e-12), (1e-12, 2500e-12)]}

    #Local optimization
    result = minimize(targetFuncC, x0=a_x0, bounds=a_bounds, method="Nelder-Mead", tol=1e-5) 

    # Perform global optimization
#    result = basinhopping(targetFuncC, T=0.05, x0=[cf["C_m1_init"], cf["C_m2_init"]], minimizer_kwargs=minimizer_kwargs, niter=1000) 
    
    print(result)

def iterateC():
    
    rt["miter"] = 0
#    miter_max = 50
    matching_cond = True
    
    while matching_cond:
        rt["miter"] = rt["miter"] + 1
        print(f'-- matching iteration #{rt["miter"]: =2} starts: C1={cf["val_C_m1"] * 1e12:.2f} [pF], C2={cf["val_C_m2"] * 1e12:.2f} [pF]')
        val_C_m2_prev = cf["val_C_m2"]
        val_C_m1_prev = cf["val_C_m1"]
        
        calc_dischargePoint()
        plot_UI()
        
        # Импеданс на выходе C-C звена для расчета согласования
        (Rmm, Xmm) = calcImpedance_1Harm('3', '0', '4', '5', cf["val_R_m"])
        (cf["val_C_m2"], cf["val_C_m1"]) = calcMatchingNetwork(Rmm, Xmm, 2 * np.pi * cf["f0"], 50)

        beta2 = 1
        cf["val_C_m1"] = val_C_m1_prev + beta2 * (cf["val_C_m1"] - val_C_m1_prev)
        cf["val_C_m2"] = val_C_m2_prev + beta2 * (cf["val_C_m2"] - val_C_m2_prev)

        matching_cond = np.abs(cf["val_C_m1"] - val_C_m1_prev) > 1e-12 or np.abs(cf["val_C_m2"] - val_C_m2_prev) > 1e-12

        print(f'-- dCm1={(cf["val_C_m1"] - val_C_m1_prev) * 1e12:.2f} [pF], dCm2={(cf["val_C_m2"] - val_C_m2_prev) * 1e12:.2f} [pF]', end=' ')

        if matching_cond:
            print(f'<- NEW MATCHING VALUES: C1={cf["val_C_m1"] * 1e12:.2f} C2={cf["val_C_m2"] * 1e12:.2f}\n')
            if cf["val_C_m1"] <= 0 or cf["val_C_m2"] <= 0:
                sys.exit("Wrong C1 or C2 value. STOP.")
            if rt["miter"] >= cf["miter_max"]:
                sys.exit("MATCHING NOT CONVERGED. MATCHING ITERATIONS LIMIT REACHED. STOP.")
        else:
            print(f'<- MATCHING CONVERGED: C1={val_C_m1_prev * 1e12:.2f} C2={val_C_m2_prev * 1e12:.2f}\n')

 
def solveDischargePoint(a_df, optimizeMatching=False):
        redefineRuntimeParams()
        if optimizeMatching:
            iterateC()
        else:
            calc_dischargePoint()
            plot_UI()
        printSimulationResults()

        # Add circuit used in an iteration to report
        subtitle = Paragraph("Ngspice circuit", cf['styles']['Heading2'])
        cf['story_iterations'].append(subtitle)
        cf['story_iterations'].append(Spacer(1, 12))
        text = Paragraph(cf['sim_circ'], cf['styles']['Normal'])
        cf['story_iterations'].append(text)
        
        subtitle = Paragraph("VI plot", cf['styles']['Heading2'])
        cf['story_iterations'].append(subtitle)
        cf['story_iterations'].append(Spacer(1, 12))
        cf['story_iterations'].append(cf['fig_vi'])

        subtitle = Paragraph("Transient plot", cf['styles']['Heading2'])
        cf['story_iterations'].append(subtitle)
        cf['story_iterations'].append(Spacer(1, 12))
        cf['story_iterations'].append(cf['fig_tr'])

        subtitle = Paragraph("Sheath voltage plot", cf['styles']['Heading2'])
        cf['story_iterations'].append(subtitle)
        cf['story_iterations'].append(Spacer(1, 12))
        cf['story_iterations'].append(cf['fig_sh'])

        subtitle = Paragraph("Spectrum plot", cf['styles']['Heading2'])
        cf['story_iterations'].append(subtitle)
        cf['story_iterations'].append(Spacer(1, 12))
        cf['story_iterations'].append(cf['fig_sp'])

        return pd.concat([a_df, cf['pd']], ignore_index=True)


# 'sp' - Проход по давлению с подбором C1, C2 при постоянной _входной_ мощности
# 'sf' - Проход по частоте с подбором C1, C2 при постоянной _входной_ мощности
# 'sn' - Расчет одной точки с подбором C1, C2
# 'mc' - Картирование импеданса нагрузки в зависимости от C1, C2
# 'of' - Подбор частоты на минимизацию отражения при уходе C1, C2 

workmode = 'sn'

df = pd.DataFrame()
cf['next_aaaa'] = get_next_available_aaaa('out/', cf['name'], workmode)
cf['out_path'], cf['current_date'] = create_subdirectory('out/', cf['next_aaaa'], cf['name'], workmode)
# Redirect stdout to the Logger
sys.stdout = Logger(f'{cf["out_path"]}/output.log')    
initReport()
shutil.copy(f'conf/{cf["name"]}.json5', f'{cf["out_path"]}/{cf["name"]}.json5');
shutil.copy(f'conf/{sw["name"]}.json5', f'{cf["out_path"]}/{sw["name"]}.json5');

match workmode:
    case 'sp':
        print(f'Pressure sweep')
        
        addReportPressureIterHeader(sw['sp']['press'])
        
        for i, p in enumerate(sw['sp']['press']):
            cf["p0"] = p
            
            pstr = f'=== SWEEP STEP #{i+1}: p0={p} [Pa]'
            print(pstr)

            subtitle = Paragraph(pstr, cf['styles']['Heading2'])
            cf['story_iterations'].append(subtitle)
            cf['story_iterations'].append(Spacer(1, 36))
            
            df = solveDischargePoint(df, True)
        plot_sweepResult(df)
            
    case 'sf':
            print(f'Frequency sweep')
        
            addReportFrequencyIterHeader(sw['sf']['freqs'], sw['sf']['inds'])
            
            for i in range(len(sw['sf']['freqs'])):
                cf['f0'] = sw['sf']['freqs'][i]
                cf['val_L_m2'] = sw['sf']['inds'][i]
                cf['beta'] = sw['sf']['beta_override'][i]
                cf['max_iter_ne'] = sw['sf']['max_iter_ne_override'][i]
                cf['C_m1_init'] = sw['sf']['Cm1_override'][i]
                cf['C_m2_init'] = sw['sf']['Сm2_override'][i]
                cf["ne_init"] = sw['sf']['ne_init_override'][i]
                cf["num_periods_sim"] = sw['sf']['num_periods_sim_override'][i]

                pstr = f'=== SWEEP STEP #{i+1}: f0={cf["f0"]/1e6} [MHz], L_m2={cf["val_L_m2"]*1e9} [nH], C_m1={cf["C_m1_init"]*1e12} [pF], beta={cf["beta"]}, max_iter_ne={cf["max_iter_ne"]}'
                print(pstr)

                df = solveDischargePoint(df, True)
            plot_sweepFreqResult(df)

    case 'sn':
            print(f'Single point simulation')
            df = solveDischargePoint(df, optimizeMatching=True)

    case 'of':
            print(f'C1, C2 deviation mode')
            N = 10
            percentage = 20;
            C1s, C2s = sample_deviatedC([cf["val_C_m1"], cf["val_C_m2"]], [percentage, percentage], N, linear=True, seed=None, symmetric=True)
            f0_init = cf["f0"]
            
            ii = 0
            for i in range(N):
                for j in range(N):
                    ii = ii + 1
                    print(f'=== CAPS SAMPLE #{ii} of {N*N}: C1={C1s[i]*1e12:.1f} [pF], C2={C2s[i]*1e12:.1f} [pF]')
                    redefineRuntimeParams()
                    cf["f0"] = f0_init 
                    cf["val_C_m1"] = C1s[i]
                    cf["val_C_m2"] = C2s[j]
                    optimizeF()
                    printSimulationResults()
                    df = pd.concat([df, cf['pd']], ignore_index=True)
            plot_devResult(df)    
            
    case 'mc':
            redefineRuntimeParams()

            start_val_x = 1200e-12      # 1 ps
            end_val_x = 1550e-12     # 2500 ps
            step_size_x = 5e-12      # 5 ps

            # Calculate number of steps
            num_steps_x = int((end_val_x - start_val_x) / step_size_x) + 1


            start_val_y = 2800e-12      # 1 ps
            end_val_y = 4800e-12     # 2500 ps
            step_size_y = 5e-12      # 5 ps

            # Calculate number of steps
            num_steps_y = int((end_val_y - start_val_y) / step_size_y) + 1

            # Initialize 2D results array
            results = np.zeros((num_steps_x, num_steps_y))

            ii = 0
            # Nested loops with integer iterators
            for i in range(num_steps_x):
                x = start_val_x + i * step_size_x
                for j in range(num_steps_y):
                    ii = ii + 1
                    print(ii)
                    y = start_val_y + j * step_size_y
                    cf["val_C_m1"] = x[0]
                    cf["val_C_m2"] = x[1]
                    df = solveDischargePoint(df, False)
                    result[i, j] = df

            # Create coordinate arrays for plotting
            x_vals = np.arange(0, num_steps_x) * step_size_x + start_val_x
            y_vals = np.arange(0, num_steps_y) * step_size_y + start_val_y

            # Create contour plot
            plt.figure(figsize=(10, 8))
            contour = plt.contourf(y_vals * 1e12, x_vals * 1e12, results, levels=20, cmap='viridis')
            plt.colorbar(contour, label='Gamma')
            plt.xlabel('C1 [pF]')
            plt.ylabel('C2 [pF]')
            plt.title('Contour Plot of Bruteforced Values')
            plt.grid(True)
            plt.show()

    case _:
        sys.exit("Unknown work mode. STOP.")
        
df.to_excel(f'{cf["out_path"]}/{cf["name"]}_{workmode}_{cf["next_aaaa"]:04d}_{cf["current_date"]}_tables.xlsx', index=False)
finalizeReport()

#TODO Сделать расчет скорости и селективности травления через оценку IEDF
#TODO Приделать поиск оптимальных значений последовательной индуктивности и емкости на разных частотах в заданных рамках
