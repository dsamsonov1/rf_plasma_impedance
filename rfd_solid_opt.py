import sys
import os
from rfd_plots import *
from rfd_utils import *
from rfd_conf import *
import pandas as pd
from datetime import datetime

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
    
    miter = 0
    miter_max = 50
    matching_cond = True
    
    while matching_cond:
        miter = miter + 1
        print(f'-- matching iteration #{miter: =2} starts: C1={cf["val_C_m1"] * 1e12:.2f} [pF], C2={cf["val_C_m2"] * 1e12:.2f} [pF]')
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

        print(f'-- dCm1={np.abs(cf["val_C_m1"] - val_C_m1_prev) * 1e12:.2f} [pF], dCm2={np.abs(cf["val_C_m2"] - val_C_m2_prev) * 1e12:.2f} [pF]', end=' ')

        if matching_cond:
            print(f'<- NEW MATCHING VALUES: C1={cf["val_C_m1"] * 1e12:.2f} C2={cf["val_C_m2"] * 1e12:.2f}\n')
            if cf["val_C_m1"] <= 0 or cf["val_C_m2"] <= 0:
                sys.exit("Wrong C1 or C2 value. STOP.")
            if miter >= miter_max:
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
        return pd.concat([a_df, cf['pd']], ignore_index=True)


# 'sp' - Проход по давлению с подбором C1, C2 при постоянной _входной_ мощности
# 'sf' - Проход по частоте с подбором C1, C2 при постоянной _входной_ мощности
# 'sn' - Расчет одной точки с подбором C1, C2
# 'mc' - Картирование импеданса нагрузки в зависимости от C1, C2
# 'of' - Подбор частоты на минимизацию отражения при уходе C1, C2 

workmode = 'of'

df = pd.DataFrame()
cf['next_aaaa'] = get_next_available_aaaa('out/', cf['name'])
cf['out_path'], cf['current_date'] = create_subdirectory('out/', cf['next_aaaa'], cf['name'])
# Redirect stdout to the Logger
sys.stdout = Logger(f'{cf['out_path']}/output.log')    

match workmode:
    case 'sp':
        print(f'Pressure sweep')
        
        press = [1, 2.5, 5, 7.5, 10]

        for i, p in enumerate(press):
            cf["p0"] = p
            
            print(f'=== SWEEP STEP #{i}: p0={p} Pa')
            df = solveDischargePoint(df, True)
        plot_sweepResult(df)
            
    case 'sf':
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
            df = solveDischargePoint(df, True)

    case 'sn':
            df = solveDischargePoint(df, True)

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
        
df.to_excel(f'{cf['out_path']}/{cf['next_aaaa']:04d}_{cf['name']}_{cf['current_date']}.xlsx', index=False)


#TODO Копировать исходный файл конфигурации модели в каталог с результатами расчета
#TODO Сделать PDF отчет, включающий графики каждой итерации для контроля качества модели
#TODO Сделать расчет скорости и селективности травления через оценку IEDF
#TODO Приделать контроль установившегося режима в цепи (breakpoint в ngspice?)