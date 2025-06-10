import json5
from typing import Any

#################
# 1. Определяем физические константы
#################

ct = {
    "qe": 1.6e-19,                        # Заряд электрона [Кл]
    "me": 9.11e-31,                       # Масса электрона [кг]
    "Mi": 6.6335209e-26 - 9.1093837e-31,  # Масса иона Ar [кг]
    "eps_0": 8.85e-12,                    # Диэлектрическая постоянная [Ф/м]
    "k_B": 1.380649e-23                   # Постоянная Больцмана [Дж/К]
}


def loadConf(a_cfname):
    with open(f"conf/{a_cfname}.json5", 'r') as file:
        data = json5.load(file)

    o_cf: dict[str, Any] = {}

    o_cf["name"] = data["name"]
    o_cf["comment"] = data["comment"]
    o_cf["Vm"] = data["Vm"]
    o_cf["f0"] = data["f0"]
    o_cf["p0"] = data["p0"]
    o_cf["T0"] = data["T0"]
    o_cf["ne"] = data["ne"]
    o_cf["beta"] = data["beta"]
    o_cf["l_B"] = data["l_B"]
    o_cf["Ae"] = data["Ae"]
    o_cf["Ag"] = data["Ag"]
    o_cf["val_R_rf"] = data["val_R_rf"]
    o_cf["val_C_m1"] = data["val_C_m1"]
    o_cf["val_C_m2"] = data["val_C_m2"]
    o_cf["val_L_m2"] = data["val_L_m2"]
    o_cf["val_R_m"] = data["val_R_m"]
    o_cf["val_C_stray"] = data["val_C_stray"]
    o_cf["val_R_stray"] = data["val_R_stray"]
    o_cf["matching_flag"] = data["matching_flag"]
    o_cf["eps_ne"] = data["eps_ne"]
    o_cf["max_iter_ne"] = data["max_iter_ne"]
    o_cf["verbose_plots"] = data["verbose_plots"]

    return o_cf


#################
# 2. Загружаем параметры рабочей точки ВЧ разряда
#################

cfname = "pc1"

cf = loadConf(cfname)
# Размер зеркала макета М1: 250х185 -> площадь 0.0462 м2, емкость при d=1 мм: 410 пФ
# Значения Rm и Rstray примерно соответствуют расчетам СУ с потерями

def redefineRuntimeParams():

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

redefineRuntimeParams()