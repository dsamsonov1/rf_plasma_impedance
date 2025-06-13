from typing import Any
import json5

#################
# 1. Определяем физические константы
#################

ct = {
    "qe": 1.6e-19,  # Заряд электрона [Кл]
    "me": 9.11e-31,  # Масса электрона [кг]
    "Mi": 6.6335209e-26 - 9.1093837e-31,  # Масса иона Ar [кг]
    "eps_0": 8.85e-12,  # Диэлектрическая постоянная [Ф/м]
    "k_B": 1.380649e-23  # Постоянная Больцмана [Дж/К]
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
    o_cf["cooling"] = data["cooling"]

    return o_cf


#################
# 2. Загружаем параметры рабочей точки ВЧ разряда
#################

cfname = 'as1-c'

cf = loadConf(cfname)

# Размер зеркала макета М1: 250х185 -> площадь 0.0462 м2, емкость при d=1 мм: 410 пФ
# Значения Rm и Rstray примерно соответствуют расчетам СУ с потерями
