from typing import Any
import json5
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

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

    o_cf["name"] = a_cfname
    o_cf["comment"] = data["comment"]
    o_cf["Vm"] = data["Vm"]
    o_cf["f0"] = data["f0"]
    o_cf["p0"] = data["p0"]
    o_cf["T0"] = data["T0"]
    o_cf["ne"] = data["ne"]
    o_cf["ne_init"] = data["ne"]
    o_cf["beta"] = data["beta"]
    o_cf["l_B"] = data["l_B"]
    o_cf["Ae"] = data["Ae"]
    o_cf["Ag"] = data["Ag"]
    o_cf["val_R_rf"] = data["val_R_rf"]
    o_cf["val_C_m1"] = data["val_C_m1"]
    o_cf["val_C_m2"] = data["val_C_m2"]
    o_cf["C_m1_init"] = data["val_C_m1"]
    o_cf["C_m2_init"] = data["val_C_m2"]
    o_cf["val_L_m2"] = data["val_L_m2"]
    o_cf["val_R_m"] = data["val_R_m"]
    o_cf["val_C_stray"] = data["val_C_stray"]
    o_cf["val_R_stray"] = data["val_R_stray"]
    o_cf["matching_flag"] = data["matching_flag"]
    o_cf["eps_ne"] = data["eps_ne"]
    o_cf["max_iter_ne"] = data["max_iter_ne"]
    o_cf["verbose_plots"] = data["verbose_plots"]
    o_cf["verbose_circuit"] = data["verbose_circuit"]
    o_cf['verbose_circ_plots'] = data['verbose_circ_plots']
    o_cf["num_periods_sim"] = data['num_periods_sim']  # Количество периодов ВЧ поля, которое надо просчитать
    o_cf["cooling"] = data["cooling"]
    o_cf["steady_state_threshold"] = data["steady_state_threshold"]
    o_cf["max_periods"] = data["max_periods"]
    o_cf["miter_max"] = data["miter_max"]
    o_cf["version"] = '0.5.002'

    return o_cf

def loadSweeps(a_cfname):
    with open(f"conf/{a_cfname}.json5", 'r') as file:
        data = json5.load(file)

    s_cf: dict[str, Any] = {}

    s_cf["comment"] = data["comment"]
    s_cf["sp"] = data["sp"]
    s_cf["sf"] = data["sf"]
    s_cf["mc"] = data["mc"]
    s_cf["of"] = data["of"]
    s_cf["name"] = a_cfname;

    return s_cf

def add_header_footer(canvas, doc):
    # Сохраняем текущее состояние canvas
    canvas.saveState()
    
    # Верхний колонтитул (header)
    header_text = f"rfd ver. {cf['version']}; config {cf['name']}; run# {cf['next_aaaa']}; date {cf['current_date']}"

    # Убедимся, что текст не выходит за пределы страницы
    header_y = A4[1] - 20*mm  # 20 мм от верхнего края
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawCentredString(A4[0]/2, header_y, header_text)  # По центру
    
    # Линия под колонтитулом
    canvas.line(20*mm, header_y - 2*mm, A4[0] - 20*mm, header_y - 2*mm)    
    
    # Нижний колонтитул (footer)
    footer_text = f"Page {doc.page}"
    canvas.setFont('Helvetica', 8)
    canvas.drawCentredString(A4[0]/2, 20, footer_text)
    
    # Линия разделитель
    canvas.line(50, A4[1] - 40, A4[0] - 50, A4[1] - 40)
    canvas.line(50, 30, A4[0] - 50, 30)
    
    # Восстанавливаем состояние canvas
    canvas.restoreState()    

def initReport():
    # Create PDF document
    cf['doc'] = SimpleDocTemplate(f"{cf['out_path']}/{cf['next_aaaa']:04d}_{cf['name']}_{cf['current_date']}_report.pdf", pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    
    # Prepare styles
    cf['styles'] = getSampleStyleSheet()
#    styles.add(ParagraphStyle(name='Heading2', 
#                             fontSize=14, 
#                             leading=16,
#                             spaceAfter=12,
#                             textColor=colors.darkblue))
    
    # Story will hold all the flowables (elements) of the document
    cf['story'] = []
    cf['story_iterations'] = []
    
def finalizeReport():
    # Build the PDF
    trueStory = cf['story'] + cf['story_iterations']
    cf['doc'].build(trueStory, onFirstPage=add_header_footer, onLaterPages=add_header_footer)

    
#################
# 2. Загружаем параметры рабочей точки ВЧ разряда
#################

#cfname = 'test_of_mode'
cfname = 'dts6'
sfname = 'sweeps-dts6'

cf = loadConf(cfname)
sw = loadSweeps(sfname)

rt: dict[str, Any] = {};
#rt.setdefault("miter", 0)
#rt.setdefault("iter_no", 0)

# Размер зеркала макета М1: 250х185 -> площадь 0.0462 м2, емкость при d=1 мм: 410 пФ
# Значения Rm и Rstray примерно соответствуют расчетам СУ с потерями
