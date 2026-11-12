"""Diccionario de datos del dataset UCI 697.

*Predict Students' Dropout and Academic Success* (Realinho et al., 2022).

Este módulo es la **fuente única de verdad** sobre:

* El nombre interno (snake_case) de cada columna y su nombre original en UCI.
* La etiqueta en español que se muestra en la interfaz.
* El tipo de variable (categórica nominal, binaria, numérica) y su unidad.
* Los códigos de las variables categóricas y su traducción al español.

Los códigos y sus significados se tomaron de la descripción oficial de
variables del repositorio UCI (https://archive.ics.uci.edu/dataset/697). Las
traducciones al español son propias; ante cualquier duda prevalece el texto
original en inglés de UCI.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Variable objetivo
# ---------------------------------------------------------------------------
TARGET = "target"
TARGET_ORIGINAL = "Target"
CLASSES = ["Dropout", "Enrolled", "Graduate"]
CLASS_LABELS_ES = {
    "Dropout": "Deserción",
    "Enrolled": "Matriculado",
    "Graduate": "Graduado",
}
CLASS_COLORS = {
    "Dropout": "#D62F4B",
    "Enrolled": "#D99000",
    "Graduate": "#2A78D6",
}

# ---------------------------------------------------------------------------
# Mapeo de nombres originales (UCI) -> nombres internos
# Nota: en el CSV original la columna "Daytime/evening attendance" contiene un
# tabulador al final y el archivo inicia con BOM UTF-8; ambos se limpian en
# ``data_loader.clean_column_names`` antes de aplicar este mapeo.
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "Marital status": "marital_status",
    "Application mode": "application_mode",
    "Application order": "application_order",
    "Course": "course",
    "Daytime/evening attendance": "daytime_evening_attendance",
    "Previous qualification": "previous_qualification",
    "Previous qualification (grade)": "previous_qualification_grade",
    "Nacionality": "nationality",
    "Mother's qualification": "mothers_qualification",
    "Father's qualification": "fathers_qualification",
    "Mother's occupation": "mothers_occupation",
    "Father's occupation": "fathers_occupation",
    "Admission grade": "admission_grade",
    "Displaced": "displaced",
    "Educational special needs": "educational_special_needs",
    "Debtor": "debtor",
    "Tuition fees up to date": "tuition_fees_up_to_date",
    "Gender": "gender",
    "Scholarship holder": "scholarship_holder",
    "Age at enrollment": "age_at_enrollment",
    "International": "international",
    "Curricular units 1st sem (credited)": "curricular_units_1st_sem_credited",
    "Curricular units 1st sem (enrolled)": "curricular_units_1st_sem_enrolled",
    "Curricular units 1st sem (evaluations)": "curricular_units_1st_sem_evaluations",
    "Curricular units 1st sem (approved)": "curricular_units_1st_sem_approved",
    "Curricular units 1st sem (grade)": "curricular_units_1st_sem_grade",
    "Curricular units 1st sem (without evaluations)": "curricular_units_1st_sem_without_evaluations",
    "Curricular units 2nd sem (credited)": "curricular_units_2nd_sem_credited",
    "Curricular units 2nd sem (enrolled)": "curricular_units_2nd_sem_enrolled",
    "Curricular units 2nd sem (evaluations)": "curricular_units_2nd_sem_evaluations",
    "Curricular units 2nd sem (approved)": "curricular_units_2nd_sem_approved",
    "Curricular units 2nd sem (grade)": "curricular_units_2nd_sem_grade",
    "Curricular units 2nd sem (without evaluations)": "curricular_units_2nd_sem_without_evaluations",
    "Unemployment rate": "unemployment_rate",
    "Inflation rate": "inflation_rate",
    "GDP": "gdp",
    "Target": TARGET,
}

# ---------------------------------------------------------------------------
# Tipos de variables
# ---------------------------------------------------------------------------
# Categóricas nominales: el número es un código, no una magnitud. Se codifican
# con One-Hot (regresión logística / random forest) o como categóricas nativas
# (HistGradientBoosting). Nunca se escalan como números.
NOMINAL_FEATURES = [
    "marital_status",
    "application_mode",
    "course",
    "previous_qualification",
    "nationality",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
]

# Binarias 0/1: se dejan como están.
BINARY_FEATURES = [
    "daytime_evening_attendance",
    "displaced",
    "educational_special_needs",
    "debtor",
    "tuition_fees_up_to_date",
    "gender",
    "scholarship_holder",
    "international",
]

# Numéricas (discretas o continuas).
NUMERIC_FEATURES = [
    "application_order",
    "previous_qualification_grade",
    "admission_grade",
    "age_at_enrollment",
    "curricular_units_1st_sem_credited",
    "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations",
    "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade",
    "curricular_units_1st_sem_without_evaluations",
    "curricular_units_2nd_sem_credited",
    "curricular_units_2nd_sem_enrolled",
    "curricular_units_2nd_sem_evaluations",
    "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_grade",
    "curricular_units_2nd_sem_without_evaluations",
    "unemployment_rate",
    "inflation_rate",
    "gdp",
]

SECOND_SEMESTER_FEATURES = [c for c in NUMERIC_FEATURES if c.startswith("curricular_units_2nd_sem_")]

ALL_FEATURES = NOMINAL_FEATURES + BINARY_FEATURES + NUMERIC_FEATURES

# Rangos válidos documentados (mínimo, máximo). Se usan para validar el
# esquema de entrada; no para eliminar outliers.
VALID_RANGES = {
    "application_order": (0, 9),
    "previous_qualification_grade": (0.0, 200.0),
    "admission_grade": (0.0, 200.0),
    "age_at_enrollment": (15, 100),
    "curricular_units_1st_sem_credited": (0, 60),
    "curricular_units_1st_sem_enrolled": (0, 60),
    "curricular_units_1st_sem_evaluations": (0, 100),
    "curricular_units_1st_sem_approved": (0, 60),
    "curricular_units_1st_sem_grade": (0.0, 20.0),
    "curricular_units_1st_sem_without_evaluations": (0, 60),
    "curricular_units_2nd_sem_credited": (0, 60),
    "curricular_units_2nd_sem_enrolled": (0, 60),
    "curricular_units_2nd_sem_evaluations": (0, 100),
    "curricular_units_2nd_sem_approved": (0, 60),
    "curricular_units_2nd_sem_grade": (0.0, 20.0),
    "curricular_units_2nd_sem_without_evaluations": (0, 60),
    "unemployment_rate": (0.0, 100.0),
    "inflation_rate": (-50.0, 100.0),
    "gdp": (-50.0, 50.0),
}

# ---------------------------------------------------------------------------
# Etiquetas en español y unidades
# ---------------------------------------------------------------------------
FEATURE_INFO = {
    "marital_status": ("Estado civil", "código"),
    "application_mode": ("Modalidad de admisión", "código"),
    "application_order": ("Orden de preferencia de la solicitud", "0 = primera opción … 9 = última"),
    "course": ("Programa académico", "código"),
    "daytime_evening_attendance": ("Jornada", "1 = diurna, 0 = nocturna"),
    "previous_qualification": ("Formación previa", "código"),
    "previous_qualification_grade": ("Nota de la formación previa", "escala 0–200"),
    "nationality": ("Nacionalidad", "código"),
    "mothers_qualification": ("Nivel educativo de la madre", "código"),
    "fathers_qualification": ("Nivel educativo del padre", "código"),
    "mothers_occupation": ("Ocupación de la madre", "código"),
    "fathers_occupation": ("Ocupación del padre", "código"),
    "admission_grade": ("Nota de admisión", "escala 0–200"),
    "displaced": ("Estudiante desplazado de su residencia", "1 = sí, 0 = no"),
    "educational_special_needs": ("Necesidades educativas especiales", "1 = sí, 0 = no"),
    "debtor": ("Tiene deudas con la institución", "1 = sí, 0 = no"),
    "tuition_fees_up_to_date": ("Matrícula al día", "1 = sí, 0 = no"),
    "gender": ("Sexo", "1 = masculino, 0 = femenino"),
    "scholarship_holder": ("Becario", "1 = sí, 0 = no"),
    "age_at_enrollment": ("Edad al matricularse", "años"),
    "international": ("Estudiante internacional", "1 = sí, 0 = no"),
    "curricular_units_1st_sem_credited": ("1.er sem.: unidades convalidadas", "n.º de asignaturas"),
    "curricular_units_1st_sem_enrolled": ("1.er sem.: unidades inscritas", "n.º de asignaturas"),
    "curricular_units_1st_sem_evaluations": ("1.er sem.: evaluaciones presentadas", "n.º de evaluaciones"),
    "curricular_units_1st_sem_approved": ("1.er sem.: unidades aprobadas", "n.º de asignaturas"),
    "curricular_units_1st_sem_grade": ("1.er sem.: promedio de notas", "escala 0–20"),
    "curricular_units_1st_sem_without_evaluations": ("1.er sem.: unidades sin evaluación", "n.º de asignaturas"),
    "curricular_units_2nd_sem_credited": ("2.º sem.: unidades convalidadas", "n.º de asignaturas"),
    "curricular_units_2nd_sem_enrolled": ("2.º sem.: unidades inscritas", "n.º de asignaturas"),
    "curricular_units_2nd_sem_evaluations": ("2.º sem.: evaluaciones presentadas", "n.º de evaluaciones"),
    "curricular_units_2nd_sem_approved": ("2.º sem.: unidades aprobadas", "n.º de asignaturas"),
    "curricular_units_2nd_sem_grade": ("2.º sem.: promedio de notas", "escala 0–20"),
    "curricular_units_2nd_sem_without_evaluations": ("2.º sem.: unidades sin evaluación", "n.º de asignaturas"),
    "unemployment_rate": ("Tasa de desempleo (Portugal)", "%"),
    "inflation_rate": ("Tasa de inflación (Portugal)", "%"),
    "gdp": ("PIB (Portugal)", "variación %"),
    # Variables derivadas (ver preprocessing.AcademicFeatureEngineer)
    "curricular_units_1st_sem_approval_rate": ("1.er sem.: tasa de aprobación", "aprobadas / inscritas"),
    "curricular_units_2nd_sem_approval_rate": ("2.º sem.: tasa de aprobación", "aprobadas / inscritas"),
}


def label(col: str) -> str:
    """Etiqueta legible en español para una columna interna."""
    return FEATURE_INFO.get(col, (col, ""))[0]


# ---------------------------------------------------------------------------
# Códigos de variables categóricas (UCI) traducidos al español
# ---------------------------------------------------------------------------
MARITAL_STATUS = {
    1: "Soltero/a",
    2: "Casado/a",
    3: "Viudo/a",
    4: "Divorciado/a",
    5: "Unión de hecho",
    6: "Separado/a legalmente",
}

APPLICATION_MODE = {
    1: "1.ª fase – contingente general",
    2: "Ordenanza n.º 612/93",
    5: "1.ª fase – contingente especial (Azores)",
    7: "Titulares de otros cursos superiores",
    10: "Ordenanza n.º 854-B/99",
    15: "Estudiante internacional (grado)",
    16: "1.ª fase – contingente especial (Madeira)",
    17: "2.ª fase – contingente general",
    18: "3.ª fase – contingente general",
    26: "Ordenanza n.º 533-A/99, b2 (plan diferente)",
    27: "Ordenanza n.º 533-A/99, b3 (otra institución)",
    39: "Mayores de 23 años",
    42: "Traslado",
    43: "Cambio de programa",
    44: "Titulares de diploma de especialización tecnológica",
    51: "Cambio de institución/programa",
    53: "Titulares de diploma de ciclo corto",
    57: "Cambio de institución/programa (internacional)",
}

COURSE = {
    33: "Tecnologías de Producción de Biocombustibles",
    171: "Diseño de Animación y Multimedia",
    8014: "Servicio Social (nocturno)",
    9003: "Agronomía",
    9070: "Diseño de Comunicación",
    9085: "Enfermería Veterinaria",
    9119: "Ingeniería Informática",
    9130: "Equinocultura",
    9147: "Gestión",
    9238: "Servicio Social",
    9254: "Turismo",
    9500: "Enfermería",
    9556: "Higiene Oral",
    9670: "Gestión de Publicidad y Marketing",
    9773: "Periodismo y Comunicación",
    9853: "Educación Básica",
    9991: "Gestión (nocturno)",
}

PREVIOUS_QUALIFICATION = {
    1: "Educación secundaria",
    2: "Educación superior – bachillerato",
    3: "Educación superior – licenciatura",
    4: "Educación superior – maestría",
    5: "Educación superior – doctorado",
    6: "Educación superior cursada (sin título)",
    9: "12.º año – no completado",
    10: "11.º año – no completado",
    12: "Otro – 11.º año",
    14: "10.º año",
    15: "10.º año – no completado",
    19: "Educación básica 3.er ciclo (9.º/10.º/11.º) o equiv.",
    38: "Educación básica 2.º ciclo (6.º/7.º/8.º) o equiv.",
    39: "Curso de especialización tecnológica",
    40: "Educación superior – licenciatura (1.er ciclo)",
    42: "Curso técnico superior profesional",
    43: "Educación superior – maestría (2.º ciclo)",
}

NATIONALITY = {
    1: "Portuguesa",
    2: "Alemana",
    6: "Española",
    11: "Italiana",
    13: "Neerlandesa",
    14: "Inglesa",
    17: "Lituana",
    21: "Angoleña",
    22: "Caboverdiana",
    24: "Guineana",
    25: "Mozambiqueña",
    26: "Santotomense",
    32: "Turca",
    41: "Brasileña",
    62: "Rumana",
    100: "Moldava",
    101: "Mexicana",
    103: "Ucraniana",
    105: "Rusa",
    108: "Cubana",
    109: "Colombiana",
}

PARENT_QUALIFICATION = {
    1: "Secundaria – 12.º año o equiv.",
    2: "Superior – bachillerato",
    3: "Superior – licenciatura",
    4: "Superior – maestría",
    5: "Superior – doctorado",
    6: "Educación superior cursada (sin título)",
    9: "12.º año – no completado",
    10: "11.º año – no completado",
    11: "7.º año (plan antiguo)",
    12: "Otro – 11.º año",
    13: "2.º año de curso complementario de secundaria",
    14: "10.º año",
    18: "Curso general de comercio",
    19: "Básica 3.er ciclo (9.º/10.º/11.º) o equiv.",
    20: "Curso complementario de secundaria",
    22: "Curso técnico-profesional",
    25: "Curso complementario de secundaria – no concluido",
    26: "7.º año de escolaridad",
    27: "2.º ciclo del curso general de secundaria",
    29: "9.º año – no completado",
    30: "8.º año de escolaridad",
    31: "Curso general de administración y comercio",
    33: "Curso suplementario de contabilidad y administración",
    34: "Desconocido",
    35: "No sabe leer ni escribir",
    36: "Sabe leer sin haber cursado 4.º año",
    37: "Básica 1.er ciclo (4.º/5.º) o equiv.",
    38: "Básica 2.º ciclo (6.º/7.º/8.º) o equiv.",
    39: "Curso de especialización tecnológica",
    40: "Superior – licenciatura (1.er ciclo)",
    41: "Curso de estudios superiores especializados",
    42: "Curso técnico superior profesional",
    43: "Superior – maestría (2.º ciclo)",
    44: "Superior – doctorado (3.er ciclo)",
}

PARENT_OCCUPATION = {
    0: "Estudiante",
    1: "Poder legislativo/ejecutivo, directivos y gerentes",
    2: "Especialistas en actividades intelectuales y científicas",
    3: "Técnicos y profesionales de nivel intermedio",
    4: "Personal administrativo",
    5: "Servicios personales, seguridad y vendedores",
    6: "Agricultores y trabajadores calificados (agro, pesca, forestal)",
    7: "Trabajadores calificados de industria, construcción y artesanos",
    8: "Operadores de instalaciones y máquinas, ensambladores",
    9: "Trabajadores no calificados",
    10: "Fuerzas armadas",
    90: "Otra situación",
    99: "(En blanco)",
    101: "Oficiales de las fuerzas armadas",
    102: "Sargentos de las fuerzas armadas",
    103: "Otro personal de las fuerzas armadas",
    112: "Directores de servicios administrativos y comerciales",
    114: "Directores de hotelería, restauración, comercio y otros servicios",
    121: "Especialistas en ciencias físicas, matemáticas e ingeniería",
    122: "Profesionales de la salud",
    123: "Docentes",
    124: "Especialistas en finanzas, contabilidad y administración",
    125: "Especialistas en TIC",
    131: "Técnicos de nivel intermedio en ciencias e ingeniería",
    132: "Técnicos de nivel intermedio en salud",
    134: "Técnicos de nivel intermedio en servicios jurídicos, sociales, deportivos y culturales",
    135: "Técnicos en tecnologías de la información y comunicación",
    141: "Oficinistas, secretarios y operadores de datos",
    143: "Operadores de datos, contabilidad, estadística y finanzas",
    144: "Otro personal de apoyo administrativo",
    151: "Trabajadores de servicios personales",
    152: "Vendedores",
    153: "Trabajadores de cuidados personales",
    154: "Personal de protección y seguridad",
    161: "Agricultores y trabajadores agropecuarios orientados al mercado",
    163: "Agricultores, ganaderos, pescadores de subsistencia",
    171: "Trabajadores calificados de la construcción (excepto electricistas)",
    172: "Trabajadores calificados de metalurgia y similares",
    173: "Trabajadores calificados de imprenta, precisión, joyería y artesanía",
    174: "Trabajadores calificados en electricidad y electrónica",
    175: "Trabajadores de alimentos, madera, confección y otras industrias",
    181: "Operadores de instalaciones fijas y máquinas",
    182: "Trabajadores de ensamblaje",
    183: "Conductores de vehículos y operadores de equipos móviles",
    191: "Trabajadores de limpieza",
    192: "No calificados en agricultura, ganadería, pesca y forestal",
    193: "No calificados en industria extractiva, construcción, manufactura y transporte",
    194: "Asistentes de preparación de alimentos",
    195: "Vendedores ambulantes (excepto alimentos) y servicios callejeros",
}

BINARY_LABELS = {
    "daytime_evening_attendance": {1: "Diurna", 0: "Nocturna"},
    "gender": {1: "Masculino", 0: "Femenino"},
}
YES_NO = {1: "Sí", 0: "No"}

CATEGORY_CODES = {
    "marital_status": MARITAL_STATUS,
    "application_mode": APPLICATION_MODE,
    "course": COURSE,
    "previous_qualification": PREVIOUS_QUALIFICATION,
    "nationality": NATIONALITY,
    "mothers_qualification": PARENT_QUALIFICATION,
    "fathers_qualification": PARENT_QUALIFICATION,
    "mothers_occupation": PARENT_OCCUPATION,
    "fathers_occupation": PARENT_OCCUPATION,
}


def code_label(col: str, code: int) -> str:
    """Traduce un código categórico a su descripción en español."""
    if col in CATEGORY_CODES:
        return CATEGORY_CODES[col].get(int(code), f"Código {int(code)}")
    if col in BINARY_LABELS:
        return BINARY_LABELS[col].get(int(code), str(code))
    if col in BINARY_FEATURES:
        return YES_NO.get(int(code), str(code))
    return str(code)
