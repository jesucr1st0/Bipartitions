# tests/contar_casos.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from funcs.cargar import excel_a_configs

EXCEL = os.path.join(os.path.dirname(__file__), "..", "data", "DatosPruebas2026_1.xlsx")
hojas = {
    "N10A": "10A-Elementos",
    "N15B": "15B-Elementos",
    "N20A": "20A-Elementos",
    "N22A": "22A-Elementos",
    "N25A": "25A-Elementos ",
}
for sistema, hoja in hojas.items():
    try:
        configs = excel_a_configs(EXCEL, hoja)
        print(f"  {sistema}: {len(configs)} casos")
    except Exception as e:
        print(f"  {sistema}: ERROR - {e}")