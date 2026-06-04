import os

import numpy as np

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


def csv_to_tpm(nombre: str) -> np.ndarray:
    path = os.path.join(_PROJECT_ROOT, "src", ".samples", f"{nombre}.csv")
    return np.genfromtxt(path, delimiter=",")


ABECEDARY = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def excel_str_a_bits(excel_str: str, sistema: str) -> str:
    bits = []
    for letra in sistema:
        bits.append("1" if letra in excel_str else "0")
    return "".join(bits)


def excel_a_configs(path_excel: str, hoja: str) -> list[dict]:
    import openpyxl
    wb = openpyxl.load_workbook(path_excel)
    ws = wb[hoja]

    rows = list(ws.iter_rows(values_only=True))

    estado_inicial = str(int(rows[0][1]))
    sistema = rows[1][1]

    configs = []
    for row in rows[5:]:
        alcance_excel = row[1]
        mecanismo_excel = row[2]
        if alcance_excel is None or mecanismo_excel is None:
            continue

        configs.append(
            {
                "estado_inicial": estado_inicial,
                "sistema": sistema,
                "alcance": excel_str_a_bits(alcance_excel, sistema),
                "mecanismo": excel_str_a_bits(mecanismo_excel, sistema),
                "alcance_excel": alcance_excel,
                "mecanismo_excel": mecanismo_excel,
            }
        )
    return configs
