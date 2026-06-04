import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time

import numpy as np

from funcs.cargar import csv_to_tpm, excel_a_configs
from controllers.strategies.kgeometric import KGeometric
from controllers.strategies.kqnodes import KQNodes

EXCEL_PATH = "data/DatosPruebas2026_1.xlsx"
MAX_CASOS = 5
K_VALS = [2]
ESTRATEGIAS = {
    "KGeometric": KGeometric,
    "KQNodes": KQNodes,
}


def probar(sistema: str, k: int):
    print(f"\n{'='*60}")
    print(f"  SISTEMA: {sistema}  k={k}")
    print(f"{'='*60}")

    configs = excel_a_configs(EXCEL_PATH, f"{sistema}-Elementos")
    if not configs:
        print("  No se encontraron configuraciones en el Excel")
        return

    print(f"  Total configs en Excel: {len(configs)}")
    configs = configs[:MAX_CASOS]

    tpm = csv_to_tpm(sistema)
    print(f"  TPM: {tpm.shape[0]} estados x {tpm.shape[1]} variables")

    for i, cfg in enumerate(configs):
        print(f"\n  --- Caso {i+1} ---")
        print(f"  Alcance:  {cfg['alcance_excel']} -> {cfg['alcance']}")
        print(f"  Mecanismo: {cfg['mecanismo_excel']} -> {cfg['mecanismo']}")
        print(f"  Estado inicial: {cfg['estado_inicial']}")

        for nombre, Estrategia in ESTRATEGIAS.items():
            t0 = time.time()
            estrategia = Estrategia(tpm)
            try:
                resultado = estrategia.find_k_mip(
                    k,
                    cfg["estado_inicial"],
                    "1" * len(cfg["estado_inicial"]),
                    cfg["alcance"],
                    cfg["mecanismo"],
                )
                dt = time.time() - t0
                print(
                    f"  [{nombre:12s}] pérdida={resultado.perdida:.6f}  "
                    f"tiempo={dt:.4f}s  {resultado.particion}"
                )
            except Exception as e:
                dt = time.time() - t0
                print(f"  [{nombre:12s}] ERROR: {e}  (tras {dt:.1f}s)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sistemas = sys.argv[1:]
    else:
        sistemas = ["N10A"]

    for sistema in sistemas:
        for k in K_VALS:
            probar(sistema, k)
