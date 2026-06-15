# tests/debug_entre_casos.py
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

t0 = time.time()
from funcs.cargar import csv_to_tpm, excel_a_configs
print(f"imports: {time.time()-t0:.2f}s")

t1 = time.time()
tpm = csv_to_tpm("N10A")
print(f"csv_to_tpm: {time.time()-t1:.2f}s")

t2 = time.time()
configs = excel_a_configs("data/DatosPruebas2026_1.xlsx", "10A-Elementos")
print(f"excel_a_configs: {time.time()-t2:.2f}s")

from controllers.strategies.geometric import Geometric

for i, cfg in enumerate(configs[:5]):
    print(f"\n--- Caso {i+1} ---")
    
    t3 = time.time()
    geo = Geometric(tpm)
    print(f"  Geometric.__init__: {time.time()-t3:.2f}s")
    
    t4 = time.time()
    geo.sia_preparar_subsistema(
        cfg["estado_inicial"],
        "1" * len(cfg["estado_inicial"]),
        cfg["alcance"],
        cfg["mecanismo"],
    )
    print(f"  sia_preparar_subsistema: {time.time()-t4:.2f}s")