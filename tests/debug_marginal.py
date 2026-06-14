# tests/debug_confirmar.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from funcs.cargar import csv_to_tpm
from controllers.strategies.geometric import Geometric
from funcs.iit import emd_efecto

tpm = csv_to_tpm("N10A")
n   = tpm.shape[1]
estado    = "1" + "0" * (n - 1)
cond      = "1" * n
alcance   = "1" * (n - 1) + "0"
mecanismo = "1" * (n - 1) + "0"

geo = Geometric(tpm)
geo.sia_preparar_subsistema(estado, cond, alcance, mecanismo)
S  = geo.sia_subsistema
P0 = geo.sia_dists_marginales

# Probar todas las variables una por una con mec=[]
print("bipartir(alc=[v], mec=[]) para cada v:")
for v in range(9):
    sp = S.bipartir(np.array([v], dtype=np.int8), np.array([], dtype=np.int8))
    marginal = sp.distribucion_marginal()
    perdida = emd_efecto(marginal, P0)
    marca = " ← ÓPTIMO" if abs(perdida - 0.019531) < 0.001 else ""
    print(f"  v={v}  EMD={perdida:.6f}{marca}")