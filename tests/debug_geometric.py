# pega esto en un archivo tests/debug_geometric.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from funcs.cargar import csv_to_tpm
from controllers.strategies.geometric import Geometric
from controllers.strategies.qnodes import QNodes

tpm = csv_to_tpm("N10A")
n = tpm.shape[1]
estado = "1" + "0"*(n-1)
cond = "1"*n
alcance  = "1111111110"   # ABCDEFGHI
mecanismo = "1111111110"  # ABCDEFGHI

rq = QNodes(tpm).aplicar_estrategia(estado, cond, alcance, mecanismo)
rg = Geometric(tpm).aplicar_estrategia(estado, cond, alcance, mecanismo)

print(f"QNodes:    pérdida={rq.perdida:.6f}  partición={rq.particion}")
print(f"Geometric: pérdida={rg.perdida:.6f}  partición={rg.particion}")
print(f"¿Candidatas generadas? Revisa _generar_candidatas_2partes")