print("INICIO TEST")

import sys
import os

sys.path.append(os.path.abspath("src"))

import numpy as np
from controllers.strategies.geometric import Geometric

print("Import OK")

n = 3
tpm = np.random.rand(2**n, n)  # (8, 3)

geo = Geometric(tpm)

print("Objeto creado")

resultado = geo.aplicar_estrategia(
    estado_inicial="000",
    condicion="111",
    alcance="111",
    mecanismo="111",
)

print("RESULTADO:")
print(resultado)
