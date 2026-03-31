
import itertools
import numpy as np

from src.models.sia import SIA


class Geometric(SIA):
    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ):
        # 1. Preparar subsistema (usa tu arquitectura existente)
        self.sia_preparar_subsistema(
            estado_inicial, condicion, alcance, mecanismo
        )

        # 2. Obtener estados del subsistema
        estados = self.generar_estados()

        # 3. Obtener distribución (tensor → vector)
        P = self.sia_dists_marginales

        # 4. Buscar mejor bipartición
        mejor_costo = float("inf")
        mejor_particion = None

        for A, B in self.generar_biparticiones(estados):
            costo = self.calcular_costo(A, B, estados, P)

            if costo < mejor_costo:
                mejor_costo = costo
                mejor_particion = (A, B)

        return {
            "costo": mejor_costo,
            "particion": mejor_particion,
        }

    # -------------------------
    # FUNCIONES AUXILIARES
    # -------------------------

    def generar_estados(self):
        """
        Genera estados binarios según dimensión del subsistema
        """
        n = len(self.sia_subsistema.dims_ncubos)
        return list(itertools.product([0, 1], repeat=n))

    def hamming(self, a, b):
        return sum(x != y for x, y in zip(a, b))

    def calcular_costo(self, A, B, estados, P):
        total = 0

        for i, a in enumerate(estados):
            for j, b in enumerate(estados):
                if a in A and b in B:
                    total += P[i] * P[j] * self.hamming(a, b)

        return total

    def generar_biparticiones(self, estados):
        n = len(estados)

        for r in range(1, n // 2 + 1):
            for subset in itertools.combinations(estados, r):
                A = set(subset)
                B = set(estados) - A
                yield A, B
