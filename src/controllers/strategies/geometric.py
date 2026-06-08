import time
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from sklearn.cluster import KMeans

from funcs.iit import emd_efecto
from models.core.solution import Solution
from models.core.system import System
from models.sia import SIA


class Geometric(SIA):
    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        self._T: NDArray[np.float64] | None = None

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> Solution:
        return self.find_k_mip(2, estado_inicial, condicion, alcance, mecanismo)

    def find_k_mip(
        self,
        k: int,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> Solution:
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)
        S = self.sia_subsistema
        P0 = self.sia_dists_marginales
        n_vars = len(S.indices_ncubos)
        n_dims = len(S.dims_ncubos)

        if n_vars < k:
            raise ValueError(
                f"No se puede particionar {n_vars} variables en {k} partes"
            )

        X = self._extraer_valores(S)
        T = self._calcular_tabla_costos(X)
        self._T = T

        if k == 2:
            candidatas = self._generar_candidatas_2partes(T, n_dims)
        else:
            candidatas = self._generar_candidatas_kpartes(T, k, n_vars)

        mejor_perdida = np.inf
        mejores_partes = None
        mejor_dist = None

        for partes in candidatas:
            perdida, dist = self._evaluar_kparticion(S, P0, partes)
            if perdida < mejor_perdida:
                mejor_perdida = perdida
                mejores_partes = partes
                mejor_dist = dist

        mejores_partes = self._refinar_kparticion(S, P0, mejores_partes, k)

        fmt = self._fmt_partes(mejores_partes)
        return Solution(
            estrategia="Geometric",
            perdida=mejor_perdida,
            distribucion_subsistema=P0,
            distribucion_particion=mejor_dist,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt,
            k=k,
        )

    def _extraer_valores(self, S: System) -> list[np.ndarray]:
        n_states = 1 << len(S.dims_ncubos)
        return [
            ncubo.data.flatten().copy().astype(np.float64)
            for ncubo in S.ncubos
        ]

    def _calcular_tabla_costos(
        self, X: list[np.ndarray]
    ) -> NDArray[np.float64]:
        n_vars = len(X)
        n_states = len(X[0])
        n_bits = int(np.log2(n_states))
        T = np.zeros((n_vars, n_states, n_states), dtype=np.float64)

        for v in range(n_vars):
            xv = X[v]
            for j in range(n_states):
                T[v, j, j] = 0.0
                visited = {j}
                frontier = {j}
                dist = 0
                while frontier:
                    dist += 1
                    alpha = 1 << dist
                    _next = set()
                    for i in frontier:
                        mask = 1
                        for b in range(n_bits):
                            k = i ^ mask
                            if k not in visited:
                                _next.add(k)
                            mask <<= 1
                    for k in _next:
                        direct = alpha * abs(xv[k] - xv[j])
                        accum = 0.0
                        mask = 1
                        for b in range(n_bits):
                            nb = k ^ mask
                            if nb in visited:
                                accum += T[v, nb, j]
                            mask <<= 1
                        T[v, k, j] = direct + accum
                    visited.update(_next)
                    frontier = _next
        return T

    def _generar_candidatas_2partes(
        self, T: NDArray[np.float64], n_dims: int
    ) -> list[tuple[frozenset, frozenset]]:
        n_vars = T.shape[0]
        candidatas = set()

        for v in range(n_vars):
            candidatas.add((frozenset({v}), frozenset(set(range(n_dims)) - {v})))

        n_clusters = min(n_vars, n_dims)
        if n_clusters >= 2:
            n_states = T.shape[1]
            profiles = T.reshape(n_vars, n_states * n_states)
            labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(profiles)
            parte_a = frozenset(int(i) for i in range(n_vars) if labels[i] == 0)
            parte_b = frozenset(int(i) for i in range(n_vars) if labels[i] == 1)
            if parte_a and parte_b:
                candidatas.add((parte_a, parte_b))

        return list(candidatas)

    def _generar_candidatas_kpartes(
        self, T: NDArray[np.float64], k: int, n_vars: int
    ) -> list[tuple[frozenset, ...]]:
        candidatas = set()

        n_states = T.shape[1]
        profiles = T.reshape(n_vars, n_states * n_states)

        n_clusters = min(n_vars, max(k, 3))
        if n_clusters >= k:
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(profiles)
            partes = tuple(
                frozenset(int(i) for i in range(n_vars) if labels[i] == c)
                for c in range(k)
            )
            if all(p for p in partes):
                candidatas.add(partes)

        return list(candidatas)

    def _evaluar_kparticion(
        self,
        S: System,
        P0: NDArray[np.float32],
        partes: tuple[frozenset, ...],
    ) -> tuple[float, NDArray[np.float32]]:
        indices = set(S.indices_ncubos)
        reconst = np.empty(len(P0), dtype=np.float32)
        for parte in partes:
            alc = np.array(
                sorted(indices - parte), dtype=np.int8
            )
            sp = S.substraer(alc, alc)
            marginal = sp.distribucion_marginal()
            for idx, val in zip(sp.indices_ncubos, marginal):
                reconst[idx] = val
        perdida = emd_efecto(P0, reconst)
        return perdida, reconst

    def _refinar_kparticion(
        self,
        S: System,
        P0: NDArray[np.float32],
        partes: tuple[frozenset, ...],
        k: int,
        max_iter: int = 10,
    ) -> tuple[frozenset, ...]:
        partes = [set(p) for p in partes]
        indices = set(S.indices_ncubos)
        for _ in range(max_iter):
            mejorado = False
            for v in indices:
                idx_actual = next(
                    (i for i, p in enumerate(partes) if v in p), None
                )
                if idx_actual is None or len(partes[idx_actual]) <= 1:
                    continue
                mejor_perdida, _ = self._evaluar_kparticion(S, P0, tuple(partes))
                mejor_idx = None
                for idx_nuevo in range(k):
                    if idx_nuevo == idx_actual:
                        continue
                    partes[idx_actual].discard(v)
                    partes[idx_nuevo].add(v)
                    p, _ = self._evaluar_kparticion(S, P0, tuple(partes))
                    if p < mejor_perdida:
                        mejor_perdida = p
                        mejor_idx = idx_nuevo
                    partes[idx_nuevo].discard(v)
                    partes[idx_actual].add(v)
                if mejor_idx is not None:
                    partes[idx_actual].discard(v)
                    partes[mejor_idx].add(v)
                    mejorado = True
            if not mejorado:
                break
        return tuple(frozenset(p) for p in partes)

    def _fmt_partes(self, partes: tuple[frozenset, ...]) -> str:
        from funcs.format import fmt_kparticion
        return fmt_kparticion(
            [set(p) for p in partes], len(partes)
        )
