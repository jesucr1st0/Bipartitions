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
        self._T: NDArray[np.float32] | None = None

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
        # Recalcular pérdida final tras refinamiento
        mejor_perdida, mejor_dist = self._evaluar_kparticion(S, P0, mejores_partes)

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
        return [
            ncubo.data.flatten().copy().astype(np.float32)
            for ncubo in S.ncubos
        ]

    def _calcular_tabla_costos(
        self, X: list[np.ndarray]
    ) -> NDArray[np.float32]:
        """
        Calcula la tabla de costos T usando la fórmula del documento (ec. 3.1):

            t(i,j) = gamma * (|X[i] - X[j]| + sum_{k in N(i,j)} t(k,j))

        donde gamma = 2^(-d_H(i,j))  —  DECRECE con la distancia Hamming.

        Implementación vectorizada con numpy (float32) para mayor velocidad.
        Para n<=12 usa matrices densas. Para n>12 cae al BFS original.
        """
        n_vars   = len(X)
        n_states = len(X[0])
        n_bits   = int(np.log2(n_states))

        # Para sistemas grandes la matriz densa no cabe en memoria
        if n_bits > 12:
            return self._calcular_tabla_costos_bfs(X)

        # ── Precalcular distancias Hamming ──────────────────────────────
        states   = np.arange(n_states, dtype=np.int32)
        popcount = np.array([bin(x).count('1') for x in range(n_states)],
                            dtype=np.int8)
        xor_m    = states[:, None] ^ states[None, :]   # (n_states, n_states)
        H        = popcount[xor_m]                     # distancias Hamming

        # Matriz de adyacencia directa (distancia 1)
        adj1 = (H == 1).astype(np.float32)             # (n_states, n_states)

        T = np.zeros((n_vars, n_states, n_states), dtype=np.float32)

        for v in range(n_vars):
            xv   = X[v].astype(np.float32)
            diff = np.abs(xv[:, None] - xv[None, :])  # |X[i]-X[j]|
            Tv   = np.zeros((n_states, n_states), dtype=np.float32)

            for d in range(1, n_bits + 1):
                gamma  = np.float32(2.0 ** (-d))
                mask_d = (H == d)
                # accum[i,j] = sum_{k: H[k,j]==d-1, H[i,k]==1} T[k,j]
                # = (adj1 @ Tv)[i,j]  cuando Tv ya tiene niveles 1..d-1
                accum = (adj1 @ Tv) if d > 1 else np.zeros_like(Tv)
                Tv    = np.where(mask_d, gamma * (diff + accum), Tv)

            T[v] = Tv

        return T

    def _calcular_tabla_costos_bfs(
        self, X: list[np.ndarray]
    ) -> NDArray[np.float32]:
        """
        BFS original corregido (gamma = 2^-dist) para sistemas con n > 12
        donde la matriz densa no cabe en memoria.
        """
        n_vars   = len(X)
        n_states = len(X[0])
        n_bits   = int(np.log2(n_states))
        T = np.zeros((n_vars, n_states, n_states), dtype=np.float32)

        for v in range(n_vars):
            xv = X[v]
            for j in range(n_states):
                visited  = {j}
                frontier = {j}
                dist     = 0
                while frontier:
                    dist  += 1
                    gamma  = np.float32(2.0 ** (-dist))
                    _next  = set()
                    for i in frontier:
                        mask = 1
                        for _ in range(n_bits):
                            nb = i ^ mask
                            if nb not in visited:
                                _next.add(nb)
                            mask <<= 1
                    for nb in _next:
                        direct = abs(float(xv[nb]) - float(xv[j]))
                        accum  = 0.0
                        mask   = 1
                        for _ in range(n_bits):
                            prev = nb ^ mask
                            if prev in visited:
                                accum += float(T[v, prev, j])
                            mask <<= 1
                        T[v, nb, j] = gamma * (direct + accum)
                    visited.update(_next)
                    frontier = _next
        return T

    def _generar_candidatas_2partes(
        self, T: NDArray[np.float32], n_dims: int
    ) -> list[tuple[frozenset, frozenset]]:
        n_vars   = T.shape[0]
        n_states = T.shape[1]
        todos    = set(range(n_dims))
        candidatas = set()

        # 1. Una variable vs el resto
        for v in range(n_vars):
            candidatas.add((frozenset({v}), frozenset(todos - {v})))

        # 2. KMeans con reducción de dimensión
        profiles = T.reshape(n_vars, n_states * n_states)
        if profiles.shape[1] > 100:
            from sklearn.decomposition import TruncatedSVD
            n_comp = min(20, n_vars - 1)
            profiles = TruncatedSVD(n_components=n_comp,
                                    random_state=0).fit_transform(profiles)
        for seed in range(5):
            try:
                labels = KMeans(n_clusters=2, n_init=5,
                                random_state=seed).fit_predict(profiles)
                a = frozenset(int(i) for i in range(n_vars) if labels[i] == 0)
                b = frozenset(todos - a)
                if a and b:
                    candidatas.add((a, b))
            except Exception:
                pass

        # 3. Cortes por peso — solo los 5 mejores
        pesos = T.sum(axis=(1, 2))
        orden = np.argsort(pesos)
        for corte in range(1, min(6, n_vars)):   # ← era range(1, n_vars)
            a = frozenset(int(orden[i]) for i in range(corte))
            b = frozenset(todos - a)
            if a and b:
                candidatas.add((a, b))

        # 4. Pares de variables vs el resto (solo si n <= 15)
        # 4. Pares de variables vs el resto
        if n_vars <= 8:
            # Para n pequeño: todos los pares
            for par in combinations(range(n_vars), 2):
                a = frozenset(par)
                b = frozenset(todos - a)
                if b:
                    candidatas.add((a, b))
        else:
            # Para n grande: solo los 10 mejores pares según peso de T
            pesos = T.sum(axis=(1, 2))
            top = np.argsort(pesos)[:6].tolist()
            for par in combinations(top, 2):
                a = frozenset(par)
                b = frozenset(todos - a)
                if b:
                    candidatas.add((a, b))

        return list(candidatas)

    def _generar_candidatas_kpartes(
        self, T: NDArray[np.float32], k: int, n_vars: int
    ) -> list[tuple[frozenset, ...]]:
        candidatas = set()
        n_states   = T.shape[1]
        profiles   = T.reshape(n_vars, n_states * n_states)

        for seed in range(5):
            try:
                labels = KMeans(n_clusters=k, n_init=5,
                                random_state=seed).fit_predict(profiles)
                partes = tuple(
                    frozenset(int(i) for i in range(n_vars) if labels[i] == c)
                    for c in range(k)
                )
                if all(p for p in partes):
                    candidatas.add(partes)
            except Exception:
                pass

        # Jerarquía por peso
        pesos = T.sum(axis=(1, 2))
        orden = list(np.argsort(pesos))
        chunk = max(1, n_vars // k)
        partes_h = []
        for i in range(k - 1):
            partes_h.append(frozenset(orden[i * chunk:(i + 1) * chunk]))
        partes_h.append(frozenset(orden[(k - 1) * chunk:]))
        if all(p for p in partes_h):
            candidatas.add(tuple(partes_h))

        return list(candidatas)

    def _evaluar_kparticion(
        self,
        S: System,
        P0: NDArray[np.float32],
        partes: tuple[frozenset, ...],
    ) -> tuple[float, NDArray[np.float32]]:
        """
        Evalúa una k-partición usando bipartir() igual que QNodes:
        para cada parte, corta solo el alcance (mecanismo vacío)
        y toma la distribución marginal con menor EMD.
        """
        mejor_perdida = np.inf
        mejor_dist    = None

        for parte in partes:
            alc = np.array(sorted(parte), dtype=np.int8)
            mec = np.array([], dtype=np.int8)
            sp      = S.bipartir(alc, mec)
            marginal = sp.distribucion_marginal()
            perdida  = emd_efecto(marginal, P0)
            if perdida < mejor_perdida:
                mejor_perdida = perdida
                mejor_dist    = marginal

        return float(mejor_perdida), mejor_dist

    def _refinar_kparticion(
        self,
        S: System,
        P0: NDArray[np.float32],
        partes: tuple[frozenset, ...],
        k: int,
        max_iter: int = 10,
    ) -> tuple[frozenset, ...]:
        partes  = [set(p) for p in partes]
        indices = set(S.indices_ncubos)
        for _ in range(max_iter):
            mejorado = False
            for v in indices:
                idx_actual = next(
                    (i for i, p in enumerate(partes) if v in p), None
                )
                if idx_actual is None or len(partes[idx_actual]) <= 1:
                    continue
                mejor_perdida, _ = self._evaluar_kparticion(
                    S, P0, tuple(frozenset(p) for p in partes)
                )
                mejor_idx = None
                for idx_nuevo in range(k):
                    if idx_nuevo == idx_actual:
                        continue
                    partes[idx_actual].discard(v)
                    partes[idx_nuevo].add(v)
                    p, _ = self._evaluar_kparticion(
                        S, P0, tuple(frozenset(p) for p in partes)
                    )
                    if p < mejor_perdida:
                        mejor_perdida = p
                        mejor_idx     = idx_nuevo
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