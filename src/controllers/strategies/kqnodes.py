import time
from itertools import combinations

import numpy as np

from constants.base import COLS_IDX
from constants.models import QNODES_LABEL
from funcs.iit import emd_efecto
from models.core.solution import Solution
from models.sia import SIA


class KQNodes(SIA):
    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)

    def aplicar_estrategia(self, estado_inicial, condicion, alcance, mecanismo):
        from controllers.strategies.qnodes import QNodes
        qn = QNodes(self.tpm)
        return qn.aplicar_estrategia(estado_inicial, condicion, alcance, mecanismo)

    def find_k_mip(self, k, estado_inicial, condicion, alcance, mecanismo):
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        if k == 2:
            return self.aplicar_estrategia(
                estado_inicial, condicion, alcance, mecanismo
            )

        S = self.sia_subsistema
        P0 = self.sia_dists_marginales
        all_indices = set(S.indices_ncubos)

        groups = [all_indices.copy()]
        ultimo_loss = np.inf
        ultimo_groups = None
        ultimo_dist = None

        while len(groups) < k:
            largest_idx = max(
                range(len(groups)), key=lambda i: len(groups[i])
            )
            largest = groups[largest_idx]

            if len(largest) <= 1:
                break

            best_loss = np.inf
            best_split = None

            vars_list = sorted(largest)
            max_exhaustive = min(8, len(vars_list))
            if len(vars_list) <= max_exhaustive:
                for r in range(1, len(vars_list)):
                    for subset in combinations(vars_list, r):
                        part_a = set(subset)
                        part_b = largest - part_a
                        candidate = (
                            groups[:largest_idx]
                            + groups[largest_idx + 1 :]
                            + [part_a, part_b]
                        )
                        p, _ = self._evaluar_partes(S, P0, candidate)
                        if p < best_loss:
                            best_loss = p
                            best_split = (part_a, part_b)
            else:
                from sklearn.cluster import KMeans

                X = [
                    ncubo.data.flatten()
                    for ncubo in S.ncubos
                    if ncubo.indice in largest
                ]
                if X:
                    profiles = np.array(X)
                    labels = KMeans(
                        n_clusters=2, n_init=5, random_state=0
                    ).fit_predict(profiles)
                    part_a = set(
                        int(v)
                        for i, v in enumerate(vars_list)
                        if labels[i] == 0
                    )
                    part_b = largest - part_a
                    if part_a and part_b:
                        candidate = (
                            groups[:largest_idx]
                            + groups[largest_idx + 1 :]
                            + [part_a, part_b]
                        )
                        p, _ = self._evaluar_partes(S, P0, candidate)
                        best_loss = p
                        best_split = (part_a, part_b)

            if best_split is None:
                break

            groups.pop(largest_idx)
            groups.extend(best_split)

            loss, dist = self._evaluar_partes(S, P0, groups)
            ultimo_loss = loss
            ultimo_groups = list(groups)
            ultimo_dist = dist

        grupos_base = ultimo_groups or groups
        grupos_base = self._refinar_partes(S, P0, grupos_base)

        fmt = self._fmt_partes(grupos_base)
        return Solution(
            estrategia=QNODES_LABEL,
            perdida=self._evaluar_partes(
                S, P0, grupos_base
            )[0],
            distribucion_subsistema=P0,
            distribucion_particion=ultimo_dist if ultimo_dist is not None else P0,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt,
            k=k,
        )

    def _evaluar_partes(
        self, S, P0, groups
    ):
        indices = set(S.indices_ncubos)
        pos = {int(idx): i for i, idx in enumerate(S.indices_ncubos)}
        reconst = np.empty(len(P0), dtype=np.float32)
        for g in groups:
            alc = np.array(sorted(indices - g), dtype=np.int8)
            sp = S.substraer(alc, alc)
            marginal = sp.distribucion_marginal()
            for idx, val in zip(sp.indices_ncubos, marginal):
                reconst[pos[int(idx)]] = val
        return emd_efecto(P0, reconst), reconst

    def _refinar_partes(self, S, P0, groups, max_iter=10):
        groups = [set(g) for g in groups]
        k = len(groups)
        indices = set(S.indices_ncubos)
        for _ in range(max_iter):
            mejorado = False
            for v in indices:
                idx_actual = next(
                    (i for i, p in enumerate(groups) if v in p), None
                )
                if idx_actual is None or len(groups[idx_actual]) <= 1:
                    continue
                best_p, _ = self._evaluar_partes(S, P0, groups)
                best_idx = None
                for idx_nuevo in range(k):
                    if idx_nuevo == idx_actual:
                        continue
                    groups[idx_actual].discard(v)
                    groups[idx_nuevo].add(v)
                    p, _ = self._evaluar_partes(S, P0, groups)
                    if p < best_p:
                        best_p = p
                        best_idx = idx_nuevo
                    groups[idx_nuevo].discard(v)
                    groups[idx_actual].add(v)
                if best_idx is not None:
                    groups[idx_actual].discard(v)
                    groups[best_idx].add(v)
                    mejorado = True
            if not mejorado:
                break
        return groups

    def _fmt_partes(self, partes):
        from funcs.format import fmt_kparticion
        return fmt_kparticion([set(p) for p in partes], len(partes))
