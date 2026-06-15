from sklearn.cluster import KMeans

from controllers.strategies.geometric import Geometric


class KGeometric(Geometric):
    def __init__(self, tpm):
        super().__init__(tpm)

    def _generar_candidatas_kpartes(self, profiles, pesos, k, n_vars):
        candidatas = set()

        n_clusters = min(n_vars, max(k, 3))
        if n_clusters >= k:
            for seed in range(5):
                labels = KMeans(
                    n_clusters=k, n_init=1, random_state=seed
                ).fit_predict(profiles)
                partes = tuple(
                    frozenset(int(i) for i in range(n_vars) if labels[i] == c)
                    for c in range(k)
                )
                if all(p for p in partes):
                    candidatas.add(partes)

        from scipy.cluster.hierarchy import linkage, fcluster
        if n_vars >= 2:
            Z = linkage(profiles, method="ward")
            labels = fcluster(Z, k, criterion="maxclust")
            partes = tuple(
                frozenset(int(i) for i in range(n_vars) if labels[i] == c)
                for c in range(1, k + 1)
            )
            if all(p for p in partes):
                candidatas.add(partes)

        remaining = list(range(n_vars))
        partes_h = []
        for _ in range(k - 1):
            idx = remaining.pop(0)
            partes_h.append(frozenset({idx}))
        partes_h.append(frozenset(remaining))
        if all(p for p in partes_h):
            candidatas.add(tuple(partes_h))

        remaining = set(range(n_vars))
        hier_parts = []
        for _ in range(k - 1):
            if len(remaining) <= 1:
                break
            sub = sorted(remaining)
            sub_p = profiles[sub]
            lbl = KMeans(n_clusters=2, n_init=1, random_state=0).fit_predict(sub_p)
            a = frozenset(sub[i] for i in range(len(sub)) if lbl[i] == 0)
            b = remaining - a
            if not a or not b:
                a = frozenset({sub[0]})
                b = remaining - a
            hier_parts.append(a)
            remaining = b
        hier_parts.append(frozenset(remaining))
        if all(p for p in hier_parts) and len(hier_parts) == k:
            candidatas.add(tuple(hier_parts))

        return list(candidatas)
