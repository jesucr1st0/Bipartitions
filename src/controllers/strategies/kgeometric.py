from sklearn.cluster import KMeans

from controllers.strategies.geometric import Geometric


class KGeometric(Geometric):
    def __init__(self, tpm):
        super().__init__(tpm)

    def _generar_candidatas_kpartes(self, T, k, n_vars):
        candidatas = set()
        n_states = T.shape[1]
        profiles = T.reshape(n_vars, n_states * n_states)

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

        return list(candidatas)
