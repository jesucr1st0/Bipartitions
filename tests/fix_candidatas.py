"""
fix_candidatas.py
=================
Coloca este archivo en tests/ y córrelo para verificar la corrección.

El problema: _generar_candidatas_2partes solo genera n+1 candidatas
(una por variable + KMeans), cubriendo solo ~4% del espacio de búsqueda.

La solución: usar la tabla T para generar candidatas más inteligentes
que cubran mejor el espacio, incluyendo todas las particiones por
variable individual Y agrupaciones basadas en similitud de perfiles.

Copia el método corregido a geometric.py cuando lo verifiques.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from funcs.cargar import csv_to_tpm
from controllers.strategies.geometric import Geometric
from controllers.strategies.qnodes import QNodes


# ─── Método corregido ────────────────────────────────────────────────────────

def _generar_candidatas_2partes_corregido(self, T, n_dims):
    """
    Genera candidatas a bipartición usando múltiples estrategias:

    1. Una partición por cada variable individual {v} vs resto  → n candidatas
    2. KMeans con k=2 sobre perfiles de T                       → 1 candidata
    3. Agrupaciones por similitud de columnas de T              → varias
    4. Particiones por umbral sobre suma de costos              → varias
    5. Todas las particiones de tamaño 2 vs resto               → C(n,2) candidatas

    Para n=10 esto da ~60 candidatas en lugar de 10,
    cubriendo mucho más del espacio sin ser exhaustivo.
    """
    from sklearn.cluster import KMeans

    n_vars  = T.shape[0]
    n_states = T.shape[1]
    todos   = set(range(n_dims))
    candidatas = set()

    # ── 1. Una variable vs el resto (igual que antes) ──────────────────────
    for v in range(n_vars):
        candidatas.add((frozenset({v}), frozenset(todos - {v})))

    # ── 2. KMeans k=2 sobre perfiles de T ─────────────────────────────────
    profiles = T.reshape(n_vars, n_states * n_states)
    try:
        for seed in range(5):
            labels = KMeans(n_clusters=2, n_init=5, random_state=seed).fit_predict(profiles)
            parte_a = frozenset(int(i) for i in range(n_vars) if labels[i] == 0)
            parte_b = frozenset(int(i) for i in range(n_vars) if labels[i] == 1)
            if parte_a and parte_b:
                candidatas.add((parte_a, parte_b))
    except Exception:
        pass

    # ── 3. Agrupación por suma de costos por variable ──────────────────────
    # Cada variable tiene un "peso total" = suma de todos sus costos de transición
    # Ordenamos por peso y probamos distintos puntos de corte
    pesos = T.sum(axis=(1, 2))           # shape (n_vars,)
    orden = np.argsort(pesos)            # variables de menor a mayor peso
    for corte in range(1, n_vars):
        ligeras = frozenset(int(orden[i]) for i in range(corte))
        pesadas  = frozenset(todos - ligeras)
        if ligeras and pesadas:
            candidatas.add((ligeras, pesadas))

    # ── 4. Particiones de tamaño 2 vs resto ────────────────────────────────
    # Para n<=15 esto agrega C(n,2) candidatas más (~45 para n=10)
    if n_vars <= 15:
        from itertools import combinations
        for par in combinations(range(n_vars), 2):
            parte_a = frozenset(par)
            parte_b = frozenset(todos - parte_a)
            if parte_b:
                candidatas.add((parte_a, parte_b))

    return list(candidatas)


# ─── Subclase con la corrección ───────────────────────────────────────────────

class GeometricMejorado(Geometric):
    def _generar_candidatas_2partes(self, T, n_dims):
        return _generar_candidatas_2partes_corregido(self, T, n_dims)


# ─── Verificación ─────────────────────────────────────────────────────────────

def verificar(sistema="N10A"):
    tpm = csv_to_tpm(sistema)
    n   = tpm.shape[1]
    estado   = "1" + "0" * (n - 1)
    cond     = "1" * n
    alcance  = "1" * (n - 1) + "0"   # ABCDEFGHI (sin J)
    mecanismo = "1" * (n - 1) + "0"

    print(f"Sistema: {sistema}  n={n}")
    print()

    rq = QNodes(tpm).aplicar_estrategia(estado, cond, alcance, mecanismo)
    print(f"QNodes (referencia):      pérdida={rq.perdida:.6f}  {rq.particion}")

    rg_bug = Geometric(tpm).aplicar_estrategia(estado, cond, alcance, mecanismo)
    print(f"Geometric (original):     pérdida={rg_bug.perdida:.6f}  {rg_bug.particion}")

    rg_fix = GeometricMejorado(tpm).aplicar_estrategia(estado, cond, alcance, mecanismo)
    print(f"Geometric (corregido):    pérdida={rg_fix.perdida:.6f}  {rg_fix.particion}")

    print()
    dif_bug = rg_bug.perdida - rq.perdida
    dif_fix = rg_fix.perdida - rq.perdida
    print(f"Diferencia original vs QNodes:  {dif_bug:+.6f}")
    print(f"Diferencia corregido vs QNodes: {dif_fix:+.6f}")

    if abs(dif_fix) < abs(dif_bug):
        print("✅ La corrección mejora los resultados.")
    else:
        print("⚠️  La corrección no mejoró en este caso.")

    print()
    print("─" * 50)
    print("INSTRUCCIONES PARA APLICAR LA CORRECCIÓN:")
    print("─" * 50)
    print("""
En geometric.py, reemplaza _generar_candidatas_2partes con:

    def _generar_candidatas_2partes(self, T, n_dims):
        from sklearn.cluster import KMeans
        from itertools import combinations

        n_vars   = T.shape[0]
        n_states = T.shape[1]
        todos    = set(range(n_dims))
        candidatas = set()

        # 1. Una variable vs el resto
        for v in range(n_vars):
            candidatas.add((frozenset({v}), frozenset(todos - {v})))

        # 2. KMeans con múltiples semillas
        profiles = T.reshape(n_vars, n_states * n_states)
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

        # 3. Cortes por peso total de transiciones
        pesos = T.sum(axis=(1, 2))
        orden = np.argsort(pesos)
        for corte in range(1, n_vars):
            a = frozenset(int(orden[i]) for i in range(corte))
            b = frozenset(todos - a)
            if a and b:
                candidatas.add((a, b))

        # 4. Pares de variables vs el resto (solo si n <= 15)
        if n_vars <= 15:
            for par in combinations(range(n_vars), 2):
                a = frozenset(par)
                b = frozenset(todos - a)
                if b:
                    candidatas.add((a, b))

        return list(candidatas)
""")


if __name__ == "__main__":
    sistema = sys.argv[1] if len(sys.argv) > 1 else "N10A"
    verificar(sistema)
