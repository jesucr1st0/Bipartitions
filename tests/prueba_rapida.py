import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from funcs.cargar import csv_to_tpm
from controllers.strategies.kgeometric import KGeometric
from controllers.strategies.kqnodes import KQNodes
from controllers.strategies.geometric import Geometric
from controllers.strategies.qnodes import QNodes


def main():
    if len(sys.argv) < 3:
        print("Uso: python tests/prueba_rapida.py <sistema> <k> [estado_inicial]")
        print("Ej:  python tests/prueba_rapida.py N3A 2")
        print("     python tests/prueba_rapida.py N10A 3 1000000000")
        sys.exit(1)

    sistema = sys.argv[1]
    k = int(sys.argv[2])
    tpm = csv_to_tpm(sistema)
    n = tpm.shape[1]

    estado = sys.argv[3] if len(sys.argv) > 3 else "0" * n
    estado = estado.zfill(n)
    cond = "1" * n

    print(f"Sistema: {sistema}  n={n}  k={k}  estado_inicial={estado}")
    print()

    if k == 2:
        for nombre, Cls in [("Geometric", Geometric), ("QNodes", QNodes)]:
            r = Cls(tpm).aplicar_estrategia(estado, cond, cond, cond)
            print(f"  {nombre:<10} pérdida={r.perdida:.4f}  t={r.tiempo_total:.3f}s  {r.particion}")
        print()

    for nombre, Cls in [("KGeom", KGeometric), ("KQNodes", KQNodes)]:
        r = Cls(tpm).find_k_mip(k, estado, cond, cond, cond)
        print(f"  {nombre:<10} pérdida={r.perdida:.4f}  t={r.tiempo_total:.3f}s  {r.particion}")


if __name__ == "__main__":
    main()
