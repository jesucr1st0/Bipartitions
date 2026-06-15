import os
import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "src", ".samples")


def generar_tpm(n: int, nombre: str, semilla: int = 42, chunk_size: int = 10000):
    rng = np.random.default_rng(semilla)
    total_rows = 2 ** n
    path = os.path.join(SAMPLES_DIR, f"{nombre}.csv")

    print(f"Generando {nombre}.csv: n={n}, {total_rows} filas, {n} columnas")
    print(f"  Escribiendo en chunks de {chunk_size} filas...")

    with open(path, "w") as f:
        written = 0
        while written < total_rows:
            this_chunk = min(chunk_size, total_rows - written)
            data = rng.uniform(0, 1, size=(this_chunk, n))
            for row in data:
                f.write(",".join(f"{v:.6f}" for v in row) + "\n")
            written += this_chunk
            pct = written / total_rows * 100
            print(f"  {written:>10} / {total_rows} ({pct:.1f}%)")

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Completado: {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) >= 2:
        n = int(args[0])
        nombre = args[1]
        semilla = int(args[2]) if len(args) > 2 else 42
        generar_tpm(n, nombre, semilla)
    else:
        print("Uso: python generar_tpm.py <n> <nombre> [semilla]")
        print("  Genera un TPM aleatorio de n variables en src/.samples/<nombre>.csv")
        print()
        print("  Ejemplos:")
        print("    python generar_tpm.py 20 N20A")
        print("    python generar_tpm.py 22 N22A")
        print("    python generar_tpm.py 25 N25A")
