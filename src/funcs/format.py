from funcs.iit import ABECEDARY


def _etiqueta(vertice: tuple[int, int]) -> str:
    tiempo, indice = vertice
    letra = ABECEDARY[indice]
    return letra.lower() if tiempo == 0 else letra


def fmt_biparticion_q(
    mip: list[tuple[int, int]],
    complemento: list[tuple[int, int]],
) -> str:
    """
    Formatea una bipartición al estilo Q-Nodes.
    mip: vertices del primer grupo.
    complemento: vertices del segundo grupo.
    """
    mip_str = ", ".join(sorted(_etiqueta(v) for v in mip))
    comp_str = ", ".join(sorted(_etiqueta(v) for v in complemento))
    return f"{{{mip_str}}} x {{{comp_str}}}"


def fmt_kparticion(partes: list[set[int]], k: int) -> str:
    """
    Formatea una k-partición de variables.
    partes: lista de conjuntos de índices de variables.
    """
    return " x ".join(
        ",".join(ABECEDARY[i] for i in sorted(parte)) for parte in partes
    )
