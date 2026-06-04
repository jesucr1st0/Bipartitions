from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Solution:
    estrategia: str
    perdida: float
    distribucion_subsistema: NDArray[np.float32]
    distribucion_particion: NDArray[np.float32]
    tiempo_total: float
    particion: str
    k: int = 2

    def __str__(self) -> str:
        return (
            f"Solution(estrategia={self.estrategia}, k={self.k}, "
            f"perdida={self.perdida:.6f}, "
            f"tiempo={self.tiempo_total:.4f}s, "
            f"particion={self.particion})"
        )
