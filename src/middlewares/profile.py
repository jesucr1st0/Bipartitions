import time
from functools import wraps
from typing import Any, Callable

from constants.base import PATH_PROFILING


class GestorPerfilado:
    def __init__(self):
        self.sesion_activa = False
        self.etiqueta_sesion = ""
        self.inicio_sesion = 0.0

    def start_session(self, etiqueta: str):
        self.sesion_activa = True
        self.etiqueta_sesion = etiqueta
        self.inicio_sesion = time.time()

    def end_session(self):
        self.sesion_activa = False

    def log_profile(self, funcion: str, tiempo: float, contexto: dict | None = None):
        pass


gestor_perfilado = GestorPerfilado()


def profile(func: Callable | None = None, *, context: dict | None = None) -> Callable:
    if func is None:
        return lambda f: profile(f, context=context)

    @wraps(func)
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        duracion = time.time() - inicio
        gestor_perfilado.log_profile(func.__name__, duracion, context)
        return resultado

    return wrapper
