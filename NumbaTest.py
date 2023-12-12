import numba
from functools import wraps
from time import time
import numpy as np

def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        print(f'fun: {f.__name__}, args: [{args}, {kwargs}] took: {te-ts} sec')
        return result
    return wrap


@numba.jit(nopython=True)
@timing
def expmean_jit(rea):
    val = rea.mean() ** 2
    return val

# Exemple d'utilisation avec une variable définie
rea = np.array([1, 2, 3, 4, 5])
result = expmean_jit(rea)
print("Résultat de expmean_jit :", result)
