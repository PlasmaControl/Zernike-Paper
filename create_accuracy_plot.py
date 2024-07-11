import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import mpmath
import timeit
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("./ZERN/"))

from zern.zern_core import Zernike
import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("./zernike/"))
from zernike import RZern

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("./zernpy/src/"))
from zernpy import ZernPol

from zernipax.basis import ZernikePolynomial
from zernipax.zernike import *
from zernipax.plotting import plot_comparison
from zernipax.backend import jax


def fun_zernipax_clear(ns, ms, r):
    jax.clear_caches()
    return zernike_radial(r, ns, ms, 0)


def fun_zernipax(ns, ms, r):
    return zernike_radial(r, ns, ms, 0)


def fun_zern(ns, ms, r):
    zern = Zernike(0)
    all = []
    for i in range(len(ms)):
        all.append(zern.R_nm_Jacobi(int(ns[i]), int(ms[i]), r))

    return np.array(all)


def get_Noll(n, m):
    j = n * (n + 1) // 2 + abs(m)
    if m >= 0 and (n % 4 == 2 or n % 4 == 3):
        j += 1
    elif m <= 0 and (n % 4 == 0 or n % 4 == 1):
        j += 1
    return j


def fun_zernike(ns, ms, r):
    all = []
    cart = RZern(int(max(ns)))
    for i in range(len(ms)):
        id_Noll = get_Noll(ns[i], ms[i]) - 1
        all.append(cart.Rnm(id_Noll, r))

    return np.array(all)


def fun_zernpy(ns, ms, r):
    all = []
    for i in range(len(ms)):
        zp = ZernPol(m=int(ms[i]), n=int(ns[i]))
        all.append(zp.radial(r))

    return np.array(all)


mpmath.mp.dps = 100
def fun_exact(ns, ms, r):
    c = zernike_radial_coeffs(ns, ms, exact=True)
    zt0 = np.array([np.asarray(mpmath.polyval(list(ci), r), dtype=float) for ci in c]).T
    return zt0

# Accuracy
res = 50
basis = ZernikePolynomial(L=res, M=res, spectral_indexing="ansi", sym="cos")
ms = basis.modes[:, 1]
ns = basis.modes[:, 0]
r = np.linspace(0, 1, 100)

all_zernipax = fun_zernipax(ns, ms, r)
all_zern = fun_zern(ns, ms, r).T
all_zernike = fun_zernike(ns, ms, r).T
if res <= 50:
    all_zernpy = fun_zernpy(ns, ms, r).T
exact = fun_exact(ns, ms, r)

if res <= 50:
    plot_comparison(
        exact,
        (all_zernipax, all_zern, all_zernpy, all_zernike),
        basis,
        dx=0,
        type="absolute",
        names=("ZERNIPAX:", "ZERN:", "ZERNPY:", "ZERNIKE:"),
        print_error=True,
    )
else:
    plot_comparison(
        exact,
        (all_zernipax, all_zern, all_zernike),
        basis,
        dx=0,
        type="absolute",
        names=("ZERNIPAX:", "ZERN:", "ZERNIKE:"),
        print_error=True,
    )
plt.savefig("compare_all_accuracy.png", dpi=1000)
