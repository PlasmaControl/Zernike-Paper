import numpy as np
import matplotlib.pyplot as plt
import mpmath

from zernipax.basis import ZernikePolynomial
from zernipax.zernike import *
from tqdm import tqdm


# Exact computation
def fun_exact_prec(ns, ms, r, prec):
    mpmath.mp.dps = prec
    c = zernike_radial_coeffs(ns, ms, exact=True)
    zt0 = np.array([np.asarray(mpmath.polyval(list(ci), r), dtype=float) for ci in c]).T
    return zt0


range_prec = np.arange(8, 101, 2)
res = 100

basis = ZernikePolynomial(L=res, M=res, spectral_indexing="ansi", sym="cos")
ms = basis.modes[:, 1]
ns = basis.modes[:, 0]
r = np.linspace(0, 1, 100)

diff = []
exact_prec = 200
exact = fun_exact_prec(ns, ms, r, exact_prec)

for prec in tqdm(range_prec):
    val = fun_exact_prec(ns, ms, r, prec)
    diff.append(np.max(val - exact))

diff = np.array(diff)
results = np.vstack((range_prec, diff)).T

np.savetxt("results_precision.txt", results)
