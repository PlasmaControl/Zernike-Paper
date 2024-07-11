import numpy as np
import matplotlib.pyplot as plt
import mpmath

from zernipax.basis import ZernikePolynomial
from zernipax.zernike import *
from zernipax.backend import jax
from tqdm import tqdm


# Exact computation
def fun_exact_prec(ns, ms, r, prec):
    mpmath.mp.dps = prec
    c = zernike_radial_coeffs(ns, ms, exact=True)
    zt0 = np.array([np.asarray(mpmath.polyval(list(ci), r), dtype=float) for ci in c]).T
    return zt0


range_prec = np.arange(20, 81, 2)
res = 150

basis = ZernikePolynomial(L=res, M=res, spectral_indexing="ansi", sym="cos")
ms = basis.modes[:, 1]
ns = basis.modes[:, 0]
r = np.linspace(0, 1, 100)

diff = []
val_old = fun_exact_prec(ns, ms, r, 200)

for prec in tqdm(range_prec):
    val = fun_exact_prec(ns, ms, r, prec)
    diff.append(np.max(val - val_old))
diff = np.array(diff)

plt.plot(range_prec, diff)
