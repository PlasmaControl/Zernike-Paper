import sys
import os

from zernipax import set_device

set_device("gpu")

import numpy as np
import matplotlib.pyplot as plt
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


def fun_zernipax_cpu(ns, ms, r):
    with jax.default_device(jax.devices("cpu")[0]):
        out = zernike_radial(r, ns, ms, 0)
    return out


def fun_zernipax(ns, ms, r):
    return zernike_radial_old_desc(r[:, np.newaxis], ns, ms, 0)


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


# Timing
r = np.linspace(0, 1, 100)
times = []
num_exec = 100
range_res = np.arange(10, 101, 2)

for res in tqdm(range_res):
    basis = ZernikePolynomial(L=res, M=res, spectral_indexing="ansi", sym="cos")
    ms = basis.modes[:, 1]
    ns = basis.modes[:, 0]
    _ = fun_zernipax(ns, ms, r)  # run to compile it once
    _ = fun_zernipax_cpu(ns, ms, r)  # run to compile it once

    t1 = timeit.timeit(lambda: fun_zern(ns, ms, r), number=num_exec)
    t2 = timeit.timeit(lambda: fun_zernike(ns, ms, r), number=num_exec)
    t3 = timeit.timeit(
        lambda: fun_zernipax(ns, ms, r).block_until_ready(), number=num_exec
    )
    t4 = timeit.timeit(lambda: fun_zernipax_cpu(ns, ms, r).block_until_ready(), number=num_exec)
    times.append([t1, t2, t3, t4])
times = np.array(times) * 1000 / num_exec


plt.figure()
plt.plot(range_res, times[:, 0], label="ZERN")
plt.plot(range_res, times[:, 1], label="ZERNIKE")
plt.plot(range_res, times[:, 2], label="ZERNIPAX GPU")
plt.plot(range_res, times[:, 3], label="ZERNIPAX CPU")
plt.xlabel("Resolution")
plt.ylabel("Time (ms)")
plt.title("Execution Times of Radial Zernike Polynomials")
plt.grid()
plt.legend()
plt.savefig("all_t_compare.png", dpi=1000)

plt.figure()
plt.semilogy(range_res, times[:, 0], label="ZERN")
plt.semilogy(range_res, times[:, 1], label="ZERNIKE")
plt.semilogy(range_res, times[:, 2], label="ZERNIPAX GPU")
plt.semilogy(range_res, times[:, 3], label="ZERNIPAX CPU")
plt.xlabel("Resolution")
plt.ylabel("Time (ms)")
plt.title("Execution Times of Radial Zernike Polynomials")
plt.grid()
plt.legend()
plt.savefig("all_t_compare_log.png", dpi=1000)

plt.figure()
plt.plot(range_res[:10], times[:10, 0], label="ZERN")
plt.plot(range_res[:10], times[:10, 1], label="ZERNIKE")
plt.plot(range_res[:10], times[:10, 2], label="ZERNIPAX GPU")
plt.plot(range_res[:10], times[:10, 3], label="ZERNIPAX CPU")
plt.xlabel("Resolution")
plt.ylabel("Time (ms)")
plt.xticks(range_res[:10])
plt.title("Execution Times of Radial Zernike Polynomials")
plt.grid()
plt.legend()
plt.savefig("all_t_compare_low_res.png", dpi=1000)
