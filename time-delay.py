# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Time-delay example

# %% [markdown]
# ## Imports

# %%
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spla
import scipy.optimize as spopt
from pymor.models.iosys import LinearDelayModel
from pymor.operators.numpy import NumpyMatrixOperator
from scipy.special import lambertw

# %% [markdown]
# ## Full-order model


# %%
# fmt: off
A = np.array([
    [-0.1,   0,    0],
    [  0, -0.1,    1],
    [  0,   -1, -0.1],
])
Ad = np.array([
    [-0.1,    0,    0],
    [   0, -0.1,  0.1],
    [   0, -0.1, -0.1],
])
B = np.array([
    [1, 1, 1],
    [1, 2, 2],
    [1, 3, 1],
])
C = np.array([
    [1, 1, 1],
    [1, 2, 3],
])
# fmt: on
Aop = NumpyMatrixOperator(A)
Adop = NumpyMatrixOperator(Ad)
Bop = NumpyMatrixOperator(B)
Cop = NumpyMatrixOperator(C)
tau = 1
fom = LinearDelayModel(Aop, (Adop,), (tau,), Bop, Cop)

# %%
fom

# %%
mus = [-0.1, -0.1 + 1j, -0.1 - 1j]
sigmas = [-0.1, -0.1 + 0.1j, -0.1 - 0.1j]
N = 100_000
lam = lambda mu, sigma, k: mu + lambertw(tau * sigma * np.exp(-tau * mu), k=k) / tau
poles = np.array([lam(mu, sigma, k) for mu, sigma in zip(mus, sigmas) for k in range(-N, N + 1)])

# %%
poles.real.max()

# %%
fig, ax = plt.subplots()
ax.plot(poles.real, poles.imag, '.')
ax.set_yscale('symlog', linthresh=1)
ax.grid()

# %%
w = (1e-3, 1e3)
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = plt.legend()

# %% [markdown]
# ## Initial reduced-order model

# %%
Ar0 = A[:2, :2]
Adr0 = Ad[:2, :2]
Br0 = B[:2, :]
Cr0 = C[:, :2]
Ar0op = NumpyMatrixOperator(Ar0)
Adr0op = NumpyMatrixOperator(Adr0)
Br0op = NumpyMatrixOperator(Br0)
Cr0op = NumpyMatrixOperator(Cr0)
rom0 = LinearDelayModel(Ar0op, (Adr0op,), (tau,), Br0op, Cr0op)

# %%
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = rom0.transfer_function.mag_plot(w, label='Initial ROM')
_ = plt.legend()

# %%
_ = (fom - rom0).transfer_function.mag_plot(w, label='Initial ROM error')
_ = plt.legend()

# %%
fom.transfer_function.h2_norm(epsrel=1e-10, limit=1000)

# %%
(fom - rom0).transfer_function.h2_norm(epsrel=1e-10, limit=1000) / fom.transfer_function.h2_norm(
    epsrel=1e-10, limit=1000
)


# %% [markdown]
# ## Diagonal ROM optimization

# %% [markdown]
# ROM:
# $$
# \begin{align*}
#   \hat{A} =
#   \begin{bmatrix}
#     a_1 & 0 \\
#     0 & a_2
#   \end{bmatrix}, \quad
#   \hat{A}_\tau =
#   \begin{bmatrix}
#     t_1 & 0 \\
#     0 & t_2
#   \end{bmatrix}, \quad
#   \hat{B} \in \mathbb{R}^{2 \times 3}, \quad
#   \hat{C} \in \mathbb{R}^{2 \times 2}
# \end{align*}
# $$


# %%
def vec_to_rom(x):
    a, t, b, c = np.split(x, np.cumsum([2, 2, 6]))
    A = np.diag(a)
    Ad = np.diag(t)
    B = b.reshape(2, 3)
    C = c.reshape(2, 2)
    A = NumpyMatrixOperator(A)
    Ad = NumpyMatrixOperator(Ad)
    B = NumpyMatrixOperator(B)
    C = NumpyMatrixOperator(C)
    return LinearDelayModel(A, (Ad,), (tau,), B, C)


# %%
x0 = np.concatenate(
    (
        Ar0.diagonal(),
        Adr0.diagonal(),
        Br0.reshape(-1),
        Cr0.reshape(-1),
    )
)


# %%
def obj(x, epsrel=1e-10, limit=1000):
    rom = vec_to_rom(x)
    return (fom - rom).transfer_function.h2_norm(
        epsrel=epsrel,
        limit=limit,
    )


# %%
rel_error = lambda x, y: spla.norm(x - y) / spla.norm(x)

# %%
tic = perf_counter()
res = spopt.minimize(
    lambda x: obj(x, limit=10),
    x0,
    method='Nelder-Mead',
    options={'disp': True, 'xatol': 1e-10, 'maxfev': 100_000},
)
elapsed = perf_counter() - tic
print(f'Elapsed time: {elapsed/3600}h')
print(res)

# %%
tic = perf_counter()
res2 = spopt.minimize(
    lambda x: obj(x, limit=100),
    res.x,
    method='Nelder-Mead',
    options={'disp': True, 'xatol': 1e-10, 'maxfev': 100_000},
)
elapsed2 = perf_counter() - tic
print(f'Elapsed time: {elapsed2/3600}h')
print(res2)

# %%
rel_error(res.x, res2.x)

# %%
tic = perf_counter()
res3 = spopt.minimize(
    obj,
    res2.x,
    method='Nelder-Mead',
    options={'disp': True, 'xatol': 1e-10, 'maxfev': 100_000},
)
elapsed3 = perf_counter() - tic
print(f'Elapsed time: {elapsed3/3600}h')
print(res3)

# %%
rel_error(res2.x, res3.x)

# %%
rom_diag = vec_to_rom(res3.x)

# %%
rom_diag.A.matrix

# %%
rom_diag.Ad[0].matrix

# %%
rom_diag.B.matrix

# %%
rom_diag.C.matrix

# %%
mus_rom_diag = rom_diag.A.matrix.diagonal()
sigmas_rom_diag = rom_diag.Ad[0].matrix.diagonal()
poles_rom_diag = np.array(
    [
        lam(mu, sigma, k)
        for mu, sigma in zip(mus_rom_diag, sigmas_rom_diag)
        for k in range(-N, N + 1)
    ]
)

# %%
poles_rom_diag.real.max()

# %%
sorted(poles_rom_diag, reverse=True)[:10]

# %%
fig, ax = plt.subplots()
ax.plot(poles.real, poles.imag, '.')
ax.plot(poles_rom_diag.real, poles_rom_diag.imag, 'x')
ax.set_yscale('symlog', linthresh=1)
ax.grid()

# %%
(fom - rom_diag).transfer_function.h2_norm(
    epsrel=1e-10, limit=1000
) / fom.transfer_function.h2_norm(epsrel=1e-10, limit=1000)

# %%
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = rom0.transfer_function.mag_plot(w, label='Initial ROM')
_ = rom_diag.transfer_function.mag_plot(w, label='Diagonal ROM')
_ = plt.legend()

# %%
_ = (fom - rom0).transfer_function.mag_plot(w, label='Initial ROM error')
_ = (fom - rom_diag).transfer_function.mag_plot(w, label='Diagonal ROM error')
_ = plt.legend()

# %% [markdown]
# ## Checking interpolation

# %%
b1, b2 = rom_diag.B.matrix
c1, c2 = rom_diag.C.matrix.T

# %%
b1, b2

# %%
c1, c2

# %%
H = fom.transfer_function.eval_tf
dH = fom.transfer_function.eval_dtf
Hr = rom_diag.transfer_function.eval_tf
dHr = rom_diag.transfer_function.eval_dtf

# %%
phi = lambda mu, lam: 1 / (1 + tau * (lam - mu))
psi = lambda mu, lam: 1 / (1 + tau * (lam - mu)) ** 2
rho = lambda mu, lam: tau**2 * (lam - mu) / (1 + tau * (lam - mu)) ** 3
lambdas_rom = np.array(
    [[lam(mus_rom_diag[i], sigmas_rom_diag[i], k) for k in range(-N, N + 1)] for i in range(2)]
)

# %%
H2 = lambda i: np.sum(
    [phi(mus_rom_diag[i], lam).conj() * H(-lam.conj()) for lam in lambdas_rom[i]],
    axis=0,
)
Hr2 = lambda i: np.sum(
    [phi(mus_rom_diag[i], lam).conj() * Hr(-lam.conj()) for lam in lambdas_rom[i]],
    axis=0,
)
dH2 = lambda i: np.sum(
    [phi(mus_rom_diag[i], lam).conj() * dH(-lam.conj()) for lam in lambdas_rom[i]],
    axis=0,
)
dHr2 = lambda i: np.sum(
    [phi(mus_rom_diag[i], lam).conj() * dHr(-lam.conj()) for lam in lambdas_rom[i]],
    axis=0,
)
ddH2 = lambda i: np.sum(
    [
        psi(mus_rom_diag[i], lam).conj() * dH(-lam.conj())
        - rho(mus_rom_diag[i], lam).conj() * H(-lam.conj())
        for lam in lambdas_rom[i]
    ],
    axis=0,
)
ddHr2 = lambda i: np.sum(
    [
        psi(mus_rom_diag[i], lam).conj() * dHr(-lam.conj())
        - rho(mus_rom_diag[i], lam).conj() * Hr(-lam.conj())
        for lam in lambdas_rom[i]
    ],
    axis=0,
)

# %%
H2s = [H2(i) for i in range(2)]
Hr2s = [Hr2(i) for i in range(2)]
dH2s = [dH2(i) for i in range(2)]
dHr2s = [dHr2(i) for i in range(2)]
ddH2s = [ddH2(i) for i in range(2)]
ddHr2s = [ddHr2(i) for i in range(2)]

# %%
H2s[0].real

# %%
Hr2s[0].real

# %%
rel_error(H2s[0] @ b1, Hr2s[0] @ b1)

# %%
rel_error(H2s[1] @ b2, Hr2s[1] @ b2)

# %%
rel_error(c1.conj() @ H2s[0], c1.conj() @ Hr2s[0])

# %%
rel_error(c2.conj() @ H2s[1], c2.conj() @ Hr2s[1])

# %%
rel_error(c1.conj() @ dH2s[0] @ b1, c1.conj() @ dHr2s[0] @ b1)

# %%
rel_error(c2.conj() @ dH2s[1] @ b2, c2.conj() @ dHr2s[1] @ b2)

# %%
rel_error(c1.conj() @ ddH2s[0] @ b1, c1.conj() @ ddHr2s[0] @ b1)

# %%
rel_error(c2.conj() @ ddH2s[1] @ b2, c2.conj() @ ddHr2s[1] @ b2)
