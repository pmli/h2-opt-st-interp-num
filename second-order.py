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
# # Second-order example

# %% [markdown]
# $$(s + 0.1) (s + 0.2) = s^2 + 0.3 s + 0.02, \quad (s + 0.1)^2 + 1 = s^2 + 0.2 s + 1.01$$
#
# $$(s + 1) (s + 2) = s^2 + 3 s + 2, \quad (s + 1)^2 + 1 = s^2 + 2 s + 2$$

# %% [markdown]
# ## Imports

# %%
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spla
import scipy.optimize as spopt
from pymor.models.iosys import SecondOrderModel

# %% [markdown]
# ## Full-order model


# %%
M = np.eye(4)
# fmt: off
E = np.array([
    [0.3,   0, 0, 0],
    [  0, 0.2, 0, 0],
    [  0,   0, 3, 0],
    [  0,   0, 0, 2],
])
K = np.array([
    [0.02,    0, 0, 0],
    [   0, 1.01, 0, 0],
    [   0,    0, 2, 0],
    [   0,    0, 0, 2],
])
B = np.array([
    [1, 1, 1],
    [1, 2, 2],
    [1, 3, 1],
    [1, 4, 2],
])
C = np.array([
    [1, 1, 1, 1],
    [1, 2, 3, 4],
])
# fmt: on
fom = SecondOrderModel.from_matrices(M, E, K, B, C)

# %%
fom

# %%
poles = fom.poles()

# %%
fig, ax = plt.subplots()
ax.plot(poles.real, poles.imag, '.')
ax.grid()

# %%
w = (1e-2, 1e1)
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = plt.legend()

# %% [markdown]
# ## Initial reduced-order model

# %%
Mr0 = M[:2, :2]
Er0 = E[:2, :2]
Kr0 = K[:2, :2]
Br0 = B[:2, :]
Cr0 = C[:, :2]
rom0 = SecondOrderModel.from_matrices(Mr0, Er0, Kr0, Br0, Cr0)

# %%
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = rom0.transfer_function.mag_plot(w, label='Initial ROM')
_ = plt.legend()

# %%
_ = (fom - rom0).transfer_function.mag_plot(w, label='Initial ROM error')
_ = plt.legend()

# %%
(fom - rom0).h2_norm() / fom.h2_norm()


# %% [markdown]
# ## Diagonal ROM optimization

# %% [markdown]
# ROM:
# $$
# \begin{align*}
#   \hat{M} = I, \quad
#   \hat{E} = \operatorname{diag}(e_1, e_2), \quad
#   \hat{K} = \operatorname{diag}(k_1, k_2), \quad
#   \hat{B} \in \mathbb{R}^{2 \times 3}, \quad
#   \hat{C} \in \mathbb{R}^{2 \times 2}
# \end{align*}
# $$


# %%
def vec_to_som(x):
    es, ks, bs, cs = np.split(x, np.cumsum([2, 2, 6]))
    M = np.eye(2)
    E = np.diag(es)
    K = np.diag(ks)
    B = bs.reshape(2, 3)
    C = cs.reshape(2, 2)
    return SecondOrderModel.from_matrices(M, E, K, B, C)


def obj(x):
    rom = vec_to_som(x)
    return (fom - rom).h2_norm()


# %%
x0 = np.concatenate(
    (
        rom0.E.matrix.diagonal(),
        rom0.K.matrix.diagonal(),
        rom0.B.matrix.reshape(-1),
        rom0.Cp.matrix.reshape(-1),
    )
)
tic = perf_counter()
res = spopt.minimize(
    obj,
    x0,
    method='Nelder-Mead',
    options={'disp': True, 'xatol': 1e-10, 'maxfev': 10000},
)
elapsed = perf_counter() - tic
print(f'Elapsed time: {elapsed}')

# %%
res

# %%
rom_diag = vec_to_som(res.x)

# %%
poles_rom_diag = rom_diag.poles()

# %%
poles_rom_diag.real.max()

# %%
fig, ax = plt.subplots()
ax.plot(poles.real, poles.imag, '.')
ax.plot(poles_rom_diag.real, poles_rom_diag.imag, 'x')
ax.grid()

# %%
(fom - rom_diag).h2_norm() / fom.h2_norm()

# %%
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = rom0.transfer_function.mag_plot(w, label='Initial ROM')
_ = rom_diag.transfer_function.mag_plot(w, label='Diagonal ROM')
_ = plt.legend()

# %%
_ = (fom - rom0).transfer_function.mag_plot(w, c='C1', label='Initial ROM error')
_ = (fom - rom_diag).transfer_function.mag_plot(w, c='C2', label='Diagonal ROM error')
_ = plt.legend()

# %% [markdown]
# ## Checking interpolation

# %%
rom_diag.E.matrix

# %%
rom_diag.K.matrix

# %%
rom_diag.B.matrix

# %%
rom_diag.Cp.matrix

# %%
p = rom_diag.poles()
print(p)

# %%
lam11, lam12, lam21, lam22 = p

# %%
c1, c2 = rom_diag.Cp.matrix.T
b1, b2 = rom_diag.B.matrix

# %%
print(c1)

# %%
print(c2)

# %%
print(b1)

# %%
print(b2)

# %%
H = fom.transfer_function.eval_tf
dH = fom.transfer_function.eval_dtf
Hr = rom_diag.transfer_function.eval_tf
dHr = rom_diag.transfer_function.eval_dtf
H2 = lambda s1, s2: H(s1) - H(s2)
Hr2 = lambda s1, s2: Hr(s1) - Hr(s2)


# %%
def interp_rel_err(f1, f2, s, b=None, c=None):
    m1 = f1(s)
    m2 = f2(s)
    if b is not None:
        m1 = m1 @ b
        m2 = m2 @ b
    if c is not None:
        m1 = c @ m1
        m2 = c @ m2
    return spla.norm(m1 - m2) / spla.norm(m1)


def interp_rel_err2(f1, f2, s1, s2, b=None, c=None):
    m1 = f1(s1, s2)
    m2 = f2(s1, s2)
    if b is not None:
        m1 = m1 @ b
        m2 = m2 @ b
    if c is not None:
        m1 = c @ m1
        m2 = c @ m2
    return spla.norm(m1 - m2) / spla.norm(m1)


# %% [markdown]
# $H(-\lambda) = \hat{H}(-\lambda)$?

# %%
[interp_rel_err(H, Hr, -lam) for lam in p]

# %% [markdown]
# $H'(-\lambda) = \hat{H}'(-\lambda)$?

# %%
[interp_rel_err(dH, dHr, -lam) for lam in p]


# %% [markdown]
# $H(-\lambda) b = \hat{H}(-\lambda) b$ or
# $c^* H(-\lambda) = c^* \hat{H}(-\lambda)$?


# %%
[1 / np.linalg.cond(H(-lam) - Hr(-lam)) for lam in p]

# %%
interp_rel_err(H, Hr, -lam11, b=b1)

# %%
interp_rel_err(H, Hr, -lam12, b=b1)

# %%
interp_rel_err(H, Hr, -lam21, b=b2)

# %%
interp_rel_err(H, Hr, -lam22, b=b2)

# %%
interp_rel_err(H, Hr, -lam11, c=c1)

# %%
interp_rel_err(H, Hr, -lam12, c=c1)

# %%
interp_rel_err(H, Hr, -lam21, c=c2)

# %%
interp_rel_err(H, Hr, -lam22, c=c2)

# %% [markdown]
# $H'(-\lambda) b = \hat{H}'(-\lambda) b$ or
# $c^* H'(-\lambda) = c^* \hat{H}'(-\lambda)$?

# %%
[1 / np.linalg.cond(dH(-lam) - dHr(-lam)) for lam in p]

# %% [markdown]
# $H(-\lambda^+) - H(-\lambda^-) = \hat{H}(-\lambda^+) - \hat{H}(-\lambda^-)$?

# %%
interp_rel_err2(H2, Hr2, -lam11, -lam12)

# %%
interp_rel_err2(H2, Hr2, -lam21, -lam22)

# %% [markdown]
# $(H(-\lambda^+) - H(-\lambda^-)) b = (\hat{H}(-\lambda^+) - \hat{H}(-\lambda^-)) b$?

# %%
[[1 / np.linalg.cond(H2(-lam1, -lam2) - Hr2(-lam1, -lam2)) for lam2 in p] for lam1 in p]

# %%
interp_rel_err2(H2, Hr2, -lam11, -lam12, b=b1)

# %%
interp_rel_err2(H2, Hr2, -lam21, -lam22, b=b2)

# %% [markdown]
# $c^* (H(-\lambda^+) - H(-\lambda^-)) = c^* (\hat{H}(-\lambda^+) - \hat{H}(-\lambda^-))$?

# %%
interp_rel_err2(H2, Hr2, -lam11, -lam12, c=c1)

# %%
interp_rel_err2(H2, Hr2, -lam21, -lam22, c=c2)

# %% [markdown]
# $c^* H'(-\lambda) b = c^* \hat{H}'(-\lambda) b$?

# %%
interp_rel_err(dH, dHr, -lam11, b=b1, c=c1)

# %%
interp_rel_err(dH, dHr, -lam12, b=b1, c=c1)

# %%
interp_rel_err(dH, dHr, -lam21, b=b2, c=c2)

# %%
interp_rel_err(dH, dHr, -lam22, b=b2, c=c2)

# %% [markdown]
# ## Non-diagonal ROM optimization

# %% [markdown]
# ROM:
# $$
# \begin{align*}
#   \hat{M} = I, \quad
#   \hat{E} \in \mathbb{R}^{2 \times 2}, \quad
#   \hat{K} \in \mathbb{R}^{2 \times 2}, \quad
#   \hat{B} \in \mathbb{R}^{2 \times 3}, \quad
#   \hat{C} \in \mathbb{R}^{2 \times 2}
# \end{align*}
# $$


# %%
def vec_to_som2(x):
    es, ks, bs, cs = np.split(x, np.cumsum([4, 4, 6]))
    M = np.eye(2)
    E = es.reshape(2, 2)
    K = ks.reshape(2, 2)
    B = bs.reshape(2, 3)
    C = cs.reshape(2, 2)
    return SecondOrderModel.from_matrices(M, E, K, B, C)


def obj2(x):
    rom = vec_to_som2(x)
    return (fom - rom).h2_norm()


# %%
x20 = np.concatenate(
    (
        rom_diag.E.matrix.reshape(-1),
        rom_diag.K.matrix.reshape(-1),
        rom_diag.B.matrix.reshape(-1),
        rom_diag.Cp.matrix.reshape(-1),
    )
)
tic2 = perf_counter()
res2 = spopt.minimize(
    obj2,
    x20,
    method='Nelder-Mead',
    options={'disp': True, 'xatol': 1e-10, 'maxfev': 20000},
)
elapsed2 = perf_counter() - tic2
print(f'Elapsed time: {elapsed2}')

# %%
res2

# %%
rom_full = vec_to_som2(res2.x)

# %%
poles_rom_full = rom_full.poles()

# %%
poles_rom_full.real.max()

# %%
fig, ax = plt.subplots()
ax.plot(poles.real, poles.imag, '.')
ax.plot(poles_rom_diag.real, poles_rom_diag.imag, 'x')
ax.plot(poles_rom_full.real, poles_rom_full.imag, '+')
ax.grid()

# %%
(fom - rom_full).h2_norm() / fom.h2_norm()

# %%
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = rom0.transfer_function.mag_plot(w, label='Initial ROM')
_ = rom_diag.transfer_function.mag_plot(w, label='Diagonal ROM')
_ = rom_full.transfer_function.mag_plot(w, label='Full ROM')
_ = plt.legend()

# %%
_ = (fom - rom0).transfer_function.mag_plot(w, c='C1', label='Initial ROM error')
_ = (fom - rom_diag).transfer_function.mag_plot(w, c='C2', label='Diagonal ROM error')
_ = (fom - rom_full).transfer_function.mag_plot(w, c='C3', label='Full ROM error')
_ = plt.legend()

# %%
rom_full.E.matrix

# %%
rom_full.K.matrix

# %%
spla.norm(
    rom_full.E.matrix @ rom_full.K.matrix - rom_full.K.matrix @ rom_full.E.matrix
) / spla.norm(rom_full.E.matrix @ rom_full.K.matrix)

# %%
rom_full.B.matrix

# %%
rom_full.Cp.matrix

# %% [markdown]
# ## Checking interpolation

# %%
pf = rom_full.poles()
print(pf)

# %%
Hrf = rom_full.transfer_function.eval_tf
dHrf = rom_full.transfer_function.eval_dtf
Hrf2 = lambda s1, s2: Hrf(s1) - Hrf(s2)

# %% [markdown]
# $H(-\lambda) = \hat{H}(-\lambda)$?

# %%
[interp_rel_err(H, Hrf, -lam) for lam in pf]

# %% [markdown]
# $H'(-\lambda) = \hat{H}'(-\lambda)$?

# %%
[interp_rel_err(dH, dHrf, -lam) for lam in pf]

# %% [markdown]
# $H(-\lambda) b = \hat{H}(-\lambda) b$ or
# $c^* H(-\lambda) = c^* \hat{H}(-\lambda)$?

# %%
[np.linalg.cond(H(-lam) - Hrf(-lam)) for lam in pf]

# %% [markdown]
# $H(-\lambda^+) - H(-\lambda^-) = \hat{H}(-\lambda^+) - \hat{H}(-\lambda^-)$?

# %%
interp_rel_err2(H2, Hrf2, -lam11, -lam12)

# %%
interp_rel_err2(H2, Hrf2, -lam21, -lam22)

# %% [markdown]
# $(H(-\lambda^+) - H(-\lambda^-)) b = (\hat{H}(-\lambda^+) - \hat{H}(-\lambda^-)) b$?

# %%
[[np.linalg.cond(H2(-lam1, -lam2) - Hrf2(-lam1, -lam2)) for lam1 in pf] for lam2 in pf]
