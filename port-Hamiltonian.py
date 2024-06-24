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
# # Port-Hamiltonian example

# %% [markdown]
# ## Imports

# %%
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spla
import scipy.optimize as spopt
from pymor.models.iosys import PHLTIModel

# %% [markdown]
# ## Full-order model


# %%
# fmt: off
A = np.array([
    [-0.1,    0,    0,  0,  0,  0],
    [   0, -0.1,    1,  0,  0,  0],
    [   0,   -1, -0.1,  0,  0,  0],
    [   0,    0,    0, -1,  0,  0],
    [   0,    0,    0,  0, -1,  1],
    [   0,    0,    0,  0,  -1, -1],
])
B = np.array([
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5],
    [1, 6],
])
# fmt: on
J = (A - A.T) / 2
R = -(A + A.T) / 2
fom = PHLTIModel.from_matrices(J, R, B)

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
Ar0 = A[:3, :3]
Br0 = B[:3, :]
Jr0 = (Ar0 - Ar0.T) / 2
Rr0 = -(Ar0 + Ar0.T) / 2
rom0 = PHLTIModel.from_matrices(Jr0, Rr0, Br0)

# %%
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = rom0.transfer_function.mag_plot(w, label='Initial ROM')
_ = plt.legend()

# %%
(fom - rom0).h2_norm() / fom.h2_norm()


# %% [markdown]
# ## Diagonal ROM optimization

# %% [markdown]
# ROM:
# $$
# \begin{align*}
#   \hat{J} - \hat{R} =
#   \begin{bmatrix}
#     a & 0 & 0 \\
#     0 & x & y \\
#     0 & -y & x
#   \end{bmatrix}, \quad
#   \hat{B} \in \mathbb{R}^{3 \times 2}
# \end{align*}
# $$


# %%
def vec_to_lti(x):
    J = np.array([[0, 0, 0], [0, 0, x[2]], [0, -x[2], 0]])
    R = np.diag([-x[0], -x[1], -x[1]])
    B = x[3:].reshape(3, 2)
    return PHLTIModel.from_matrices(J, R, B)


def obj(x):
    rom = vec_to_lti(x)
    return (fom - rom).h2_norm()


# %%
x0 = np.concatenate(
    (
        [-0.1, -0.1, 1],
        Br0.reshape(-1),
    )
)
tic = perf_counter()
res = spopt.minimize(obj, x0, method='Nelder-Mead', options={'disp': True, 'xatol': 1e-10})
elapsed = perf_counter() - tic
print(f'Elapsed time: {elapsed}')

# %%
res

# %%
rom_normal = vec_to_lti(res.x)

# %%
poles_rom_normal = rom_normal.poles()

# %%
poles_rom_normal.real.max()

# %%
fig, ax = plt.subplots()
ax.plot(poles.real, poles.imag, '.')
ax.plot(poles_rom_normal.real, poles_rom_normal.imag, 'x')
ax.grid()

# %%
(fom - rom_normal).h2_norm() / fom.h2_norm()


# %%
rom_normal.poles()

# %%
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = rom0.transfer_function.mag_plot(w, label='Initial ROM')
_ = rom_normal.transfer_function.mag_plot(w, label='Normal ROM')
_ = plt.legend()

# %%
_ = (fom - rom0).transfer_function.mag_plot(w, c='C1', label='Initial ROM error')
_ = (fom - rom_normal).transfer_function.mag_plot(w, c='C2', label='Normal ROM error')
_ = plt.legend()

# %% [markdown]
# ## Checking interpolation

# %%
p = rom_normal.poles()
print(p)

# %%
rom_normal.J.matrix

# %%
rom_normal.R.matrix

# %%
rom_normal.G.matrix

# %%
lam_r = -rom_normal.R.matrix[0, 0]
A1 = rom_normal.J.matrix[1:, 1:] - rom_normal.R.matrix[1:, 1:]
T, Z = spla.schur(A1, output='complex')
lam_c1, lam_c2 = T.diagonal()
tB = spla.block_diag(np.eye(1), Z.conj().T) @ rom_normal.G.matrix
b_r, b_c1, b_c2 = tB.conj()

# %%
print(lam_r)

# %%
print(lam_c1)

# %%
print(lam_c2)

# %%
print(b_r)

# %%
print(b_c1)

# %%
print(b_c2)

# %%
H = fom.transfer_function.eval_tf
dH = fom.transfer_function.eval_dtf
Hr = rom_normal.transfer_function.eval_tf
dHr = rom_normal.transfer_function.eval_dtf
H2 = lambda s: H(s) + H(s).conj().T
Hr2 = lambda s: Hr(s) + Hr(s).conj().T


# %%
def interp_rel_err(f1, f2, s, b=None, c=None):
    m1 = f1(s)
    m2 = f2(s)
    if b is not None:
        m1 = m1 @ b
        m2 = m2 @ b
    if c is not None:
        m1 = c.conj() @ m1
        m2 = c.conj() @ m2
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
interp_rel_err(H, Hr, -lam_r, b=b_r)

# %%
interp_rel_err(H, Hr, -lam_c1.conj(), b=b_c1)

# %%
interp_rel_err(H, Hr, -lam_c2.conj(), b=b_c2)

# %% [markdown]
# $H'(-\lambda) b = \hat{H}'(-\lambda) b$ or
# $c^* H'(-\lambda) = c^* \hat{H}'(-\lambda)$?

# %%
[1 / np.linalg.cond(dH(-lam) - dHr(-lam)) for lam in p]

# %% [markdown]
# $H(-\lambda) + H(-\lambda)^* = \hat{H}(-\lambda) + \hat{H}(-\lambda)^*$?

# %%
[interp_rel_err(H2, Hr2, -lam) for lam in p]

# %% [markdown]
# $(H(-\lambda) + H(-\lambda)^*) b = (\hat{H}(-\lambda) + \hat{H}(-\lambda)^*) b$?

# %%
[1 / np.linalg.cond(H2(-lam) - Hr2(-lam)) for lam in p]

# %%
interp_rel_err(H2, Hr2, -lam_r, b=b_r)

# %%
interp_rel_err(H2, Hr2, -lam_c1.conj(), b=b_c1)

# %%
interp_rel_err(H2, Hr2, -lam_c2.conj(), b=b_c2)

# %% [markdown]
# $b^* H'(-\lambda) b = b^* \hat{H}'(-\lambda) b$?

# %%
interp_rel_err(dH, dHr, -lam_r, b=b_r, c=b_r)

# %%
interp_rel_err(dH, dHr, -lam_c1.conj(), b=b_c1, c=b_c1)

# %%
interp_rel_err(dH, dHr, -lam_c2.conj(), b=b_c2, c=b_c2)

# %% [markdown]
# ## Non-diagonal ROM optimization

# %% [markdown]
# ROM:
# $$
# \begin{align*}
#   \hat{J} =
#   \begin{bmatrix}
#     0 & j_1 & j_2 \\
#     -j_1 & 0 & j_3 \\
#     -j_2 & -j_3 & 0
#   \end{bmatrix}, \quad
#   \hat{R} =
#   \begin{bmatrix}
#     r_1 & 0 & 0 \\
#     r_2 & r_4 & 0 \\
#     r_3 & r_5 & r_6
#   \end{bmatrix}
#   \begin{bmatrix}
#     r_1 & r_2 & r_3 \\
#     0 & r_4 & r_5 \\
#     0 & 0 & r_6
#   \end{bmatrix}, \quad
#   \hat{B} \in \mathbb{R}^{3 \times 2}
# \end{align*}
# $$


# %%
def vec_to_lti2(x):
    j, r, b = np.split(x, np.cumsum([3, 6]))
    J = np.array([[0, j[0], j[1]], [-j[0], 0, j[2]], [-j[1], -j[2], 0]])
    R1 = np.array([[r[0], r[1], r[2]], [0, r[3], r[4]], [0, 0, r[5]]])
    R = R1.T @ R1
    B = b.reshape(3, 2)
    return PHLTIModel.from_matrices(J, R, B)


def obj2(x):
    rom = vec_to_lti2(x)
    return (fom - rom).h2_norm()


# %%
r_diag_0 = np.sqrt(rom_normal.R.matrix.diagonal())
x20 = np.concatenate(
    (
        [0, 0, rom_normal.J.matrix[1, 2]],
        [r_diag_0[0], 0, 0, r_diag_0[1], 0, r_diag_0[2]],
        rom_normal.G.matrix.reshape(-1),
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
rom_full = vec_to_lti2(res2.x)

# %%
poles_rom_full = rom_full.poles()

# %%
poles_rom_full.real.max()

# %%
fig, ax = plt.subplots()
ax.plot(poles.real, poles.imag, '.')
ax.plot(poles_rom_normal.real, poles_rom_normal.imag, 'x')
ax.plot(poles_rom_full.real, poles_rom_full.imag, '+')
ax.grid()

# %%
(fom - rom_full).h2_norm() / fom.h2_norm()


# %%
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = rom0.transfer_function.mag_plot(w, label='Initial ROM')
_ = rom_normal.transfer_function.mag_plot(w, label='Normal ROM')
_ = rom_full.transfer_function.mag_plot(w, label='Full ROM')
_ = plt.legend()

# %%
_ = (fom - rom0).transfer_function.mag_plot(w, c='C1', label='Initial ROM error')
_ = (fom - rom_normal).transfer_function.mag_plot(w, c='C2', label='Normal ROM error')
_ = (fom - rom_full).transfer_function.mag_plot(w, c='C3', label='Full ROM error')
_ = plt.legend()

# %%
Jrf = rom_full.J.matrix
print(Jrf)

# %%
Rrf = rom_full.R.matrix
print(Rrf)

# %%
Arf = Jrf - Rrf
print(Arf)

# %%
spla.eigvalsh(Rrf)

# %%
Brf = rom_full.G.matrix
print(Brf)

# %%
np.array([res2.x[3:6], np.pad(res2.x[6:8], (1, 0)), [0, 0, res2.x[8]]])

# %% [markdown]
# ## Checking interpolation

# %%
pf = rom_full.poles()
print(pf)

# %%
Arf @ Arf.T - Arf.T @ Arf

# %%
Hrf = rom_full.transfer_function.eval_tf
dHrf = rom_full.transfer_function.eval_dtf
Hrf2 = lambda s: Hrf(s) + Hrf(s).conj().T

# %% [markdown]
# $H(-\lambda) = \hat{H}(-\lambda)$?

# %%
[interp_rel_err(H, Hr2, -lam) for lam in pf]

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
# $H'(-\lambda) b = \hat{H}'(-\lambda) b$ or
# $c^* H'(-\lambda) = c^* \hat{H}'(-\lambda)$?

# %%
[np.linalg.cond(dH(-lam) - dHrf(-lam)) for lam in pf]

# %% [markdown]
# $H(-\lambda) + H(-\lambda)^* = \hat{H}(-\lambda) + \hat{H}(-\lambda)^*$?

# %%
[interp_rel_err(H2, Hrf2, -lam) for lam in pf]

# %% [markdown]
# $(H(-\lambda) + H(-\lambda)^*) b = (\hat{H}(-\lambda) + \hat{H}(-\lambda)^*) b$?

# %%
[np.linalg.cond(H2(-lam) - Hrf2(-lam)) for lam in pf]
