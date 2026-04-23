"""
Generate Q/U maps from a pure-EE power spectrum (BB = EB = 0) using healpy.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# ── parameters ────────────────────────────────────────────────────────────────
NSIDE = 512
LMAX  = 2 * NSIDE          # safe upper limit for NSIDE=512
ell   = np.arange(LMAX + 1)

# ── define EE power spectrum ──────────────────────────────────────────────────
# Example: flat D_ell = ell(ell+1) C_ell / 2pi = 1 for ell >= 2.
# Replace this array with your actual C_ell^EE.
D_EE = np.zeros(LMAX + 1)
D_EE[2:] = 1.0                    # arbitrary units; flat angular power

with np.errstate(divide="ignore", invalid="ignore"):
    C_EE = np.where(ell >= 2, 2.0 * np.pi * D_EE / (ell * (ell + 1)), 0.0)

# All other spectra are zero
TT = np.zeros(LMAX + 1)
EE = C_EE
BB = np.zeros(LMAX + 1)   # BB  = 0
TE = np.zeros(LMAX + 1)
EB = np.zeros(LMAX + 1)   # EB  = 0
TB = np.zeros(LMAX + 1)

# ── generate alm from spectra ─────────────────────────────────────────────────
# synalm with new=True expects [TT, EE, BB, TE, EB, TB] and returns
# (alm_T, alm_E, alm_B) as complex arrays of size hp.Alm.getsize(lmax).
alm_T, alm_E, alm_B = hp.synalm(
    [TT, EE, BB, TE, EB, TB],
    lmax=LMAX,
    new=True,
)

print(f"alm_T shape: {alm_T.shape}")
print(f"alm_E shape: {alm_E.shape}  (non-zero by construction)")
print(f"alm_B shape: {alm_B.shape}  (should be ~0 — pure E-mode)")

# ── convert alm → T, Q, U maps ───────────────────────────────────────────────
# alm2map with three alm returns the (T, Q, U) tuple.
# Q = Re(  _2Y_lm * alm_E ) ,  U = Im(...)  via spin-2 harmonics.
T_map, Q_map, U_map = hp.alm2map([alm_T, alm_E, alm_B], nside=NSIDE)

print(f"\nMap pixel count : {T_map.size}")
print(f"Q rms           : {Q_map.std():.4e}")
print(f"U rms           : {U_map.std():.4e}")

# ── verify: reconstruct power spectra from the maps ──────────────────────────
cl_rec = hp.anafast([T_map, Q_map, U_map], lmax=LMAX)
# anafast returns [TT, EE, BB, TE, EB, TB]
ell_r  = np.arange(len(cl_rec[0]))
fac    = ell_r * (ell_r + 1) / (2.0 * np.pi)

labels    = ["TT", "EE", "BB", "TE", "EB", "TB"]
input_cls = [TT,   EE,   BB,   TE,   EB,   TB]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, rec, inp, lbl in zip(axes.flat, cl_rec, input_cls, labels):
    sl = slice(2, None)
    ax.plot(ell_r[sl], (fac * rec)[sl],       label="recovered", lw=1.5)
    ax.plot(ell_r[sl], (fac * inp)[sl], "--",  label="input",     lw=1.5)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(rf"$\ell(\ell+1)C_\ell^{{\rm {lbl}}}/2\pi$")
    ax.set_title(lbl)
    ax.legend(fontsize=8)

plt.suptitle(r"Round-trip check: pure $EE$ spectrum (BB = EB = 0)", y=1.01)
plt.tight_layout()
plt.savefig("alm_roundtrip_check.png", dpi=150, bbox_inches="tight")
plt.show()

# ── optional: visualise Q map ─────────────────────────────────────────────────
hp.mollview(Q_map, title="Q map (from pure EE)", unit="arb.")
plt.savefig("Q_map.png", dpi=150)
plt.show()
