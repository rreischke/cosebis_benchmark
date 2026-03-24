"""
get_cosebis.py — Compute high-precision COSEBIs benchmark filter functions.

Computes the E-mode filter functions T_+(theta), T_-(theta) and their
Fourier-space counterparts W(ell) for the Complete Orthogonal Sets of
E/B-Integrals (COSEBIs) following Schneider et al. (2010).

All T_+/T_- calculations are performed with mpmath arbitrary-precision
arithmetic (controlled by --dps).  W(ell) integrals are evaluated with
the `levin` C++ extension (part of OneCovariance) using Levin quadrature,
parallelised over ell values via joblib (threading backend, all cores).

Usage
-----
    python get_cosebis.py [options]

Examples
--------
    # Default settings (tmin=2', tmax=300', Nmax=20, dps=200)
    python get_cosebis.py

    # Custom angular range and fewer modes
    python get_cosebis.py --tmin_mm 0.5 --tmax_mm 100 --Nmax_mm 5

    # Lower precision for a quick test
    python get_cosebis.py --dps 50 --N_theta 1000 --N_ell 1000

Output
------
For each mode n = 1 … Nmax the following files are written to --outdir_path:

    Tp_bench_mark_{tmin}_{tmax}_mode_{n:02d}.txt   — theta [rad], T_+(theta)
    Tm_bench_mark_{tmin}_{tmax}_mode_{n:02d}.txt   — theta [rad], T_-(theta)
    Well_{tmin}_{tmax}_mode_{n:02d}.txt            — ell, W(ell)
"""

import numpy as np
import levin
import multiprocessing
from functools import partial
from tqdm import tqdm
from mpmath import mp
import mpmath
from pathlib import Path
import argparse
import scipy.integrate
from scipy.special import jv
from scipy.signal import argrelextrema




# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Compute COSEBIs benchmark filter functions T_+, T_-, W(ell).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--ell_min",   type=float, default=1,        help="Minimum multipole")
parser.add_argument("--ell_max",   type=float, default=1e5,      help="Maximum multipole")
parser.add_argument("--N_ell",     type=int,   default=int(1e5), help="Number of ell sampling points")
parser.add_argument("--N_theta",   type=int,   default=int(1e4), help="Number of theta sampling points")
parser.add_argument("--num_cores", type=int,   default=4,        help="Number of worker processes for W(ell) (-1 = all cores)")
parser.add_argument("--dps",       type=int,   default=200,      help="mpmath decimal places of precision")
parser.add_argument("--Nmax_mm",   type=int,   default=20,       help="Number of COSEBIS modes to compute")
parser.add_argument("--tmin_mm",   type=float, default=2,        help="Minimum angular scale [arcmin]")
parser.add_argument("--tmax_mm",   type=float, default=300,      help="Maximum angular scale [arcmin]")
parser.add_argument("--outdir_path", type=str, default="./../benchmarks_cosebis",
                    help="Directory where output files are written")
args = parser.parse_args()

# Unpack for convenience
ell_min   = args.ell_min
ell_max   = args.ell_max
N_ell     = args.N_ell
N_theta   = args.N_theta
num_cores = None if args.num_cores == -1 else args.num_cores
Nmax_mm   = args.Nmax_mm
mp.dps    = args.dps          # set mpmath global precision

outdir = Path(args.outdir_path)
outdir.mkdir(parents=True, exist_ok=True)

# Convert angular limits from arcminutes to radians (exact, kept in mpmath)
arcmintorad = mp.pi / 10800   # pi / (60 * 180)
tmin_rad = mp.mpf(args.tmin_mm) * arcmintorad
tmax_rad = mp.mpf(args.tmax_mm) * arcmintorad

# Log-spaced theta grid in radians
theta_grid = [tmin_rad * mp.exp(mp.mpf(i) / (N_theta - 1) * mp.log(tmax_rad / tmin_rad))
              for i in range(N_theta)]

theta_grid_arcmin = [args.tmin_mm * mp.exp(mp.mpf(i) / (N_theta - 1) * mp.log(args.tmax_mm / args.tmin_mm))
              for i in range(N_theta)]

# Log-ratio used as integration upper limit (eq. 32 in Schneider+2010)
zmax = mp.log(tmax_rad / tmin_rad)

# Log-spaced ell grid (standard float64 precision is fine here)
ell = np.geomspace(ell_min, ell_max, N_ell)


# ---------------------------------------------------------------------------
# Helper: J-integrals (Schneider+2010, eq. 36)
# ---------------------------------------------------------------------------

def J(k, j, zmax):
    """Compute the auxiliary integral J_j^(k) = int_0^zmax z^j exp(-k z) dz.

    Evaluated via the incomplete gamma function:
        J = [Gamma(j+1) - Gamma(j+1, -k*zmax)] / (-k)^(j+1)

    The lower incomplete gamma form raises errors in mpmath, so we use
    Gamma - upper_incomplete_Gamma instead.
    """
    J = mp.gamma(j + 1) - mp.gammainc(j + 1, -k * zmax)
    return mp.fdiv(J, mp.power(-k, j + 1))


# ---------------------------------------------------------------------------
# Coefficient matrix c_nj  (Schneider+2010, eqs. 33–34)
# ---------------------------------------------------------------------------

# Allocate (Nmax+1) x (Nmax+2) matrix; row n holds coefficients for mode n
coeff_j = mp.matrix(Nmax_mm + 1, Nmax_mm + 2)

# Normalisation constraint: c_n,n+1 = 1 for all n (eq. 33)
for i in range(Nmax_mm + 1):
    coeff_j[i, i + 1] = mp.mpf(1.)

# Bootstrap n=1: solve 2x2 system from the two normalisation equations (33)
nn = 1
aa = [J(2, 0, zmax), J(2, 1, zmax)], [J(4, 0, zmax), J(4, 1, zmax)]
bb = [-J(2, nn + 1, zmax), -J(4, nn + 1, zmax)]
coeff_j_ini = mp.lu_solve(aa, bb)
coeff_j[1, 0] = coeff_j_ini[0]
coeff_j[1, 1] = coeff_j_ini[1]

# General case n >= 2: (n+1) equations per mode —
#   (n-1) orthogonality conditions (eq. 34) + 2 normalisation equations (eq. 33)
for nn in range(2, Nmax_mm + 1):
    aa = mp.matrix(int(nn + 1))
    bb = mp.matrix(int(nn + 1), 1)

    # Orthogonality conditions with all previously computed modes m < n
    for m in range(1, nn):
        for j in range(0, nn + 1):
            for i in range(0, m + 2):
                aa[m - 1, j] += J(1, i + j, zmax) * coeff_j[m, i]
        for i in range(0, m + 2):
            bb[int(m - 1)] -= J(1, i + nn + 1, zmax) * coeff_j[m, i]

    # Two normalisation equations (eq. 33)
    for j in range(nn + 1):
        aa[nn - 1, j] = J(2, j, zmax)
        aa[nn,     j] = J(4, j, zmax)
    bb[int(nn - 1)] = -J(2, nn + 1, zmax)
    bb[int(nn)]     = -J(4, nn + 1, zmax)

    temp_coeff = mp.lu_solve(aa, bb)
    coeff_j[nn, :len(temp_coeff)] = temp_coeff[:, 0].T

# Drop the unused n=0 row; row index n now corresponds to mode n+1
coeff_j = coeff_j[1:, :]

# ---------------------------------------------------------------------------
# Normalisation constants N_n  (Schneider+2010, eq. 35)
# ---------------------------------------------------------------------------

Nn = []
for nn in range(1, Nmax_mm + 1):
    temp_sum = mp.mpf(0)
    for i in range(nn + 2):
        for j in range(nn + 2):
            temp_sum += coeff_j[nn - 1, i] * coeff_j[nn - 1, j] * J(1, i + j, zmax)
    temp_Nn = mp.expm1(zmax) / temp_sum
    Nn.append(mp.sqrt(mp.fabs(temp_Nn)))   # choose N_n > 0

# ---------------------------------------------------------------------------
# Roots of T_+n^log  (used in the product form of T_+)
# ---------------------------------------------------------------------------

rn = []
for nn in range(1, Nmax_mm + 1):
    rn.append(mpmath.polyroots(coeff_j[nn - 1, :nn + 2][::-1], maxsteps=500, extraprec=100))


# ---------------------------------------------------------------------------
# Filter functions
# ---------------------------------------------------------------------------

def tplus(tmin, tmax, n, norm, root):
    """Compute T_+(theta) for mode n on a log-spaced theta grid.

    Parameters
    ----------
    tmin, tmax : mpmath.mpf
        Angular limits in radians.
    n : int
        Mode number (1-based).
    norm : mpmath.mpf
        Normalisation constant N_n.
    root : list of mpmath.mpf
        Roots of the polynomial filter.

    Returns
    -------
    theta : list of mpmath.mpf
    result : list of mpmath.mpf
    """
    theta = [mp.power(10, mp.log(tmin, 10) + (mp.log(tmax, 10) - mp.log(tmin, 10))
                      * mp.mpf(i) / (N_theta - 1))
             for i in range(N_theta)]
    z = [mp.log(t / theta[0]) for t in theta]
    result = [mp.mpf(1)] * len(theta)
    for r in range(n + 1):
        for i in range(len(result)):
            result[i] *= (z[i] - root[r])
    for i in range(len(result)):
        result[i] *= norm / mp.mpf(2)
    return theta, result


def t_minus(tmin, n, norm, coeff_row, theta):
    """Compute T_-(theta) from the closed-form expression in Schneider+2010.

    Eq. 38:
        T_{-n}^log(z) = a_{n2} * exp(-2z) - a_{n4} * exp(-4z)
                       + sum_{m=0}^{n} d_{nm} * z^m

    Coefficients (eq. 38):
        a_{n2} = 4  * sum_{j=0}^{n+1} c_{nj} * j! / (-2)^{j+1}
        a_{n4} = 12 * sum_{j=0}^{n+1} c_{nj} * j! / (-4)^{j+1}

    Polynomial coefficients (eq. 39):
        d_{nm} = c_{nm} + 4/m! * sum_{j=m}^{n+1} c_{nj} * j! * (-2)^{m-j-1}
                          * (3 * 2^{m-j-1} - 1)

    Parameters
    ----------
    tmin : mpmath.mpf
        Lower angular limit in radians (= theta[0]).
    n : int
        Mode number (1-based).
    norm : mpmath.mpf
        Normalisation constant N_n.
    coeff_row : mpmath matrix row
        Coefficients c_{nj} for this mode.
    theta : list of mpmath.mpf
        Theta grid (shared with T_+).

    Returns
    -------
    theta : list of mpmath.mpf
    result : list of mpmath.mpf
    """
    # Eq. 38: exponential amplitudes a_{n2}, a_{n4}
    a_n2 = mp.mpf(0)
    a_n4 = mp.mpf(0)
    for j in range(n + 2):
        c_nj  = coeff_row[j]
        fac_j = mp.factorial(j)
        a_n2 += c_nj * fac_j / mp.power(-2, j + 1)
        a_n4 += c_nj * fac_j / mp.power(-4, j + 1)
    a_n2 *= 4
    a_n4 *= 12

    # Eq. 39: polynomial coefficients d_{nm}
    d = []
    for m in range(n + 1):
        fac_m = mp.factorial(m)
        s = mp.mpf(0)
        for j in range(m, n + 2):
            exp = m - j - 1
            s += coeff_row[j] * mp.factorial(j) * mp.power(-2, exp) * (3 * mp.power(2, exp) - 1)
        d.append(coeff_row[m] + 4 / fac_m * s)

    # Evaluate T_- at each theta
    tmin_mp = mp.mpf(tmin)
    result = []
    for t in theta:
        z   = mp.log(t / tmin_mp)
        val = a_n2 * mp.exp(-2 * z) - a_n4 * mp.exp(-4 * z)
        for m in range(n + 1):
            val += d[m] * mp.power(z, m)
        result.append(val * norm / mp.mpf(2))

    return theta, result


def integrate_single_ell(ell_val, theta, tp, nmax=50):
    """Evaluate W(ell) at a single ell value using Levin quadrature.

    A fresh levin.Levin instance is created per call so workers do not share
    mutable state.  Sub-intervals are placed at every nmax-th local maximum of
    J_0(ell*theta) to guide the adaptive quadrature across oscillations.

    Parameters
    ----------
    ell_val : float
        Single multipole at which to evaluate W.
    theta : np.ndarray, shape (N_theta,)
        Angular grid in radians (float64).
    tp : np.ndarray, shape (N_theta, 1)
        T_+(theta) values (float64).
    nmax : int, optional
        Stride for selecting J_0 maxima as sub-interval boundaries.

    Returns
    -------
    float
        W(ell_val).
    """
    lev_w = levin.Levin(0, 16, 128, 1e-14, 1000, 1)
    well = jv(0, ell_val * theta)
    lev_w.init_w_ell(theta, well[:,None])
    lev_w.init_integral(theta,(theta*tp[:,0])[:,None],True,True)
    maxima_idx = argrelextrema(well, np.greater)[0][::nmax]
    theta_maxima = np.zeros(len(maxima_idx) + 2)
    theta_maxima[1:-1] = theta[maxima_idx]
    theta_maxima[0] = theta[0]
    theta_maxima[-1] = theta[-1]
    return lev_w.cquad_integrate_single_well(theta_maxima,0)[0]


def _integrate_chunk(ell_chunk, theta, tp):
    """Evaluate W(ell) for a contiguous block of ell values (one process).

    Parameters
    ----------
    ell_chunk : np.ndarray
        Subset of ell values assigned to this worker.
    theta : np.ndarray, shape (N_theta,)
        Angular grid in radians (float64).
    tp : np.ndarray, shape (N_theta, 1)
        T_+(theta) values (float64).

    Returns
    -------
    list of float
        W(ell) for each element of ell_chunk, in order.
    """
    return [integrate_single_ell(e, theta, tp, nmax=5) for e in ell_chunk]


def get_W_ell(ell, theta, tp, n_jobs=None):
    """Compute W(ell) = int T_+(theta) J_0(ell*theta) theta d(theta)
    using the Levin.

    The ell array is randomly shuffled before chunking so that each worker
    gets a mix of cheap (small ell, few J_0 oscillations) and expensive
    (large ell, many oscillations) values.

    Parameters
    ----------
    ell : np.ndarray, shape (N_ell,)
        Multipoles at which to evaluate W.
    theta : np.ndarray, shape (N_theta,), dtype float64
        Angular grid in radians.
    tp : np.ndarray, shape (N_theta, 1), dtype float64
        T_+(theta) values.
    n_jobs : int or None, optional
        Number of worker processes.  None uses all available cores.

    Returns
    -------
    result_levin : np.ndarray, shape (N_ell, 1)
        W(ell) evaluated at each ell, in the original ell order.
    """
    n_workers = n_jobs or multiprocessing.cpu_count()
    # shuffle indices so each chunk gets a mix of small and large ell
    perm = np.random.permutation(len(ell))
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(ell))
    chunks = np.array_split(ell[perm], n_workers)
    worker = partial(_integrate_chunk, theta=theta, tp=tp)
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(processes=n_workers) as pool:
        chunk_results = list(tqdm(
            pool.imap(worker, chunks),
            total=len(chunks), desc="W(ell)", unit="chunk"
        ))
    vals_shuffled = np.array([v for chunk in chunk_results for v in chunk])
    return vals_shuffled[inv_perm, None]
# ---------------------------------------------------------------------------
# Compute T_+ and T_- for all modes
# ---------------------------------------------------------------------------

Tp_bench_mark = []
Tm_bench_mark = []

Nmin = 1
for nn in range(Nmin, Nmax_mm + 1):
    n = nn - 1
    print(f"Computing Tpm mode {nn}/{Nmax_mm} ...")
    Tp_bench_mark.append(tplus(tmin_rad, tmax_rad, nn, Nn[n], rn[n]))
    Tm_bench_mark.append(t_minus(tmin_rad, nn, Nn[n], coeff_j[n, :], Tp_bench_mark[n][0]))


# ---------------------------------------------------------------------------
# Save T_+ and T_- to disk
# ---------------------------------------------------------------------------

n_digits = mp.dps   # use full precision when writing

# Common parameter header written into every output file
param_header = (
    f"# tmin_mm={args.tmin_mm} tmax_mm={args.tmax_mm} Nmax_mm={args.Nmax_mm} "
    f"dps={args.dps} N_theta={args.N_theta} N_ell={args.N_ell} "
    f"ell_min={args.ell_min} ell_max={args.ell_max} num_cores={args.num_cores}\n"
)

for i in range(1, Nmax_mm + 1):
    theta_i = Tp_bench_mark[i - 1][0]
    tp_i    = Tp_bench_mark[i - 1][1]
    tm_i    = Tm_bench_mark[i - 1][1]

    fname_p = outdir / f"Tp_benchmark_{args.tmin_mm}_{args.tmax_mm}_mode_{i:02d}.txt"
    fname_m = outdir / f"Tm_benchmark_{args.tmin_mm}_{args.tmax_mm}_mode_{i:02d}.txt"

    with open(fname_p, "w", encoding="ascii") as f:
        f.write(param_header)
        f.write("# theta_arcmin Tp_benchmark\n")
        for th, val in zip(theta_grid_arcmin, tp_i):
            f.write(f"{mp.nstr(mp.mpf(th), n=n_digits)} {mp.nstr(mp.mpf(val), n=n_digits)}\n")

    with open(fname_m, "w", encoding="ascii") as f:
        f.write(param_header)
        f.write("# theta_arcmin Tm_benchmark\n")
        for th, val in zip(theta_grid_arcmin, tm_i):
            f.write(f"{mp.nstr(mp.mpf(th), n=n_digits)} {mp.nstr(mp.mpf(val), n=n_digits)}\n")


# ---------------------------------------------------------------------------
# Compute W(ell) and save to disk
# ---------------------------------------------------------------------------

import time
for i in range(1, Nmax_mm + 1):
    n = i - 1
    print(f"Computing W(ell) mode {i}/{Nmax_mm} ...")
    theta_i = np.array([float(x) for x in Tp_bench_mark[n][0]], dtype=np.float64)
    tp_i    = np.array([float(x) for x in Tp_bench_mark[n][1]], dtype=np.float64)
    
    t0 = time.time()
    Well_i = get_W_ell(ell, theta_i, tp_i[:, None], n_jobs=num_cores)
    t1 = time.time()
    print(f"Time taken for mode {i}: {t1 - t0:.2f} seconds")
    
    fname = outdir / f"high_acc_Well_{args.tmin_mm}_{args.tmax_mm}_mode_{i:02d}.txt"
    np.savetxt(fname, np.column_stack([ell, 2*np.array(Well_i)]),
               header=param_header.lstrip("# ").rstrip("\n") + "\nell Well", comments="# ")
