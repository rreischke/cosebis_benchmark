# cosebis_benchmark

High-precision benchmark computation of COSEBIs (Complete Orthogonal Sets of
E/B-Integrals) filter functions following
[Schneider et al. (2010)](https://www.aanda.org/articles/aa/abs/2010/07/aa13875-09/aa13875-09.html).

The code computes the real-space filters T_+(θ) and T_-(θ) using
arbitrary-precision arithmetic ([mpmath](https://mpmath.org/)), and their
Fourier-space counterparts W(ℓ) via the Levin integration method
(`levin`, part of [OneCovariance](https://github.com/rreischke/OneCovariance)).


## Installation

### 1. Install `levin` (required, not on PyPI)

`levin` is a compiled C++ extension shipped as part of the
[OneCovariance](https://github.com/rreischke/OneCovariance) package.
Build and install it from source:

```bash
git clone https://github.com/rreischke/OneCovariance.git
cd OneCovariance
pip install .
```

This requires `gsl` (GNU Scientific Library) and a C++14-capable compiler.
On macOS with Homebrew: `brew install gsl`.
On Debian/Ubuntu: `sudo apt install libgsl-dev`.

### 2. Install this package

```bash
git clone https://github.com/rreischke/cosebis_benchmark
cd cosebis_benchmark
pip install -e .
```

### Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `numpy` | Array operations and file I/O | PyPI |
| `mpmath` | Arbitrary-precision arithmetic for T_+/T_- | PyPI |
| `scipy` | Simpson integration and special functions | PyPI |
| `joblib` | Parallel W(ℓ) computation across ℓ values | PyPI |
| `tqdm` | Progress bar for W(ℓ) computation | PyPI |
| `levin` | Levin quadrature kernel for W(ℓ) | [OneCovariance](https://github.com/rreischke/OneCovariance) |
| `matplotlib` | Plotting (notebook only) | PyPI |

## Usage

### Computing the filters

```bash
python src/get_cosebis.py [OPTIONS]
```

All parameters have sensible defaults and can be overridden from the command line:

```
--tmin_mm      Minimum angular scale [arcmin]       (default: 2)
--tmax_mm      Maximum angular scale [arcmin]       (default: 300)
--Nmax_mm      Number of COSEBIS modes to compute   (default: 20)
--dps          mpmath decimal places of precision   (default: 200)
--N_theta      Number of theta sampling points      (default: 10000)
--N_ell        Number of ell sampling points        (default: 100000)
--ell_min      Minimum multipole                    (default: 1)
--ell_max      Maximum multipole                    (default: 1e5)
--num_cores    Recorded in output headers (unused at runtime; W(ℓ) uses all available cores via joblib) (default: 4)
--outdir_path  Output directory                     (default: ./../benchmarks_cosebis)
```

**Examples:**

```bash
# Default run (20 modes, tmin=2', tmax=300', 200 d.p.)
python src/get_cosebis.py

# Quick low-precision test
python src/get_cosebis.py --dps 50 --N_theta 1000 --N_ell 1000 --Nmax_mm 3
```

### Output files

For each mode n = 1 … Nmax the following ASCII files are written to `--outdir_path`:

| File | Columns | Description |
|------|---------|-------------|
| `Tp_benchmark_{tmin}_{tmax}_mode_{n:02d}.txt` | `theta_arcmin`, `Tp_benchmark` | T_+(θ) in arcmin |
| `Tm_benchmark_{tmin}_{tmax}_mode_{n:02d}.txt` | `theta_arcmin`, `Tm_benchmark` | T_-(θ) in arcmin |
| `Well_{tmin}_{tmax}_mode_{n:02d}.txt`         | `ell`, `Well`                  | W(ℓ) |

Every file starts with a header line recording all input parameters

```
# tmin_mm=2.0 tmax_mm=300.0 Nmax_mm=20 dps=200 N_theta=10000 ...
# theta_rad Tp_bench_mark
```

### Plotting

Open and run `plots/make_plots.ipynb`.  Set `tmin` and `tmax` at the top of
the data-loading cell to match the values used during computation.  Two PDF
figures are saved into the `plots/` directory:

- `Tp_Tm_benchmarks_tmin{tmin}_tmax{tmax}.pdf` — T_+(θ) and T_-(θ) per mode
- `Well_benchmarks_tmin{tmin}_tmax{tmax}.pdf`  — ℓ W(ℓ) per mode
