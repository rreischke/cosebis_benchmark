import yaml
import os
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import glass
import glass.ext.camb
import camb
import camb.sources
import heracles
from cosmology import Cosmology
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d


def spectra_indices(n):
    i, j = np.tril_indices(n)
    return np.transpose([i, i - j])
    
# Config
#config_path = "./sims_config.yaml"
#with open(config_path, 'r') as f:
#    config = yaml.safe_load(f)
n = 1000
nside = 512
lmax = 1024
mode = "gaussian"  # "lognormal" or "gaussian"
nbins = 6
path = f"./../{mode}_sims"

# Load nzs
z = np.linspace(0.0, 3.0, 3000)
nzs_wl_hdul = fits.open(f"/Users/robert/Documents/Git/cosebis_benchmark/data/nz_example.fits")
nzs = gaussian_filter1d(nzs_wl_hdul[1].data["N_Z"].T, 10, axis=0)

# Load theory cls
cls_dict = heracles.read(f"{path}/theory_camb.fits")
cls = [cls_dict[f"W{i+1}xW{j+1}"].array for i, j in spectra_indices(nbins)]


# Make GLASS cls
shells = [
    glass.RadialWindow(z, nz, np.trapezoid(z * nz, z) / np.trapezoid(nz, z)) for nz in nzs.T
]

# Make fields
fields = glass.lognormal_fields(shells, glass.lognormal_shift_hilbert2011)

# Solve for spectra
gls = glass.solve_gaussian_spectra(fields, cls)
gls = glass.regularized_spectra(gls)

# Check if folder exists
for i in range(1, n+1):
    folname = f"/fullsky/maps/{mode}_sim_{i}_nside_{nside}"
    print(f"Making sim {i} in folder {folname}", end='\r')
    if not os.path.exists(f"{path}/{folname}"):
        os.makedirs(f"{path}/{folname}")
        # Generate maps
        rng = np.random.default_rng(seed=i)
        maps = list(glass.generate(fields, gls, nside, rng=rng))

        SHEs = {}
        SHEs_wb = {}
        for j, _map in enumerate(maps):
            Q1, U1 = glass.shear_from_convergence(_map)
            SHE = np.array([Q1, U1])
            
            # Generate B-modes <-- Cl^{BB} = 0.1 Cl^{EE}
            SHE_wb = np.copy(SHE)
            cl_ee = cls_dict[f"W{j+1}xW{j+1}"]
            blm = hp.sphtfunc.synalm(
                [np.zeros_like(cl_ee), 
                 np.zeros_like(cl_ee), 
                 np.zeros_like(cl_ee), 
                 0.1*cl_ee.array],
            )
            bmap = hp.alm2map_spin(
                [np.zeros_like(blm[2, :]), blm[2, :]],
                nside, 2, lmax)
            SHE_wb += bmap
    
            heracles.update_metadata(SHE,
                                    nside=nside,
                                    fsky=1.0,
                                    spin=2)
            heracles.update_metadata(SHE_wb,
                                    nside=nside, 
                                    fsky=1.0,
                                    spin=2)

            SHEs["SHE", j+1] = SHE
            SHEs_wb["SHE", j+1] = SHE_wb
    
            
        # Write maps
        filename = f"SHE.fits"
        heracles.write_maps(f"{path}/{folname}/{filename}", SHEs, clobber=True)

        filename = f"SHE_wb.fits"
        heracles.write_maps(f"{path}/{folname}/{filename}", SHEs_wb, clobber=True)
