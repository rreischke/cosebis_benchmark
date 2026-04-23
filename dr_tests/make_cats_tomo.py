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

# Note: We do not model source clustering
# This makes comparing to predictions easier. 
# B-modes shouldn't care about this either.
# However, it migth be a problem in the future. 
    
# Config
config_path = "./sims_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
n = 100 #config['nsims']
nside = 1024 #config['nside']
lmax = 3000 #config['lmax_partial']
mode = "lognormal" #config['mode']  # "lognormal" or "gaussian"
path = f"/pscratch/sd/j/jaimerz/{mode}_sims"
mask_type = "tr1"

# vamp
if mask_type != "fullsky":
    path_mask = f"/pscratch/sd/j/jaimerz/masks/{mask_type}_mask_nside_{nside}.fits"
    mask = hp.read_map(path_mask)
else:
    mask = np.ones(hp.nside2npix(nside))
print("computed mask")
# Add spin information to mask
heracles.core.update_metadata(mask, spin=0)

# Load nzs
z = np.linspace(0.0, 3.0, 3000)
nzs_wl_hdul = fits.open("/pscratch/sd/j/jaimerz/lognormal_sims/TR1_v1_Nz_WL_C2020_sel_pv.fits")
dndz = gaussian_filter1d(nzs_wl_hdul[1].data["N_Z"].T, 10, axis=0)
n_bins = 6

# Config
# distribute dN/dz over the radial window functions
# ngals per sqr minute
ngals_TR1 = {
    1: 5.014,
    2: 4.656,
    3: 4.828,
    4: 4.616,
    5: 4.767,
    6: 4.505,
}

# ellipticity standard deviation as expected for a Stage-IV survey
sigma_e = {
    1: 0.2672282382294292,
    2: 0.2662526016273488,
    3: 0.2590877242553745,
    4: 0.2591418343365571,
    5: 0.2550252629962767,
    6: 0.2641187068627740,
}

#bias 
bias = 1.2

# Check if folder exists
for i in range(1, n+1):
    # Generate maps
    rng = np.random.default_rng(seed=i)
    folname = f"{mode}_sim_{i}_nside_{nside}"
    print(f"Making cat {i} in folder {folname}", end='\r')
    # Generate folder
    if not os.path.exists(f"{path}/{mask_type}/cats/{folname}"):
        os.makedirs(f"{path}/{mask_type}/cats/{folname}")
    sim_path = f"{path}/fullsky/maps/{folname}/SHE.fits"
    sim_wb_path = f"{path}/fullsky/maps/{folname}/SHE_wb.fits"
    cat_path = f"{path}/{mask_type}/cats/{folname}/SHE.fits"  
    if not os.path.exists(cat_path):
        with glass.write_catalog(cat_path) as out:
            for j in range(1, n_bins + 1):
                SHE = heracles.read_maps(sim_path)[("SHE", j)]
                SHE_wb = heracles.read_maps(sim_wb_path)[("SHE", j)]
                Q1, U1 = SHE
                Q1_wb, U1_wb = SHE_wb
                for gal_lon, gal_lat, gal_count in glass.positions_from_delta(
                    ngals_TR1[j],
                    np.zeros_like(Q1),
                    0.0, 
                    mask,
                    rng=rng,
                ):
                    gal_z = glass.redshifts_from_nz(
                        gal_count,
                        z,
                        dndz.T[j - 1], 
                        rng=rng,
                        warn=False,
                    )
    
                    gal_eps = glass.ellipticity_intnorm(
                        gal_count,
                        sigma_e[j],
                        rng=rng,
                    )
    
                    gal_she = glass.galaxy_shear(
                        gal_lon,
                        gal_lat,
                        gal_eps,
                        np.zeros_like(Q1),
                        Q1,
                        U1,
                        reduced_shear=False,
                    )
                    gal_she_wb = glass.galaxy_shear(
                        gal_lon,
                        gal_lat,
                        gal_eps,
                        np.zeros_like(Q1_wb),
                        Q1_wb,
                        U1_wb,
                        reduced_shear=False,
                    )
    
                    out.write(
                        RA=gal_lon,
                        DEC=gal_lat,
                        Z=gal_z,
                        E1=gal_she.real,
                        E2=gal_she.imag,
                        E1_wb=gal_she_wb.real,
                        E2_wb=gal_she_wb.imag,
                        TOMBINID=np.full(gal_count, j, dtype=np.int32),
                    )
