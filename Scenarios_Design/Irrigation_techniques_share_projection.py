import csv
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as scio

# — CONFIGURATION —
BASE_DIR      = Path(r"D:\ISIMIP\For_generation")
COUNTRY_IDX   = Path(r"C:\Research2\surf_data\country_index.csv")
SOCIO_CODES   = Path(r"C:\Research2\Socioeconomic\Socioeconomic_codes.csv")
COUNTRY_DIR   = Path(r"C:\Research2\ISIMIP\countries")
SCENARIOS     = ["126","370","585"]
TIMESTEPS     = 19  # 0…18 → 19
PIXEL_DIM     = (720,360)

# — HELPERS —

def load_csv_dict(path: Path, key_col=0, val_col=1, encoding="ANSI"):
    """Load a two-column CSV into a {key: val} dict of strings."""
    with path.open(encoding=encoding) as f:
        reader = csv.reader(f)
        return {row[key_col]: row[val_col] for row in reader}

def load_mat(path: Path, var: str) -> np.ndarray:
    """Load a single variable from a .mat file."""
    return scio.loadmat(path)[var]

def read_pixel_list(country_idx: int) -> np.ndarray:
    """Load country_i.csv as an array of (x,y) pairs."""
    p = COUNTRY_DIR / f"{country_idx}.csv"
    if not p.exists():
        return np.empty((0,2), int)
    df = pd.read_csv(p, header=None, dtype=int)
    # convert to zero-based
    return df.values - 1

def fill_nan_neighbor(arr: np.ndarray, x:int, y:int) -> tuple:
    """If arr[x,y] is nan, pick the first non-nan from its 4-neighborhood, else return arr[x,y]."""
    if not np.isnan(arr[x,y]):
        return arr[x,y]
    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
        i,j = x+dx, y+dy
        if 0 <= i < arr.shape[0] and 0 <= j < arr.shape[1] and not np.isnan(arr[i,j]):
            return arr[i,j]
    # fallback
    return 1.0

def compute_speed(socio_val: float) -> float:
    """Map socio-economic index to a percent-per-timestep speed adjustment."""
    if socio_val < 0.5:    delta = -0.8
    elif socio_val < 0.75: delta = -0.4
    elif socio_val < 1.0:  delta =  0.0
    elif socio_val < 1.25: delta =  0.4
    else:                  delta =  0.8
    return (1 + delta) / 100.0

# — MAIN PROCESSING —

# load global lookup tables once
code_proj_map = load_csv_dict(COUNTRY_IDX)
socio_code_map = load_csv_dict(SOCIO_CODES)

for scenario in SCENARIOS:
    print(f">>> SSP{scenario}")

    # load socioeconomic data: skip first column
    socio_data = pd.read_csv(
        Path(r"C:\Research2\Socioeconomic")/f"Socio_data_SSP{scenario}.csv",
        header=None, encoding="ANSI"
    ).iloc[:,1:].values

    base = BASE_DIR / f"landuse-15crops_image_ssp{scenario}_annual_2015_2100"
    # load all fraction & option arrays
    basic_f = load_mat(base/"act_irr_floo.mat",  "act_irr_floo")
    basic_s = load_mat(base/"act_irr_spri.mat",  "act_irr_spri")
    basic_d = load_mat(base/"act_irr_drip.mat",  "act_irr_drip")
    opt_f   = load_mat(base/"opt_floo.mat",      "opt_floo")
    opt_s   = load_mat(base/"opt_spri.mat",      "opt_spri")
    opt_d   = load_mat(base/"opt_drip.mat",      "opt_drip")
    pct     = load_mat(base/"pct_irr.mat",       "pct_irr")

    # output array: [x, y, time, method]
    grid_frac = np.zeros((*PIXEL_DIM, TIMESTEPS, 3), float)
    grid_frac[:] = np.nan  # initialize to nan

    # iterate countries
    for country in range(1,257):
        pixels = read_pixel_list(country)
        if pixels.size == 0:
            continue

        proj_code = code_proj_map.get(str(country))
        socio_idx = socio_code_map.get(proj_code, None)
        if socio_idx is None:
            continue
        socio_idx = int(socio_idx)

        # T=0: initialize from basic fractions
        for x,y in pixels:
            if pct[x,y,0] == 0:
                continue
            f = fill_nan_neighbor(basic_f, x,y)
            s = fill_nan_neighbor(basic_s, x,y)
            d = fill_nan_neighbor(basic_d, x,y)
            grid_frac[x,y,0] = (f,s,d)

        # subsequent timesteps
        for t in range(1, TIMESTEPS):
            yr = (t-1)*5
            for x,y in pixels:
                if pct[x,y,yr]==0:
                    continue
                f,s,d = grid_frac[x,y,t-1]
                if np.isnan(f):
                    # fallback neighbor from prev timestep
                    arr = grid_frac[...,t-1,0]
                    f = fill_nan_neighbor(arr,x,y)
                    # replicate for s,d
                    s = fill_nan_neighbor(grid_frac[...,t-1,1],x,y)
                    d = fill_nan_neighbor(grid_frac[...,t-1,2],x,y)

                # optimal shares at this time
                of,os,od = opt_f[x,y,yr], opt_s[x,y,yr], opt_d[x,y,yr]
                speed = compute_speed(socio_data[socio_idx, t+1])

                # how to split new capacity
                total_opt = os + od
                if total_opt>0:
                    s += 5*speed*(os/total_opt)
                    d += 5*speed*(od/total_opt)

                # enforce bounds
                d = min(d, od)
                s = min(s, os+od-d)
                f = 1 - s - d

                grid_frac[x,y,t] = (f,s,d)

    # save
    out_file = base/f"grid_frac_ssp{scenario}.mat"
    scio.savemat(str(out_file), {"grid_frac": grid_frac})
    print(f"  ✔  Saved {out_file}")
