import numpy as np
import xarray as xr
from pathlib import Path
from netCDF4 import Dataset

def fill_nan_nearest(arr):
    """
    Fill NaNs in a 2D array by nearest non-NaN neighbor (4-connectivity).
    """
    from scipy.ndimage import generic_filter
    def replace(x):
        center = x[len(x)//2]
        if np.isnan(center):
            for v in x:
                if not np.isnan(v):
                    return v
        return center
    footprint = np.array([[0,1,0],[1,1,1],[0,1,0]])
    return generic_filter(arr, replace, footprint=footprint, mode='wrap')

# --- 1) LOAD INPUTS ---
base = Path(r"D:\ISIMIP\For_generation\landuse-15crops_image_gfdl-esm4_ssp126_annual_2015_2100")

# 1.1 grid_frac: dims = [time5, y, x, method]
ds = xr.open_dataset(base/"grid_frac_ssp126.mat", engine="matlab")
grid_frac = ds['grid_frac']  # shape (19, 360,720,3)

# 1.2 seasonal pct irrigation
pct = xr.open_dataarray(base/"pct_irr.mat", engine="matlab")          # (time86,y,x)
opt_spri = xr.open_dataarray(base/"opt_spri.mat", engine="matlab").fillna(0)
opt_drip = xr.open_dataarray(base/"opt_drip.mat", engine="matlab").fillna(0)
opt_floo = xr.open_dataarray(base/"opt_floo.mat", engine="matlab").fillna(0)

# --- 2) INTERPOLATE FRACTIONS TO ANNUAL STEP ---
# time5 = 2015,2020,...,2100 → 19 points
# we want annual 2015–2100 → 86 points
year5 = np.linspace(2015,2100,19)
yearA = np.arange(2015,2101)

# Interpolate each method fraction to annual:
methods = ['floo','spri','drip']
frac5 = {m: fill_nan_nearest(grid_frac.sel(method=i).transpose(0,2,1).values)
         for i,m in enumerate(methods)}

fracA = {}
for m in methods:
    # create interp object along time axis:
    arr5 = frac5[m]  # shape (19, y, x)
    # vectorized interp:
    fracA[m] = np.stack([
        np.interp(yr, year5, arr5.reshape(19,-1)).reshape(arr5.shape[1:])
        for yr in yearA], axis=0)  # shape (86, y, x)

# enforce 0≤frac≤opt and frac_sum≤1
opt5 = xr.concat([opt_floo, opt_spri, opt_drip], dim='method').transpose('method','time','y','x').fillna(0).values
optA = np.stack([np.interp(yearA, year5, opt5[i]) for i in range(3)],axis=0)  # (3,86,y,x)

# clamp each
for i,m in enumerate(methods):
    fracA[m] = np.minimum(fracA[m], optA[i])
# ensure spri+drip ≤ opt_spri+opt_drip
sum_sd = fracA['spri']+fracA['drip']
max_sd = optA[1]+optA[2]
mask = sum_sd>max_sd
# reduce spri where needed:
fracA['spri'][mask] = max_sd[mask] - fracA['drip'][mask]
# finally floo = 1−(spri+drip)
fracA['floo'] = 1 - (fracA['spri']+fracA['drip'])

# --- 3) DISTRIBUTE TO CROPS & WRITE NETCDF ---
surf = Dataset(str(base/".nc"), 'r+')  # open for append
dims = ('lon','lat','time')
crop_vars = [
    ('temperate_cereals','spk','spri'),
    ('temperate_cereals','drp','drip'),
    ('temperate_cereals','fld','floo'),
    ('rice','spk','spri'),
    # … add the rest here …
]

for crop, suffix, method in crop_vars:
    var = surf.variables[f"{crop}_irrigated"]  # original
    out = surf.createVariable(f"{crop}_{suffix}_irrigated","f4",dims)
    data = var[:]               # (time,y,x)
    # pct fraction * total irrigated * method share
    share = fracA[method].transpose(1,2,0)  # (y,x,time)
    # original pct map:
    P = pct.transpose('y','x','time').values
    out[:] = np.where(P>0,
                      data * (share/P),
                      0)
    out.long_name = f"{suffix} irrigated {crop}"
    out.units = "1"

surf.close()
print("Done!")
