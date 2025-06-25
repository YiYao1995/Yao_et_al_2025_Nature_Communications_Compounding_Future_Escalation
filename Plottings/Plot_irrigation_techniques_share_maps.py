# Sprinkler and Drip Irrigation Mapping Script (Refactored)

import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from Load_data import Data_from_nc
import shapefile

# Load irrigation fraction from .mat

def get_irrigation_data(file, variable):
    mat = scio.loadmat(file)[variable]
    return {
        '2010_drip': mat[:, 29:, 0, 2].T,
        '2035_drip': mat[:, 29:, 5, 2].T,
        '2070_drip': mat[:, 29:, 12, 2].T,
        '2100_drip': mat[:, 29:, 18, 2].T,
        '2010_spri': mat[:, 29:, 0, 1].T,
        '2035_spri': mat[:, 29:, 5, 1].T,
        '2070_spri': mat[:, 29:, 12, 1].T,
        '2100_spri': mat[:, 29:, 18, 1].T,
    }

ssp126 = get_irrigation_data('/dodrio/scratch/projects/2022_200/project_input/cesm/yi_yao_future/pct_cft_files/ssp126/grid_frac_ssp126.mat', 'grid_frac')
ssp370 = get_irrigation_data('/dodrio/scratch/projects/2022_200/project_input/cesm/yi_yao_future/pct_cft_files/ssp370/grid_frac_ssp370.mat', 'grid_frac')
ssp585 = get_irrigation_data('/dodrio/scratch/projects/2022_200/project_input/cesm/yi_yao_future/pct_cft_files/ssp585/grid_frac_ssp585.mat', 'grid_frac')

# Load lat/lon
surf_data = Data_from_nc('/dodrio/scratch/projects/2022_200/project_output/cesm/yi_yao_IRRMIP/surfdata_irrigation_method.nc')
lon = surf_data.load_variable('LONGXY')[0, :]
lat = surf_data.load_variable('LATIXY')[29:, 0]

# IPCC borders (simplified loader)
def load_ipcc_region(idx):
    shp = shapefile.Reader('IPCC-WGI-reference-regions-v4.shp')
    return zip(*shp.shapes()[idx].points)

regions_idx = {
    "CNA": 4, "EAS": 35, "WNA": 3, "ARP": 36, "SWS": 13, "MED": 19,
    "SAS": 37, "SEA": 38, "WCA": 32, "SAH": 20, "WAF": 21, "WSAF": 25, "EASF": 26
}
region_lines = {k: load_ipcc_region(v) for k, v in regions_idx.items()}

# Plotting

def plot_irrigation(ax, data, title, subtitle, cmap_name, levels):
    cmap = mpl.colors.ListedColormap([mpl.cm.get_cmap(cmap_name)(i) for i in np.linspace(0.1, 1.0, len(levels))])
    da = xr.DataArray(data, coords={'y': lat, 'x': lon}, dims=["y", "x"])
    im = da.plot(ax=ax, cmap=cmap, vmin=0, vmax=0.05, levels=levels, extend='both', add_colorbar=False, add_labels=False)
    cb = plt.colorbar(im, fraction=0.3, pad=0.04, aspect=50, extend='both', orientation='horizontal')
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, color='whitesmoke')
    ax.set_title(title, loc='right', fontsize=10)
    ax.set_title(subtitle, loc='left', fontsize=10)
    for name, (xs, ys) in region_lines.items():
        linestyle = ':' if name in ["SAH", "WAF", "WSAF", "EASF", "SWS"] else '--' if name in ["MED", "SAS", "SEA", "WCA", "ARP"] else '-'
        ax.plot(xs, ys, color='black', linewidth=1, linestyle=linestyle, alpha=0.5)

# Levels and color map
levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cmap_name = 'YlGnBu'

# Subplot grid
fig = plt.figure(figsize=(18, 16), dpi=300)
fig.subplots_adjust(hspace=0.0, wspace=0.1, left=0.05, right=0.95, top=0.95, bottom=0.05)

scenarios = {
    'ssp126': ssp126,
    'ssp370': ssp370
}

years = ['2010', '2035', '2070', '2100']
subplot_positions = [431, 434, 437, 4*3+1, 432, 435, 438, 4*3+2]

for i, (ssp_name, data_dict) in enumerate(scenarios.items()):
    for j, year in enumerate(years):
        pos = subplot_positions[i * 4 + j]
        ax = plt.subplot(pos, projection=ccrs.PlateCarree(), frameon=True)
        sprinkler = data_dict[f"{year}_spri"]
        title = 'sprinkler'
        subtitle = f'{year} ({ssp_name.upper()})' if year != '2010' else year
        plot_irrigation(ax, sprinkler, title, subtitle, cmap_name, levels)
        ax.text(0.01, 0.92, chr(97 + i * 4 + j), color='dimgrey', fontsize=10, transform=ax.transAxes, weight='bold')

plt.show()
