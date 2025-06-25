%matplotlib inline
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import numpy as np
from Load_data import Data_from_nc
import xarray as xr
import shapefile

def get_data_from_mat(file, variable):
    var = scio.loadmat(file)[variable]
    var[var <= 0.00] = np.nan
    return (
        var[:, 29:, 9].T, var[:, 29:, 34].T, 
        var[:, 29:, 69].T, var[:, 29:, 99].T
    )

# Load data
ssp_paths = {
    "ssp126": "/dodrio/scratch/projects/2022_200/project_input/cesm/yi_yao_future/pct_cft_files/ssp126/irri_land.mat",
    "ssp370": "/dodrio/scratch/projects/2022_200/project_input/cesm/yi_yao_future/pct_cft_files/ssp370/irri_land.mat",
    "ssp585": "/dodrio/scratch/projects/2022_200/project_input/cesm/yi_yao_future/pct_cft_files/ssp585/irri_land.mat",
}
ssp_data = {k: get_data_from_mat(v, "irri_land") for k, v in ssp_paths.items()}

# Load IPCC shapes and extract specific regions
IPCC_shapes = shapefile.Reader("IPCC-WGI-reference-regions-v4.shp")
IPCC_border = IPCC_shapes.shapes()

region_indices = {
    "CNA": 4, "EAS": 35, "WNA": 3, "ARP": 36, "SWS": 13,
    "MED": 19, "SAS": 37, "SEA": 38, "WCA": 32,
    "SAH": 20, "WAF": 21, "WSAF": 25, "EASF": 26
}

region_coords = {
    name: list(zip(*IPCC_border[idx].points)) for name, idx in region_indices.items()
}

# Generalized plot function
def plot_irrigation(ax, data_xarray, title, label, cmap_name, levels):
    cmap = mpl.cm.get_cmap(cmap_name)
    colors = ['white'] + [cmap(0.1 * i) for i in range(1, 10)]
    listed_cmap = mpl.colors.ListedColormap(colors)

    im = data_xarray.plot(ax=ax, cmap=listed_cmap, vmin=0, vmax=0.05,
                          levels=levels, extend='both', add_colorbar=False, add_labels=False)
    plt.colorbar(im, fraction=0.3, pad=0.04, aspect=50, orientation='horizontal')

    ax.set_title(title, loc='right', fontsize=10)
    ax.set_title(label, loc='left', fontsize=10)
    ax.add_feature(cfeature.OCEAN, color='whitesmoke')
    ax.coastlines(linewidth=0.5)

    for name, (xs, ys) in region_coords.items():
        style = '-' if name in ['CNA', 'EAS', 'WNA'] else '--' if name in ['MED', 'SAS', 'SEA', 'WCA', 'ARP'] else ':'
        ax.plot(xs, ys, color='black', linewidth=1, linestyle=style, alpha=0.5)


# Coordinates for xarray
data_surface = Data_from_nc('/dodrio/scratch/projects/2022_200/project_output/cesm/yi_yao_IRRMIP/surfdata_irrigation_method.nc')
data_lon = data_surface.load_variable('LONGXY')[0, :]
data_lat = data_surface.load_variable('LATIXY')[29:, 0]

# Levels and colormap
levels = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
cmap_name = 'YlOrBr'

# SSP and year combinations
scenarios = ['ssp126', 'ssp370']
years = ['2010', '2035', '2070', '2100']
subplot_labels = list('abcdefgh')

# Create figure and subplots
fig = plt.figure(figsize=(12, 16), dpi=300)
fig.subplots_adjust(hspace=0.0, wspace=0.1, left=0.05, right=0.95, top=0.95, bottom=0.05)

proj = ccrs.PlateCarree()
for i, (ssp, year) in enumerate([(s, y) for s in scenarios for y in years]):
    ax = plt.subplot(4, 2, i + 1, projection=proj, frameon=True)

    # Index of the year in your 4 outputs
    year_index = years.index(year)
    data_np = ssp_data[ssp][year_index]

    data_xr = xr.DataArray(data_np, coords={'y': data_lat, 'x': data_lon}, dims=["y", "x"])
    
    title = 'irrigated fraction'
    label = f'{year} ({ssp.upper().replace("SSP", "SSP")})' if year != '2010' else '2010'

    plot_irrigation(ax, data_xr, title, label, cmap_name, levels)
    ax.text(0.01, 0.92, subplot_labels[i], color='dimgrey', fontsize=10, transform=ax.transAxes, weight='bold')

plt.show()