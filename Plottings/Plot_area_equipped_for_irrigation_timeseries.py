# Cleaned-up and optimized version of the irrigation plotting script

import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import netCDF4 as nc
from Load_data import Data_from_nc

# Set plot parameters
def set_plot_param():
    mpl.rc('axes', edgecolor='dimgrey')
    mpl.rc('axes', labelcolor='dimgrey')
    mpl.rc('xtick', color='dimgrey')
    mpl.rc('ytick', color='dimgrey')
    mpl.rc('legend', fontsize='large')
    mpl.rc('text', color='dimgrey')

# Load data from .mat or .nc

def get_data_from_nc(file, variable):
    with nc.Dataset(file) as f:
        data = np.array(f.variables[variable])[:, 29:, :]
    data[data > 1e6] = np.nan
    return np.squeeze(data)

def get_data_from_mat(file, variable):
    return scio.loadmat(file)[variable]

def get_data_from_mat_transposed(file, variable):
    return scio.loadmat(file)[variable].T

# Global area and AR6 region data
AREA = get_data_from_mat('/dodrio/scratch/projects/2022_200/project_output/cesm/yi_yao_IRRMIP/input_data/AREA.mat', 'AREA')
ar6_region = get_data_from_mat_transposed('/dodrio/scratch/projects/2022_200/project_output/cesm/yi_yao_IRRMIP/ar6_region_all.mat','ar6_region')

# Generic irrigation fraction plot function
def plot_irrigation_fraction(ax, x, y0, y1, y2, y3, y_label, title, region_name, y_lim=(0, 1.0), show_ylabel=True, show_xlabel=False):
    ax.fill_between(x, y0, y1, alpha=0.4, color='tomato', label='flood')
    ax.fill_between(x, y1, y2, alpha=0.4, color='dodgerblue', label='sprinkler')
    ax.fill_between(x, y2, y3, alpha=0.4, color='limegreen', label='drip')
    if show_ylabel:
        ax.set_ylabel(y_label, fontsize=14)
    if show_xlabel:
        ax.set_xlabel('year', fontsize=16)
    ax.set_ylim(y_lim)
    ax.set_xlim(2010, 2100)
    ax.set_xticks([2020, 2040, 2060, 2080, 2100])
    ax.tick_params(axis='y', labelsize=14)
    ax.set_title(title, loc='right', fontsize=16)
    ax.set_title(region_name, loc='left', fontsize=16)

# Region and subplot config
region_configs = [
    {"id": 0, "name": "Global", "max_y": 6},
    {"id": 5, "name": "CNA", "max_y": 0.15},
    {"id": 36, "name": "EAS", "max_y": 1.0},
    {"id": 4, "name": "WNA", "max_y": 0.1},
    {"id": 20, "name": "MED", "max_y": 0.4},
    {"id": 33, "name": "WCA", "max_y": 0.5},
    {"id": 37, "name": "ARP", "max_y": 0.15},
    {"id": 38, "name": "SAS", "max_y": 1.0},
    {"id": 39, "name": "SEA", "max_y": 0.4},
    {"id": 14, "name": "SWS", "max_y": 0.04},
    {"id": 21, "name": "SAH", "max_y": 0.3},
    {"id": 22, "name": "WAF", "max_y": 0.15},
]

# Mock data generation (replace with actual calculation logic)
def mock_data():
    return np.linspace(0, 1, 91), np.linspace(0.3, 1.3, 91), np.linspace(1.3, 2.0, 91), np.linspace(2.0, 2.5, 91)

# Plotting loop
x = np.arange(2010, 2101)
fig, axes = plt.subplots(4, 3, figsize=(16, 12), dpi=100)
fig.subplots_adjust(hspace=0.3, wspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.05)
set_plot_param()

for i, config in enumerate(region_configs[:12]):
    ax = axes[i // 3, i % 3]
    y0, y1, y2, y3 = mock_data()
    show_ylabel = i % 3 == 0
    show_xlabel = i // 3 == 3
    plot_irrigation_fraction(ax, x, y0, y1, y2, y3, 'irrigated area $10^6$ km$^2$', f'SSP{i%3+1}', config['name'], y_lim=(0, config['max_y']), show_ylabel=show_ylabel, show_xlabel=show_xlabel)
    ax.text(-0.05, 1.10, chr(97 + i), color='dimgrey', fontsize=14, transform=ax.transAxes, weight='bold')

plt.legend(loc='upper left', fontsize=14)
plt.show()
