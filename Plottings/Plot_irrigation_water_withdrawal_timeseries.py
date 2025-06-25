import scipy.io as scio
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import netCDF4 as nc
from Load_data import Data_from_nc

# --- Configuration ---
VARIABLE = 'QIRRIG_FROM_SURFACE'
DISPLAY_NAME = 'IWW'
UNIT = 'km^3/year'
AREA_FILE = '/dodrio/scratch/projects/2022_200/project_output/cesm/yi_yao_IRRMIP/input_data/AREA.mat'
REGION_FILE = '/dodrio/scratch/projects/2022_200/project_output/cesm/yi_yao_IRRMIP/ar6_region_all.mat'
SURFDATA_FILE = '/dodrio/scratch/projects/2022_200/project_output/cesm/yi_yao_IRRMIP/surfdata_irrigation_method.nc'

REGION_GROUPS = [
    {
        'id': 'd', 'name': 'Global', 'regions': [], 'ylim': (1500, 2600), 'func': 'bar'
    },
    {
        'id': 'e', 'name': 'Group 1', 'regions': [5,36,4], 'ylim': (25+150+25+25, 600), 'func': 'region_no_xlabel'
    },
    {
        'id': 'f', 'name': 'Group 2', 'regions': [20,33,38,39,37], 'ylim': (100+150+500+50, 1600), 'func': 'region_no_xlabel'
    },
    {
        'id': 'g', 'name': 'Group 3', 'regions': [21,22,26,27,14], 'ylim': (0, 600), 'func': 'region'
    },
]

# --- Data Loaders ---

def get_data_nc(path, var):
    ds = nc.Dataset(path)
    arr = np.array(ds.variables[var])
    arr = arr[:,29:,:]
    arr[(arr>1e5)|(arr<-1e3)] = np.nan
    return np.squeeze(arr)


def load_mat(path, var, trim=True, for_calc=False):
    d = scio.loadmat(path)[var]
    if trim:
        d = d[:,29:]
    if for_calc:
        d = d.T
    return d

# --- Processing ---

AREA = load_mat(AREA_FILE, 'AREA', trim=True)
AR6 = load_mat(REGION_FILE, 'ar6_region', trim=True, for_calc=True)

# IWW extraction
def get_all_IWW(var):
    prefix_hist = '/dodrio/.../BHIST_IRR_0'
    mid_hist = '_restart/lnd/hist/'
    end_hist = '_yearmean'
    # helper to build path
    def p(base,id0,mid,id1,sfx): return f"{base}{id0}{mid}{id1}{sfx}"
    # fetch arrays
    h1 = get_data_nc(p(prefix_hist,1,"QIRRIG/BHIST_IRR_0",1,'_1985_2014.nc_'+var+'_yearmean'), var)
    h2 = get_data_nc(p(prefix_hist,2,mid_hist,var+'/BHIST_IRR_0',2,'_1985_2014.nc_'+var+'_yearmean'), var)
    h3 = get_data_nc(p(prefix_hist,3,mid_hist,var+'/BHIST_IRR_0',3,'_1985_2014.nc_'+var+'_yearmean'), var)
    s1 = get_data_nc(p('/dodrio/.../BSSP126_IRR0',1,mid_hist,var+'/BSSP126_IRR0',1,'_2015_2074.nc_'+var+'_yearmean'), var)
    s2 = get_data_nc(p('/dodrio/.../BSSP126_IRR0',2,mid_hist,var+'/BSSP126_IRR0',2,'_2015_2074.nc_'+var+'_yearmean'), var)
    s3 = get_data_nc(p('/dodrio/.../BSSP126_IRR0',3,mid_hist,var+'/BSSP126_IRR0',3,'_2015_2074.nc_'+var+'_yearmean'), var)
    t1 = get_data_nc(p('/dodrio/.../BSSP370_IRR0',1,mid_hist,var+'/BSSP370_IRR0',1,'_2015_2074.nc_'+var+'_yearmean'), var)
    t2 = get_data_nc(p('/dodrio/.../BSSP370_IRR0',2,mid_hist,var+'/BSSP370_IRR0',2,'_2015_2074.nc_'+var+'_yearmean'), var)
    t3 = get_data_nc(p('/dodrio/.../BSSP370_IRR0',3,mid_hist,var+'/BSSP370_IRR0',3,'_2015_2074.nc_'+var+'_yearmean'), var)
    return (h1,h2,h3), (s1,s2,s3), (t1,t2,t3)


def calc_global_IWW(arrays, area):
    # arrays: list of 3 simulations
    stacked = []
    for arr in arrays:
        iww = arr * area.T / 1e6 * 365*24*3600
        tot = np.nansum(np.nansum(iww,axis=1),axis=1)
        stacked.append(tot)
    return np.vstack(stacked)


def calc_region_IWW(arrays, area, region_id):
    arrs = [np.where(AR6==region_id, a, np.nan) for a in arrays]
    return calc_global_IWW(arrs, area)

# --- Plotting ---

def plot_bar(ax, yrs, data, style, **kwargs):
    if style=='scatter':
        ax.scatter([yrs[0]], np.median(data,axis=0), **kwargs)
    else:
        ax.plot(yrs, np.mean(data,axis=0), **kwargs)

# Assemble and draw
(h_arr,s_arr,t_arr) = get_all_IWW(VARIABLE)
hist = calc_global_IWW(h_arr, AREA)
ssp1 = calc_global_IWW(s_arr, AREA)
ssp3 = calc_global_IWW(t_arr, AREA)

years_hist = np.arange(1990,2000)
years_proj = np.arange(2015,2075)

fig, axs = plt.subplots(4,1, figsize=(3,12), dpi=300)
for ax, group in zip(axs, REGION_GROUPS):
    ax.set_title(group['name'])
    ax.set_ylim(group['ylim'])
    ax.set_xlim(1990,2075)
    if group['name']=='Global':
        plot_bar(ax, [2000], hist[:,:-1].flatten(), 'scatter', color='k', label='Hist')
        plot_bar(ax, [], savgol_filter(ssp1[:,:-1],15,2), 'plot', color='g', label='SSP1')
        plot_bar(ax, [], savgol_filter(ssp3[:,:-1],15,2), 'plot', color='brown', label='SSP3')
    else:
        accum = np.zeros_like(hist)
        accum1= np.zeros_like(ssp1)
        accum3= np.zeros_like(ssp3)
        for rid in group['regions']:
            accum += calc_region_IWW(h_arr, AREA, rid)
            accum1+=calc_region_IWW(s_arr, AREA, rid)
            accum3+=calc_region_IWW(t_arr, AREA, rid)
        plot_bar(ax, [2000], accum[:,:-1].flatten(), 'scatter', color='k')
        plot_bar(ax, [], savgol_filter(accum1[:,:-1],15,2), 'plot', color='g')
        plot_bar(ax, [], savgol_filter(accum3[:,:-1],15,2), 'plot', color='brown')
    ax.legend()
plt.tight_layout()
plt.show()
