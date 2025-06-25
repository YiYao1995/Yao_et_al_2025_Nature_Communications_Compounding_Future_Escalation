# refactored_exposure.py
%matplotlib inline
import os
import numpy as np
import scipy.io as scio
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import shapefile

# ─── I/O UTILITIES ────────────────────────────────────────────────

def load_mat(file, var, slice1=None, transpose=False):
    arr = scio.loadmat(file)[var]
    if slice1:
        arr = arr[slice1]
    if transpose:
        arr = arr.T
    return arr

def load_nc(path, var, drop_bad_above=1e6, drop_bad_below=None, slice1=None):
    ds = nc.Dataset(path)
    arr = np.array(ds.variables[var])
    if slice1:
        arr = arr[slice1]
    if drop_bad_above is not None:
        arr[arr>drop_bad_above] = np.nan
    if drop_bad_below is not None:
        arr[arr<drop_bad_below] = np.nan
    return np.squeeze(arr)

# ─── REGION BOUNDARIES ────────────────────────────────────────────

IPCC_IDX = {
    'CNA': (4,'-'),
    'EAS': (35,'-'),
    'WNA': (3,'-'),
    'MED': (19,'--'),
    'SAS': (37,'--'),
    'SEA': (38,'--'),
    'WCA': (32,'--'),
    'ARP': (36,'--'),
    'SAH': (20,'-.'),
    'WAF': (21,'-.'),
    'WSAF':(25,'-.'),
    'EASF':(26,'-.'),
    'SWS': (13,'-.'),
}

def load_boundaries(shp='IPCC-WGI-reference-regions-v4.shp'):
    sf = shapefile.Reader(shp)
    shapes = sf.shapes()
    bds = {}
    for name,(idx,style) in IPCC_IDX.items():
        pts = shapes[idx].points
        bds[name] = (np.array(pts).T, style)
    return bds

BOUNDARIES = load_boundaries()

def draw_boundaries(ax):
    for (xs,ys),style in BOUNDARIES.values():
        ax.plot(xs, ys, 'k', ls=style, lw=1, alpha=0.5)

# ─── EXPOSURE CALCULATIONS ────────────────────────────────────────

def ensemble_mean(*arrays):
    return np.nanmean(np.stack(arrays,0),0)

def count_consistent(arrs, threshold):
    """Count grid‐cells where all ensemble‐members differ from baseline by >threshold."""
    diffs = np.stack([a - arrs[0] for a in arrs[1:]],0)
    pos = np.all(diffs> threshold, 0)
    neg = np.all(diffs< -threshold,0)
    return np.where(pos|neg, 1, 0)

def get_exposure(variable, perc, scenario, years, base_scen=None):
    """
    Load and average three ensemble members for given scenario & percentile.
    scenario: e.g. 'BHIST_IRR', 'BSSP126_IRR'
    years: '1985_2014' or '2045_2074'
    """
    arrs = []
    for run in ['01','02','03']:
        path = (
            f"/dodrio/.../archive/{scenario}{run}/lnd/hist/{variable}/"
            f"{scenario}{run}_{years}.nc_{variable}_ge_pct{perc}_hours_timmean"
        )
        arrs.append(load_nc(path, variable, slice1=(slice(None),29,None)))
    return ensemble_mean(*arrs)

def get_all_exposures(variable, perc):
    hist_irr = get_exposure(variable, perc, 'BHIST_IRR_', '1985_2014')
    hist_noi = get_exposure(variable, perc, 'BHIST_NOI_', '1985_2014')
    ssp1_irr = get_exposure(variable, perc, 'BSSP126_IRR', '2045_2074')
    ssp3_irr = get_exposure(variable, perc, 'BSSP370_IRR', '2045_2074')
    ssp1_noi = get_exposure(variable, perc, 'BSSP126_NOI', '2045_2074')
    ssp3_noi = get_exposure(variable, perc, 'BSSP370_NOI', '2045_2074')

    # consistency counts
    cons_ssp1_all = count_consistent(
        [hist_irr,
         load_nc(f".../BSSP126_IRR01_2045_2074.nc...",variable),
         load_nc(f".../BSSP126_IRR02_2045_2074.nc...",variable),
         load_nc(f".../BSSP126_IRR03_2045_2074.nc...",variable),
        ], threshold=10)
    cons_ssp3_all = count_consistent(
        [hist_irr, ssp3_irr], threshold=10)

    return {
        'hist':(hist_irr, hist_noi),
        'ssp1':(ssp1_irr, ssp1_noi, cons_ssp1_all),
        'ssp3':(ssp3_irr, ssp3_noi, cons_ssp3_all),
    }

# ─── PLOTTING ─────────────────────────────────────────────────────

def plot_map(ax, data, title, left_title, cmap, levels, hatch=None, bounds=None):
    im = data.plot(
        ax=ax, transform=ccrs.PlateCarree(),
        cmap=cmap, levels=levels, add_colorbar=False
    )
    cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                      fraction=0.03, pad=0.04, aspect=30)
    cb.set_label(left_title, fontsize=12)
    ax.set_title(title, loc='right', fontsize=12)
    ax.coastlines('black',0.5)
    ax.add_feature(cfeature.OCEAN, color='whitesmoke')
    if hatch is not None:
        ax.contourf(
            data['x'], data['y'], hatch,
            levels=bounds, colors='none', transform=ccrs.PlateCarree(),
            hatches=['////']
        )
    draw_boundaries(ax)

def plot_for_paper(percs, variables, var_labels):
    # prepare lat/lon
    surf = Data_from_nc(".../surfdata_irrigation_method.nc")
    lon = surf.load_variable('LONGXY')[0]; lat = surf.load_variable('LATIXY')[:,0][29:]
    for perc, var, lbl in zip(percs, variables, var_labels):
        expo = get_all_exposures(var, perc)
        fig,axs = plt.subplots(3,1, figsize=(6,12),
                               subplot_kw={'projection':ccrs.PlateCarree()})
        fig.subplots_adjust(hspace=0.1)
        # SSP1 minus Hist, SSP3 minus Hist, SSP3 minus SSP1
        diffs = [
            expo['ssp1'][0] - expo['hist'][0],
            expo['ssp3'][0] - expo['hist'][0],
            expo['ssp3'][0] - expo['ssp1'][0],
        ]
        hatches = [expo['ssp1'][2], expo['ssp3'][2], None]
        titles = [f"SSP1–Irr – Hist–Irr",
                  f"SSP3–Irr – Hist–Irr",
                  f"SSP3–Irr – SSP1–Irr"]
        cmaps = ['Reds','Reds','coolwarm']
        levels_list = [
            np.linspace(0,500,9),
            np.linspace(0,800,9),
            np.linspace(-200,200,9)
        ]
        for ax,diff,hatch,title,cmap,levels in zip(
            axs, diffs, hatches, titles, cmaps, levels_list):
            da = xr.DataArray(diff, coords={'y':lat,'x':lon}, dims=['y','x'])
            plot_map(ax, da, title, f"Δ hours exposure of {lbl}", cmap, levels,
                     hatch=hatch, bounds=[0.5,1.5])

        plt.suptitle(f"{lbl} {perc}th percentile changes", y=0.93, size=14)
        plt.show()

# ─── RUN ───────────────────────────────────────────────────────────

if __name__=="__main__":
    plot_for_paper([99,99], ['TSA','WBT'], ['$T_{2m}$','$T_w$'])
