%matplotlib inline
import numpy as np
import scipy.io as scio
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import shapefile

# ─── I/O & ENSEMBLE HELPERS ───────────────────────────────────────

def load_nc(path, var, slice1=(slice(None),29,None)):
    """Load a netCDF variable, drop bogus values, slice off the first 29 rows."""
    ds = nc.Dataset(path)
    arr = np.array(ds.variables[var])[slice1]
    arr[arr > 1e6] = np.nan
    return np.squeeze(arr)

def ensemble_average(pattern, var, runs=('01','02','03'), years='1985_2014'):
    """
    Given a filename pattern with placeholders {run} and {years}, load each run
    and return the ensemble mean.
    """
    members = []
    for r in runs:
        f = pattern.format(run=r, years=years, var=var)
        members.append(load_nc(f, var))
    return np.nanmean(members,0)

# ─── BOUNDARY DRAWING ─────────────────────────────────────────────

def load_boundaries(shp='IPCC-WGI-reference-regions-v4.shp'):
    sf = shapefile.Reader(shp)
    shapes = sf.shapes()
    # map region‐index → linestyle
    style = {4:'-',35:'-',3:'-',
             19:'--',37:'--',38:'--',32:'--',36:'--',
             20:':',21:':',25:':',26:':',13:':'}
    bds = []
    for idx,ls in style.items():
        pts = np.array(shapes[idx].points).T
        bds.append((pts,ls))
    return bds

BOUNDARIES = load_boundaries()

def draw_boundaries(ax):
    for (xs,ys),ls in BOUNDARIES:
        ax.plot(xs, ys, 'k', ls=ls, lw=1, alpha=0.5)

# ─── PLOTTING UTILITY ────────────────────────────────────────────

def plot_map(ax, data, cmap, levels, title, subtitle, cbar_label, hatch=None):
    """
    A one‐stop function to plot a 2D DataArray, draw coastlines, colorbar,
    and optionally hatches.
    """
    im = data.plot(ax=ax, transform=ccrs.PlateCarree(),
                   cmap=cmap, levels=levels, add_colorbar=False,
                   add_labels=False)
    cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                      fraction=0.03, pad=0.04, aspect=30)
    cb.set_label(cbar_label, fontsize=12)
    ax.coastlines('black',0.5)
    ax.add_feature(cfeature.OCEAN, color='whitesmoke')
    ax.set_title(subtitle, loc='left', fontsize=12)
    ax.set_title(title, loc='right', fontsize=12)
    if hatch is not None:
        ax.contourf(data['x'], data['y'], hatch,
                    levels=[0.5,1.5], colors='none',
                    transform=ccrs.PlateCarree(),
                    hatches=['////'])
    draw_boundaries(ax)

# ─── MAIN ROUTINE ─────────────────────────────────────────────────

def plot_IWW():
    # file patterns
    hist_pattern = (
      "/dodrio/.../archive/BHIST_IRR_{run}_restart/lnd/hist/{var}/"
      "BHIST_IRR_{run}_restart.clm2.h0.{years}.nc_{var}_timmean"
    )
    ssp1_pattern = (
      "/dodrio/.../archive/BSSP126_IRR{run}_new_modified/lnd/hist/{var}/"
      "BSSP126_IRR{run}_new_modified.clm2.h0.2045_2074.nc_{var}_timmean"
    )
    ssp3_pattern = (
      "/dodrio/.../archive/BSSP370_IRR{run}_new_modified/lnd/hist/{var}/"
      "BSSP370_IRR{run}_new_modified.clm2.h0.2045_2074.nc_{var}_timmean"
    )

    # get the three fields
    hist = ensemble_average(hist_pattern, 'QIRRIG_FROM_SURFACE')
    ssp1 = ensemble_average(ssp1_pattern, 'QIRRIG_FROM_SURFACE', years=None)
    ssp3 = ensemble_average(ssp3_pattern, 'QIRRIG_FROM_SURFACE', years=None)

    # convert to mm/year
    factor = 24*3600*365
    hist_mm = hist * factor
    ssp1_mm = ssp1 * factor
    ssp3_mm = ssp3 * factor

    # compute differences
    diffs = [
        (hist_mm,    'Hist (1985–2014)'),
        (ssp1_mm,    'SSP1–2.6 (2045–2074)'),
        (ssp1_mm - hist_mm, 'Δ SSP1–Hist'),
        (ssp3_mm - hist_mm, 'Δ SSP3–Hist'),
        (ssp3_mm - ssp1_mm, 'Δ SSP3–SSP1'),
    ]

    # lat/lon from your surfdata
    surf = Data_from_nc(".../surfdata_irrigation_method.nc")
    lon = surf.load_variable('LONGXY')[0]; lat = surf.load_variable('LATIXY')[29:,0]

    # wrap them into DataArrays
    maps = [ xr.DataArray(m, coords={'y':lat,'x':lon}, dims=['y','x'])
             for m,_ in diffs ]

    # plotting parameters for each panel
    panels = [
      dict(cmap='Blues', levels=[10,50,100,200,500,1000], cbar='IWW (mm/year)'),
      dict(cmap='Blues', levels=[10,50,100,200,500,1000], cbar='IWW (mm/year)'),
      dict(cmap='PRGn', levels=[-100,-50,-20,-5,5,20,50,100], cbar='Δ IWW (mm/yr)'),
      dict(cmap='PRGn', levels=[-100,-50,-20,-5,5,20,50,100], cbar='Δ IWW (mm/yr)'),
      dict(cmap='PRGn', levels=[-100,-50,-20,-5,5,20,50,100], cbar='Δ IWW (mm/yr)'),
    ]

    fig, axs = plt.subplots(5,1, figsize=(6,15),
                             subplot_kw={'projection':ccrs.PlateCarree()})
    fig.subplots_adjust(hspace=0.2)

    for ax, da, (m,sub), params in zip(axs, maps, diffs, panels):
        plot_map(ax, da, params['cmap'], params['levels'],
                 title=m, subtitle=sub, cbar_label=params['cbar'])

    plt.show()

if __name__=='__main__':
    plot_IWW()
