%matplotlib inline
import scipy.io as scio
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import netCDF4 as nc
import numpy as np
from Load_data import Data_from_nc
import xarray as xr
import math
import gc

# --- [ Script content truncated for brevity here ] ---

# Final plotting block
variable_to_show = '$\mathregular{T_{w}}$'
ylim1 = 1700
ylim2 = 1700

fig = plt.figure(figsize=(24, 5), dpi=300)

ax1 = plt.subplot(141, frameon=True)
title = 'a TRA SSP1-2.6'
plot_bar_both_hist_and_ssp1(ax1, title, variable_to_show, perc, ylim1, hist_IRR_tra_TSA, hist_IRR_tra_WBT, ssp1_IRR_tra_TSA, ssp1_IRR_tra_WBT)
plot_bar_both_hist_and_ssp1_noi(ax1, title, variable_to_show, perc, ylim1, hist_NOI_tra_TSA, hist_NOI_tra_WBT, ssp1_NOI_tra_TSA, ssp1_NOI_tra_WBT)
plt.xticks([2000, 2020, 2040, 2060], ['2000', '2020', '2040', '2060'])

ax2 = plt.subplot(142, frameon=True)
title = 'b TRA SSP3-7.0'
plot_bar_both_hist_and_ssp3(ax2, title, variable_to_show, perc, ylim1, hist_IRR_tra_TSA, hist_IRR_tra_WBT, ssp3_IRR_tra_TSA, ssp3_IRR_tra_WBT)
plot_bar_both_hist_and_ssp3_noi(ax2, title, variable_to_show, perc, ylim1, hist_NOI_tra_TSA, hist_NOI_tra_WBT, ssp3_NOI_tra_TSA, ssp3_NOI_tra_WBT)
plt.xticks([2000, 2020, 2040, 2060], ['2000', '2020', '2040', '2060'])

fig.subplots_adjust(hspace=0.4, wspace=0.4, left = 0.1, right = 0.9, top = 0.9, bottom = 0.2)

ax1 = plt.subplot(143, frameon=True)
title = 'c NEW SSP1-2.6'
plot_bar_both_hist_and_ssp1(ax1, title, variable_to_show, perc, ylim2, hist_IRR_new_TSA, hist_IRR_new_WBT, ssp1_IRR_new_TSA, ssp1_IRR_new_WBT)
plot_bar_both_hist_and_ssp1_noi(ax1, title, variable_to_show, perc, ylim2, hist_NOI_new_TSA, hist_NOI_new_WBT, ssp1_NOI_new_TSA, ssp1_NOI_new_WBT)
plt.xticks([2000, 2020, 2040, 2060], ['2000', '2020', '2040', '2060'])

ax2 = plt.subplot(144, frameon=True)
title = 'd NEW SSP3-7.0'
plot_bar_both_hist_and_ssp3(ax2, title, variable_to_show, perc, ylim2, hist_IRR_new_TSA, hist_IRR_new_WBT, ssp3_IRR_new_TSA, ssp3_IRR_new_WBT)
plot_bar_both_hist_and_ssp3_noi(ax2, title, variable_to_show, perc, ylim2, hist_NOI_new_TSA, hist_NOI_new_WBT, ssp3_NOI_new_TSA, ssp3_NOI_new_WBT)
plt.xticks([2000, 2020, 2040, 2060], ['2000', '2020', '2040', '2060'])
fig.subplots_adjust(hspace=0.4, wspace=0.3, left = 0.1, right = 0.9, top = 0.9, bottom = 0.2)