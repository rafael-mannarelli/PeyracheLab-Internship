# Standard library imports
import math
import os
import sys
import time
from itertools import combinations
from multiprocessing import Pool

# Third-party library imports
import matplotlib.pyplot as plt
import nwbmatic as ntm
import numpy as np
import pandas as pd
import pynacollada as pyna
import pynapple as nap
import pywt
import requests
import scipy.io as sc
import seaborn as sns
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, LogNorm, NoNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pycircstat.descriptive import mean as circmean
from scipy import signal, ndimage
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d
import pickle as pickle  
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Local application/library specific imports
import UFOphysio.python as phy
from UFOphysio.python.functions import *
from UFOphysio.python.ufo_detection import *

# Configuration settings
plt.rcParams.update({"axes.spines.right": False, "axes.spines.top": False})
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5)

##############################################################################################
# FUNCTIONS
##############################################################################################
def zscore(x):
    """Return the z-score of the provided array."""
    return (x - x.mean(axis=0)) / x.std(axis=0)

def plot_ccs(ccs,recording,save_fig_path,order_of_use='1'):
    plt.rcParams.update({'font.size': 15})
    lw = 2
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Determine the number of epochs and recordings (assuming 'ccs' is structured consistently)
    nb_epochs = max(len(value.keys()) for value in ccs.values())
    nb_recordings = len(ccs)

    # Base size for each subplot (in inches)
    base_width = 4
    base_height = 6
    figsize_width = nb_epochs * base_width
    figsize_height = nb_recordings * base_height

    plt.figure(figsize=(figsize_width, figsize_height))
    gs = GridSpec(nb_recordings + 1, nb_epochs)

    for i, r in enumerate(ccs.keys()):
        for j, e in enumerate(ccs[r].keys()):
            ax = plt.subplot(gs[i, j])
            if i == 0:
                ax.set_title(e)
            if j == 0:
                ax.set_ylabel(r, rotation=0, labelpad=30)
            
            tmp = ccs[r][e].fillna(0).values
            tmp = (tmp - tmp.mean(0)) / tmp.std(0)
            tmp = tmp[:, np.where(~np.isnan(np.sum(tmp, 0)))[0]]
            im = ax.imshow(tmp.T, aspect='auto', cmap='jet',interpolation='none')
            
            x = ccs[r][e].index.values
            ax.set_xticks([0, len(x)//2, len(x)-1])
            ax.set_xticklabels([f'{x[0]:.3f}', '0.000', f'{x[-1]:.3f}'])
            ax.set_yticks(ticks=range(len(ccs[r][e].keys())),labels=ccs[r][e].keys())
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j, e in enumerate(ccs[next(iter(ccs))].keys()):
        ax = plt.subplot(gs[-1, j])
        for i, r in enumerate(ccs.keys()):
            ax.plot(zscore(ccs[r][e].mean(1)), color=colors[i], linewidth=lw, label=r)
        ax.axvline(0.0, color='k')
        ax.set_xlim([x[0], x[-1]])
        if j == 0:
            ax.legend()
    
    # Adjust tight_layout parameters to control the spacing between subplots
    plt.suptitle(f"Recording: {recording}")
    plt.tight_layout(pad=0.5, h_pad=0.5)
    plt.savefig(save_fig_path+r'\cc_'+ order_of_use +'.svg')
    plt.savefig(save_fig_path+r'\cc_'+ order_of_use +'.png', dpi=300,bbox_inches='tight')
    plt.close()

def parse_ttl_csv(filepath):
    tracking_data=pd.read_csv(filepath, skiprows=6)
    tracking_data=tracking_data.drop('Frame', axis=1)
    tracking_data.columns=['time', 'ry','rx','rz','X','Y','Z']
    tracking_data.iloc[:, 1:4] = (np.radians(tracking_data.iloc[:, 1:4]) % (2 * np.pi))
    position = nap.TsdFrame(t=tracking_data.iloc[:,0].values,
                                    d=tracking_data.iloc[:,1:].values,
                                    time_support=nap.IntervalSet(start=min(tracking_data.iloc[:,0]), 
                                                                end=max(tracking_data.iloc[:,0])),
                                    columns=['ry','rx','rz','x','y','z']) 
    return position

############################################################################################### 
# GENERAL INFOS
###############################################################################################
compute=['CC'] #'HD', 'PC', 'UD', 'CellClass' or 'CC'
ufo_filename='ufo'

data_directory = r'E:\Data PeyracheLab'
datasets = {r'\B0714-230221':['AD', 'MEC'],
           r'\B0703-211129':['AD','MEC'],
           r'\B0702-211111':['AD','MEC']}

fs=20000

SI_thr = {
    'AD':0.0, 
    'LMN':0.0,
    'MMB':0.0,
    'PoSub':0.0,
    'MEC':0.0,
    'TRN':0.0
    }

ufo_shanks_nb = {r'\B0703-211129':[2,0],
                 r'\B0702-211111':[2,0],
                 r'\B0714-230221':[3,0]}

channel_spacing = {r'\B0714-230221':[20,20,20,20,12.5],
            r'\B0703-211129':[20,20,20,20,12.5,12.5,12.5],
            r'\B0702-211111':[20,20,20,20,12.5]}


for r in datasets.keys():
    tmp_datasets=[item for item in datasets[r] if item != 'HIP']
    ccs_long = {s:{e:[] for e in ['wake', 'rem', 'sws']} for s in tmp_datasets}
    ccs_short = {s:{e:[] for e in ['wake', 'rem', 'sws']} for s in tmp_datasets}
    ccs = {s:{e:[] for e in ['up', 'down']} for s in tmp_datasets}
    print('######## '+r+' ########')
    ### Load session data
    path = data_directory + r
    data = ntm.load_session(path, 'neurosuite')
    position = data.position


    sws_ep = data.read_neuroscope_intervals('sws')
    rem_ep = data.read_neuroscope_intervals('rem')

    tmp=np.sort(np.vstack((rem_ep.values,sws_ep)),axis=0)
    data_time=[data.time_support.values[0,0],data.time_support.values[-1,-1]]

    # Calculate wake_ep
    wake_ep = []
    last_end = data_time[0]

    for start, end in tmp:
        if start > last_end:
            wake_ep.append([last_end, start])
        last_end = max(last_end, end)

    if data_time[1] > last_end:
        wake_ep.append([last_end, data_time[1]])

    wake_ep = np.array(wake_ep)
    wake_ep = nap.IntervalSet(start=wake_ep[:,0],end=wake_ep[:,-1], time_units='s') 

    ### For saving figure
    save_fig_path=r'D:\PeyracheLab Internship\Figure'+ r
    os.makedirs(save_fig_path, exist_ok=True)

    ufo_ep, ufo_ts = loadUFOsV2(path,ufo_filename)

    data.load_neurosuite_xml(data.path)
    channels = data.group_to_channel
    nb_channels=data.nChannels

    if 'CC' in compute:
        for s in tmp_datasets:
            print('######## '+s+' ########')
            spikes = data.spikes
            idx = spikes._metadata[spikes._metadata["location"].str.contains(s)].index.values
            spikes = spikes[idx]
############################################################################################### 
# CC COMPUTING
###############################################################################################
            print('CC computing...')

            ufo_ts_start=ufo_ep['start']
            ufo_ts_start=nap.Ts(ufo_ts_start)

            #names = [s.split("/")[-1] + "_" + str(n) for n in spikes.keys()]
            names = spikes.keys()


            for e, ep in zip(['wake', 'rem', 'sws'], [wake_ep, rem_ep, sws_ep]):            
                cc = nap.compute_eventcorrelogram(spikes, ufo_ts_start, 0.001, 0.05, ep, norm=True)
                cc.columns = names
                ccs_long[s][e].append(cc)

                cc = nap.compute_eventcorrelogram(spikes, ufo_ts_start, 0.0001, 0.015, ep, norm=True)
                cc.columns = names
                ccs_short[s][e].append(cc)

            
        for s in ccs_long.keys():
            for e in ccs_long[s].keys():
                ccs_long[s][e] = pd.concat(ccs_long[s][e], axis=1)
                ccs_short[s][e] = pd.concat(ccs_short[s][e], axis=1)

        # Call plotting functions
        plot_ccs(ccs_long,r,save_fig_path,order_of_use='1')
        plot_ccs(ccs_short,r,save_fig_path, order_of_use='2')

        print('########')

############################################################################################### 
# HEAD-DIRECTIONS CELLS COMPUTING
###############################################################################################
    print('HD computing...')
    
    spikes = data.spikes

    angle = position['ry']

    ### Calculate the IQR
    Q1 = np.percentile(angle, 25)
    Q3 = np.percentile(angle, 75)
    IQR = Q3 - Q1

    # Calculate the number of data points
    n = len(angle)

    # Apply the Freedman-Diaconis rule to calculate bin width
    bin_width = 2 * (IQR) / (n ** (1/3))

    # Calculate the number of bins
    range_of_data = np.max(angle.d) - np.min(angle.d)
    num_bins = int(np.round(range_of_data / bin_width))

    if 'HD' in compute:
        for e in range(len(position.time_support)): 
            ttl_ep=nap.IntervalSet(start=position.time_support['start'][e], end=position.time_support['end'][e])

            ### Tuning curves computations
            tuning_curves = nap.compute_1d_tuning_curves(group=spikes, 
                                                        feature=angle, 
                                                        nb_bins=num_bins, 
                                                        ep = ttl_ep,
                                                        minmax=(0, 2 * np.pi))

            for k in spikes:
                tuning_curves[k]=signal.medfilt(tuning_curves[k],3)

            pref_ang = tuning_curves.idxmax()
            norm = Normalize()  # Normalizes data into the range [0,1]
            color = plt.cm.hsv(norm([i / (2 * np.pi) for i in pref_ang.values]))  # Assigns a colour in the HSV colourmap for each value of preferred angle
            color = pd.DataFrame(index=pref_ang.index, data = color, columns = ['r', 'g', 'b', 'a'])

            ### Smooth curves computations
            smoothcurves = smoothAngularTuningCurvesHD(tuning_curves, sigma=3)

            ### Plotting
            spike_groups = spikes.get_info('group')
            # Number of unique groups
            unique_groups = set(spike_groups.values)

            # Generate a color map for the groups
            group_colors = plt.cm.get_cmap('jet', len(unique_groups))  # 'hsv' colormap with as many colors as there are unique groups

            # Map each group to a color
            group_to_color = {group: group_colors(i) for i, group in enumerate(unique_groups)}

            # Convert group colors to a DataFrame for easy lookup
            color = pd.DataFrame([group_to_color[spike_groups[i]] for i in spike_groups.keys()],
                                    index=spike_groups.keys(), columns=['r', 'g', 'b', 'a'])

            num_curves = len(spikes)
            num_columns = 4
            num_rows = math.ceil(num_curves / num_columns)

            # Size per subplot (width, height) in inches
            width_per_subplot = 6
            height_per_subplot = 6

            # Total figure size
            total_width = width_per_subplot * num_columns
            total_height = height_per_subplot * num_rows
            plt.figure(figsize=(total_width, total_height))
            for (i, n), loc in zip(enumerate(spikes.keys()),spikes.get_info('location')):
                plt.subplot(num_rows, num_columns, i + 1, projection='polar')
                plt.plot(smoothcurves[n], color=color.loc[n])
                plt.plot(tuning_curves[n], color='k', alpha=0.7)
                plt.plot([pref_ang[n], pref_ang[n]], [0, max(max(smoothcurves[n]), max(tuning_curves[n]))], 'r--')
                plt.title(loc + '-' + str(n))
                plt.xlabel("Angle (rad)")
                plt.ylabel("Firing Rate (Hz)")
            plt.suptitle(f"Recording: {r} - Wake episodes {e}")
            plt.tight_layout()
            plt.savefig(save_fig_path+r'\hd_wake_ep_'+str(e)+'.png', dpi=300, bbox_inches='tight')
            plt.savefig(save_fig_path+r'\hd_wake_ep_'+str(e)+'.svg', bbox_inches='tight')
            plt.close()

    ### Tuning curves computations
    tuning_curves = nap.compute_1d_tuning_curves(group=spikes, 
                                                feature=angle, 
                                                    nb_bins=num_bins, 
                                                    ep = wake_ep,
                                                    minmax=(0, 2 * np.pi))

    for k in spikes:
        tuning_curves[k]=signal.medfilt(tuning_curves[k],3)
    
    SI = nap.compute_1d_mutual_info(tuning_curves, angle, wake_ep, minmax=(0,2*np.pi))

    SI_thr=0.2
    hd_labels={}
    for s in spikes.keys():
        if SI['SI'][s]>=SI_thr:
            hd_labels[s]=1
        else:
            hd_labels[s]=0
        

############################################################################################### 
# GRID AND PLACE CELLS COMPUTING
###############################################################################################
    print('Grid and PC computing...')
    spikes=data.spikes
    feat=position['x','z']
    bin_size=12
    max_lag=10
    tuning_curves_2d,bins_xy = nap.compute_2d_tuning_curves(group=spikes, 
                                                        features=feat, 
                                                        nb_bins=bin_size,
                                                        ep = wake_ep)
    
    for k in spikes:
        tuning_curves_2d[k]=signal.medfilt(tuning_curves_2d[k],3)

    smooth_tc_2d={}
    for k in spikes:
        smooth_tc_2d[k]=gaussian_filter(tuning_curves_2d[k], sigma=bin_size/8)

    SI_2d = nap.compute_2d_mutual_info(tuning_curves_2d, feat, wake_ep)
    
    SI_2d_thr=0.5
    grid_labels={}
    for s in spikes.keys():
        if SI_2d['SI'][s]>=SI_2d_thr:
            grid_labels[s]=1
        else:
            grid_labels[s]=0
        
    if 'PC' in compute:
        num_curves = len(spikes)
        num_columns = 4
        num_rows = math.ceil(num_curves / num_columns)

        # Size per subplot (width, height) in inches
        width_per_subplot = 6
        height_per_subplot = 6

        # Total figure size
        total_width = width_per_subplot * num_columns
        total_height = height_per_subplot * num_rows
        plt.figure(figsize=(total_width, total_height))
        for i in spikes.keys():
            ts_to_features = spikes[i].value_from(feat)
            plt.subplot(num_rows, num_columns, i + 1)
            #plt.plot(ts_to_features["x"], ts_to_features["z"], "o", color="red", markersize=4)
            im = plt.imshow(
                tuning_curves_2d[i], extent=(bins_xy[1][0], bins_xy[1][-1], bins_xy[0][0], bins_xy[0][-1]),cmap="jet",interpolation='bilinear'
            )
            plt.colorbar(im, label='Firing Rate (Hz)')
            plt.xlabel('X Position')
            plt.ylabel('Z Position')
            #plt.legend(loc="upper right")
            plt.title(f"Cell: {i}")
        plt.suptitle(f"Recording: {r}")
        plt.tight_layout()
        plt.savefig(save_fig_path+r'\grid_pc.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_fig_path+r'\grid_pc.svg', bbox_inches='tight')
        plt.close()

        num_curves = len(spikes)
        num_columns = 4
        num_rows = math.ceil(num_curves / num_columns)

        # Size per subplot (width, height) in inches
        width_per_subplot = 6
        height_per_subplot = 6

        # Total figure size
        total_width = width_per_subplot * num_columns
        total_height = height_per_subplot * num_rows
        plt.figure(figsize=(total_width, total_height))
        for i in spikes.keys():
            ts_to_features = spikes[i].value_from(feat)
            occupancy, _, _ = np.histogram2d(feat["x"], feat["z"], bins=bin_size)
            occupancy[occupancy == 0] = 1
            occupancy_time = occupancy/120
            spikes_map, _, _ = np.histogram2d(ts_to_features["x"], ts_to_features["z"], bins=bin_size)
            rate_map = spikes_map / occupancy_time

            autocorr_map = correlate2d(rate_map, rate_map, boundary='fill', mode='full', fillvalue=0)
            autocorr_map = autocorr_map[autocorr_map.shape[0]//2 - max_lag: autocorr_map.shape[0]//2 + max_lag + 1, autocorr_map.shape[1]//2 - max_lag: autocorr_map.shape[1]//2 + max_lag + 1]
            autocorr_map /= autocorr_map.max()

            plt.subplot(num_rows, num_columns, i + 1)
            #plt.scatter(ts_to_features["x"], ts_to_features["z"], c="red", alpha=0.6, s=10, label='Spike Locations')
            extents = (
                np.min(feat["z"]),
                np.max(feat["z"]),
                np.min(feat["x"]),
                np.max(feat["x"]),
            )

        # Plotting the rate map
            im = plt.imshow(rate_map.T, origin='lower', extent=extents, cmap='jet', aspect='auto', interpolation='bilinear')
            plt.colorbar(im, label='Firing Rate (spikes per second)')
            plt.xlabel('X Position')
            plt.ylabel('Z Position')
            #plt.legend(loc="upper right")
            plt.title(f"Cell: {i}")
        plt.suptitle(f'Recording: {r} - Rate Map with Spike Locations')
        plt.tight_layout()
        plt.savefig(save_fig_path+r'\rate_map.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_fig_path+r'\rate_map.svg', bbox_inches='tight')
        plt.close()

        num_curves = len(spikes)
        num_columns = 4
        num_rows = math.ceil(num_curves / num_columns)

        # Size per subplot (width, height) in inches
        width_per_subplot = 6
        height_per_subplot = 6

        # Total figure size
        total_width = width_per_subplot * num_columns
        total_height = height_per_subplot * num_rows
        plt.figure(figsize=(total_width, total_height))
        for i in spikes.keys():
            ts_to_features = spikes[i].value_from(feat)
            occupancy, _, _ = np.histogram2d(feat["x"], feat["z"], bins=bin_size)
            occupancy[occupancy == 0] = 1
            occupancy_time = occupancy
            spikes_map, _, _ = np.histogram2d(ts_to_features["x"], ts_to_features["z"], bins=bin_size)
            rate_map = spikes_map / occupancy_time

            autocorr_map = correlate2d(rate_map, rate_map, boundary='fill', mode='full', fillvalue=0)
            autocorr_map = autocorr_map[autocorr_map.shape[0]//2 - max_lag: autocorr_map.shape[0]//2 + max_lag + 1, autocorr_map.shape[1]//2 - max_lag: autocorr_map.shape[1]//2 + max_lag + 1]
            autocorr_map /= autocorr_map.max()

            plt.subplot(num_rows, num_columns, i + 1)
            extent = [-max_lag, max_lag, -max_lag, max_lag]
            heatmap = plt.imshow(autocorr_map, cmap='jet', origin='lower', extent=extent, interpolation='bilinear')
            plt.colorbar(heatmap, label='Autocorrelation')
            plt.xlabel('Lag X')
            plt.ylabel('Lag Z')
            #plt.legend(loc="upper right")
            plt.title(f"Cell: {i}")
        plt.suptitle(f'Recording: {r} - Spatial Autocorrelation Map')
        plt.tight_layout()
        plt.savefig(save_fig_path+r'\auto_corr.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_fig_path+r'\auto_corr.svg', bbox_inches='tight')
        plt.close()
############################################################################################### 
# CELLS CLASSIFICATION
###############################################################################################
    with open(os.path.join(path, data.basename + '_mean_wf.pkl'), 'rb') as file:
        mean_wf = pickle.load(file)

    with open(os.path.join(path, data.basename + '_max_ch.pkl'), 'rb') as file:
        max_ch = pickle.load(file)

    ttp_value={}
    ttp_time={}
    for w in mean_wf.keys():
        tmp=mean_wf[w]
        peak_idx=tmp.min().idxmin()
        max_wf=tmp[peak_idx]
        ttp_value[w]=max(max_wf)-min(max_wf)
        ttp_time[w]=max_wf.idxmax()-max_wf.idxmin()
    
    fs_labels={}
    ex_labels={}
    dump_labels={}
    rate=spikes.get_info('rate')
    rate_thr=10
    ttp_thr=0.0004
    for s in spikes.keys():
        if ttp_time[s]<ttp_thr and rate[s]>rate_thr:
            fs_labels[s]=1
            ex_labels[s]=0
            dump_labels[s]=0
        elif ttp_time[s]>ttp_thr and rate[s]<rate_thr:
            fs_labels[s]=0
            ex_labels[s]=1
            dump_labels[s]=0
        else:
            fs_labels[s]=0
            ex_labels[s]=0
            dump_labels[s]=1   

    spikes=data.spikes
    spikes.set_info(hd=pd.Series(hd_labels), grid=pd.Series(grid_labels), fs=pd.Series(fs_labels), excitatory=pd.Series(ex_labels), dump=pd.Series(dump_labels))
    neuron_shank_map = {}
    for i, w in enumerate(max_ch):
        for s, c in channels.items():
            if w in c:
                neuron_shank_map[i] = (s, np.where(c == w)[0][0])
    # Calculate the relative distances
    neuron_distances = {}
    for neuron_idx, (shank, pos) in neuron_shank_map.items():
        base_distance = channel_spacing[r][shank] * pos  # Distance from the first channel
        neuron_distances[neuron_idx] = base_distance

    spikes.set_info(depth=pd.Series(neuron_distances))
    
    
    if 'CellClass' in compute:
        colors = ['red' if hd_labels[idx] == 1 and ex_labels[idx] == 1 else
            ('red' if hd_labels[idx] == 1 else
            ('blue' if hd_labels[idx] == 0 and ex_labels[idx] == 1 else
            ('green' if grid_labels[idx] == 1 else
                ('purple' if fs_labels[idx] == 1 else
                ('gray' if dump_labels[idx] == 1 else 'black')))))  # black would be a safeguard, shouldn't be used
            for idx in range(len(SI_2d))]
        # Plot setup
        plt.figure(figsize=(16, 12))
        plt.scatter(SI_2d, SI, c=colors, alpha=0.8, edgecolors='w', linewidth=0.5)
        # Create legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='HD', markerfacecolor='red', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Excitatory', markerfacecolor='blue', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Grid', markerfacecolor='green', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='FS', markerfacecolor='purple', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Unidentified', markerfacecolor='gray', markersize=10)
        ]

        # Adding the legend to the plot
        plt.xlabel('Spatial Information (bits per spike)')
        plt.ylabel('Head direction information (bits per spike)')
        plt.legend(handles=legend_elements, loc='upper left')
        plt.savefig(save_fig_path+r'\cell_class.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_fig_path+r'\cell_class.svg', bbox_inches='tight')
        plt.close()   


        for s in tmp_datasets:
            idx = spikes.getby_category('location')[s].index
            distances = [neuron_distances[i]/1000 for i in idx if i in neuron_distances] 
            si_values = SI['SI'][idx]
            plt.figure(figsize=(16, 12))
            plt.scatter(si_values, distances)
            plt.gca().invert_yaxis()  # Invert the y-axis of the current plot
            plt.ylabel("Neuron Distance (mm)")
            plt.xlabel("Head direction information (bits per spike)")
            plt.title(f"{s} Cells")
            plt.grid(True)
            plt.savefig(save_fig_path+r'\cells_depth_'+str(s)+'.png', dpi=300, bbox_inches='tight')
            plt.savefig(save_fig_path+r'\cells_depth_'+str(s)+'.svg', bbox_inches='tight')
            plt.close()

############################################################################################### 
# UP-DOWN STATE
###############################################################################################
    if 'UD' in compute:
        print('UD computing...')
        
        corr_r = correlate_TC_half_epochs(spikes, angle, 120, (0, 2*np.pi))
        spikes.set_info(halfr = corr_r)

        for s in tmp_datasets:
            try:
                spikes_ud = spikes.getby_category("location")[s].getby_category('hd')[1].getby_threshold('halfr', 0.5)
            except Exception as e:
                spikes_ud = spikes.getby_category("location")[s].getby_threshold('halfr', 0.5)


            # Proceed with alternative calculation
            total = spikes_ud.count(0.01, sws_ep).sum(axis=1) / 0.01
            total = total.as_series()
            
            # Apply Gaussian rolling window
            total2 = total.rolling(window=100, win_type='gaussian', center=True, min_periods=1).mean(std=2)
            total2 = nap.Tsd(total2, time_support=sws_ep)
            
            # Compute down episodes based on thresholding
            down_ep = total2.threshold(np.percentile(total2, 20), method='below').time_support
            down_ep = down_ep.merge_close_intervals(0.25)
            down_ep = down_ep.drop_short_intervals(0.05)
            down_ep = down_ep.drop_long_intervals(2)
            
            # Compute up episodes as set difference from SWS and down episodes
            up_ep = sws_ep.set_diff(down_ep)
            
            # Calculate top episodes based on higher thresholding
            top_ep = total2.threshold(np.percentile(total2, 80), method='above').time_support
            
            # Attempt to access the starts and interval centers again
            up_ts = up_ep.starts
            down_ts = down_ep.get_intervals_center()

            spikes_sp = spikes.getby_category("location")[s]
            names = spikes_sp.keys()
            print('######## '+s+' ########')
            if ufo_ts is not None and up_ts is not None:
                for e, ts in zip(['up', 'down'], [up_ts, down_ts]):
                    #grp = nap.TsGroup({0:ts,1:ufo_ts}, evt = np.array([e, 'ufo']))
                    #cc = nap.compute_crosscorrelogram(grp, 0.001, 1, sws_ep)
                    cc = nap.compute_eventcorrelogram(spikes_sp,ts, 0.01, 1, sws_ep)
                    cc.columns = names
                    ccs[s][e].append(cc)

        for s in ccs.keys():
            for e in ccs[s].keys():
                ccs[s][e] = pd.concat(ccs[s][e], axis=1)
        plot_ccs(ccs,r,save_fig_path,order_of_use='3')
    plt.show()