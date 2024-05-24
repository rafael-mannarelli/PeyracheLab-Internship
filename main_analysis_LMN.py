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

def plot_ccs_old(ccs,recording,save_fig_path,order_of_use='1'):
    plt.rcParams.update({'font.size': 15})
    lw = 2
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Determine the number of epochs and recordings (assuming 'ccs' is structured consistently)
    nb_epochs = max(len(value.keys()) for value in ccs.values())
    nb_recordings = len(ccs)

    # Base size for each subplot (in inches)
    base_width = 4
    base_height = 4
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

def plot_ccs(ccs,recording,save_fig_path,order_of_use='1'):
    plt.rcParams.update({'font.size': 15})
    lw = 2
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Determine the number of epochs and recordings (assuming 'ccs' is structured consistently)
    nb_epochs = max(len(value.keys()) for value in ccs.values())
    nb_recordings = len(ccs)

    # Base size for each subplot (in inches)
    base_width = 3
    base_height = 4
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
            ax.set_xticklabels([round(x[0],3), 0, round(x[-1],3)])
            ax.set_yticks(ticks=range(len(ccs[r][e].keys())),labels=ccs[r][e].keys()+1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j, e in enumerate(ccs[next(iter(ccs))].keys()):
        ax = plt.subplot(gs[-1, j])
        for i, r in enumerate(ccs.keys()):
            ax.plot(ccs[r][e].mean(1), color=colors[i], linewidth=lw, label=r)
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

def plot_ufo_lfp_stacked(data, event_idx, ufo_ep, frequency,save_fig_path):
    from scipy.ndimage import uniform_filter1d
    data.load_neurosuite_xml(data.path)
    nb_channels = data.nChannels
    channels = data.group_to_channel
    filename = data.basename + ".dat"
    fp, timestep = get_memory_map(os.path.join(data.path, filename), nb_channels)

    event = ufo_ep[event_idx]
    start, end = event['start'], event['end']
    start_idx = int((start - 0.02) * frequency)
    end_idx = int((end + 0.02) * frequency)
    times = timestep[start_idx:end_idx]

    num_channels = len(channels)
    fig, axs = plt.subplots(num_channels, 1, figsize=(6, 20), sharex=True)
    pastel_purple = (0.7, 0.5, 0.9, 1.0)  # RGBA for pastel purple
    pastel_orange = (1.0, 0.7, 0.5, 1.0)  # RGBA for pastel orange

    #colors = plt.cm.get_cmap('bwr', num_channels)  
    for idx, (s, ax) in enumerate(zip(channels, axs)):
        sign_channels = channels[s]
        #color = colors(idx)
        offset = 0  
        offset_increment = 300 
        if s in [3, 4]:
                color = pastel_purple
        else:
                color = pastel_orange
        for c in sign_channels:
            signal = fp[start_idx:end_idx, c]

            # Smooth the signal using a moving average
            window_size = int(frequency * 0.0005)
            smoothed_signal = uniform_filter1d(signal, size=window_size)

            ax.plot(times, smoothed_signal+ offset, color=color)
            offset += offset_increment
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(-1000, 4000)
    plt.savefig(save_fig_path + r'\lfp_MEC.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_fig_path + r'\lfp_MEC.svg', bbox_inches='tight')
    plt.close()

    groups = unique(spikes.get_info('group').values)  
    fig, axs = plt.subplots(len(groups), 1, figsize=(6, 5)) 
    for ax, g in zip(axs, groups):
        tmp = spikes.getby_category('group')[g]
        color = pastel_purple if g in [3, 4] else pastel_orange

        # Plot on the corresponding subplot
        for i in tmp.index:
            ax.eventplot(tmp[i].t, color=color)  # Plot each set of event times

        ax.set_xlim([times[0], times[-1]])  
    fig.tight_layout()  # Adjust the layout to make sure there's no overlap
    plt.savefig(save_fig_path + r'\raster_MEC.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_fig_path + r'\raster_MEC.svg', bbox_inches='tight')
    plt.close()



# Configuration settings
fontsize = 14  # Set a fixed font size for all text
plt.rcParams.update({
    "font.size": fontsize,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "legend.fontsize": fontsize,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "text.color": (0.25, 0.25, 0.25),
    "axes.labelcolor": (0.25, 0.25, 0.25),
    "axes.edgecolor": (0.25, 0.25, 0.25),
    "xtick.color": (0.25, 0.25, 0.25),
    "ytick.color": (0.25, 0.25, 0.25),
    "axes.axisbelow": True,
    "xtick.major.size": 1.3,
    "ytick.major.size": 1.3,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.pad": 1,
    "ytick.major.pad": 1,
    "axes.linewidth": 0.4
})


############################################################################################### 
# GENERAL INFOS
###############################################################################################
compute=['CC'] #'HD' or 'CC'
ufo_filename='ufo'

data_directory = r'E:\Data PeyracheLab'
datasets = {r'/A5026/A5026-210725A':['lmn','thl'],
            r'/A5026/A5026-210727A':['lmn','thl']}
            #r'/B3007-240429A':['lmn'],
            #r'/B3007-240504A':['lmn'],
            #r'/B3009-240503A':['lmn'],
            #r'/B3009-240504C':['lmn']}
ufo_shanks_nb = {r'/B3007-240429A':[0,3],
            r'/B3007-240504A':[0,3],
            r'/B3009-240503A':[3,0],
            r'/B3009-240504C':[3,0]}

fs=20000

SI_thr = {
    'AD':0.0, 
    'LMN':0.0,
    'MMB':0.0,
    'PoSub':0.0,
    'MEC':0.0,
    'TRN':0.0
    }

save_fig_path_total=r'D:\PeyracheLab Internship\Figure'

for r in datasets.keys():
    ccs_long = {s:{e:[] for e in ['Wake', 'REM', 'NREM']} for s in ['LMN','CA1']}
    ccs_short = {s:{e:[] for e in ['Wake', 'REM', 'NREM']} for s in ['LMN','CA1']}
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
    rip_ep, rip_ts = loadRipplesV2(path)

    data.load_neurosuite_xml(data.path)
    channels = data.group_to_channel
    nb_channels=data.nChannels

    spikes=data.spikes
    if 'CC' in compute:
        for s in ['LMN', 'CA1']:
            print('######## '+s+' ########')
            if s=='LMN':
                idx= spikes._metadata[spikes._metadata["location"].str.contains('lmn')].index.values
                spikes_tmp=spikes[idx]
            elif s=='CA1':
                idx= spikes._metadata[spikes._metadata["location"].str.contains('thl')].index.values
                spikes_tmp=spikes[idx]

############################################################################################### 
# CC COMPUTING
###############################################################################################
            print('CC computing...')

            ufo_ts_start=ufo_ep['start']
            ufo_ts_start=nap.Ts(ufo_ts_start)

            #names = [s.split("/")[-1] + "_" + str(n) for n in spikes.keys()]
            names = spikes_tmp.keys()

            for e, ep in zip(['Wake', 'REM', 'NREM'], [wake_ep, rem_ep, sws_ep]):            
                cc = nap.compute_eventcorrelogram(spikes_tmp, rip_ts, 0.01, 0.250, ep, norm=True)
                #cc.columns = names
                #cc=zscore(cc.loc[-0.250:0.250])
                ccs_long[s][e].append(cc)

                cc = nap.compute_eventcorrelogram(spikes_tmp, rip_ts, 0.001, 0.025, ep, norm=True)
                #cc.columns = names
                #cc=zscore(cc.loc[-0.025:0.025])
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

    spikes.set_info(hd=pd.Series(hd_labels), count_ttl=spikes.restrict(wake_ep).count().d[0])

    print("Number of UFOs ",str(r),': ', len(ufo_ep))
    print("UFOs rate ",str(r),': ', nap.TsGroup({0:ufo_ts}).restrict(sws_ep))

if 'LFP' in compute:
    event_idx=27
    path = data_directory + r'\B0714-230221'
    data = ntm.load_session(path, 'neurosuite')
    ufo_ep, ufo_ts = loadUFOsV2(path,ufo_filename)
    plot_ufo_lfp_stacked(data,event_idx,ufo_ep,fs,save_fig_path_total)

plt.show()