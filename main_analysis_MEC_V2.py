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
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pycircstat.descriptive import mean as circmean
from scipy import signal, ndimage
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d
import pickle as pickle  

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
        if s in [4,5,6]:
                color = pastel_orange
        else:
                color = pastel_purple
        for c in sign_channels:
            signal = fp[start_idx:end_idx, c]

            # Smooth the signal using a moving average
            window_size = int(frequency * 0.0005)
            smoothed_signal = uniform_filter1d(signal, size=window_size)

            ax.plot(times, smoothed_signal+ offset, color=color)
            offset += offset_increment
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(-1000, 5000)
    plt.savefig(save_fig_path + r'\lfp_MEC.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_fig_path + r'\lfp_MEC.svg', bbox_inches='tight')
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
compute=['LFP'] #'HD', 'GC', 'RM', 'CellClass', 'CC' or 'PETH'

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

save_fig_path_total=r'D:\PeyracheLab Internship\Figure'

meanfr_ufo_total={}
error_ufo_total={}
meanfr_ctrl_total={}
error_ctrl_total={}


for r in datasets.keys():
    ccs_long = {s:{e:[] for e in ['Wake', 'REM', 'NREM']} for s in ['AD','MEC-ex','MEC-fs']}
    ccs_short = {s:{e:[] for e in ['Wake', 'REM', 'NREM']} for s in ['AD','MEC-ex','MEC-fs']}
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
        SI_thr=0.1
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

            ### Smooth curves computations
            smoothcurves = smoothAngularTuningCurvesHD(tuning_curves, sigma=3)

            SI = nap.compute_1d_mutual_info(tuning_curves, angle, ttl_ep, minmax=(0,2*np.pi))

            hd_labels={}
            for s in spikes.keys():
                if SI['SI'][s]>=SI_thr:
                    hd_labels[s]=1
                else:
                    hd_labels[s]=0

            spikes.set_info(hd=pd.Series(hd_labels),count_ttl=spikes.restrict(ttl_ep).count().d[0])
            spikes_hd=spikes.getby_category('hd')[1]
            spikes_non_hd=spikes.getby_category('hd')[0]

            ### Plotting
            spike_groups = spikes.get_info('location') 
            #spike_groups = spikes.getby_threshold('count_ttl',0,'>').get_info('hd')
            # Number of unique groups
            unique_groups = set(spike_groups.values)

            pastel_purple = (0.7, 0.5, 0.9, 1.0)  # RGBA for pastel purple
            pastel_orange = (1.0, 0.7, 0.5, 1.0)  # RGBA for pastel orange

            # Map each group to a color
            group_to_color = {group: pastel_purple if group == 'AD' else pastel_orange for group in unique_groups}

            # Convert group colors to a DataFrame for easy lookup
            color = pd.DataFrame([group_to_color[spike_groups[i]] for i in spike_groups.keys()],
                                index=spike_groups.keys(), columns=['r', 'g', 'b', 'a'])

            num_curves = len(spikes)
            num_columns = 10
            num_rows = math.ceil(num_curves / num_columns)

            # Size per subplot (width, height) in inches
            width_per_subplot = 1
            height_per_subplot = 1

            # Total figure size
            total_width = width_per_subplot * num_columns
            total_height = height_per_subplot * num_rows
            plt.figure(figsize=(total_width, total_height))

            # Assuming 'tuning_curves' is predefined.
            for (i, n) in enumerate(spikes.getby_threshold('count_ttl',0,'>').keys()):
                ax = plt.subplot(num_rows, num_columns, i + 1, projection='polar')
                ax.plot(smoothcurves[n], color=color.loc[n], linewidth=3)
                ax.set_xticklabels([])  # Remove angle labels.
                ax.set_yticklabels([])  # Remove radius labels.            
                ax.set_yticks([])  # Remove radius ticks.
            handles = [
                mpatches.Patch(facecolor=pastel_purple, edgecolor='none', label='AD'),
                mpatches.Patch(facecolor=pastel_orange, edgecolor='none', label='MEC')
            ]
            plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=2)


            plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust the spacing between the plots if necessary.
            plt.savefig(save_fig_path + r'\hd_'+str(e)+'.png', dpi=300, bbox_inches='tight')
            plt.savefig(save_fig_path + r'\hd_'+str(e)+'.svg', bbox_inches='tight')
            plt.close()

            spike_groups = spikes.get_info('location') 

            #spike_groups = spikes.getby_threshold('count_ttl',0,'>').get_info('hd')
            # Number of unique groups
            unique_groups = set(spike_groups.values)

            # Map each group to a color
            group_to_color = {group: pastel_purple if group == 'AD' else pastel_orange for group in unique_groups}

            # Convert group colors to a DataFrame for easy lookup
            color = pd.DataFrame([group_to_color[spike_groups[i]] for i in spike_groups.keys()],
                                index=spike_groups.keys(), columns=['r', 'g', 'b', 'a'])

            plt.figure(figsize=(5, 3))
            for (i, n) in enumerate(spikes.getby_threshold('count_ttl',0,'>').getby_category('location')['AD'].keys()):
                plt.scatter(SI.loc[n],spikes.get_info('rate')[n],color=pastel_purple)#color.loc[n])
            plt.ylabel('Cells mean firing rate')
            plt.xlabel('HD info')
            plt.title('HD cell classification in AD')
            plt.axvline(0.1, linewidth=2, color="k", linestyle="--")
            plt.savefig(save_fig_path + r'\hd_class.png', dpi=300, bbox_inches='tight')
            plt.savefig(save_fig_path + r'\hd_class.svg', bbox_inches='tight')
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

        hd_labels={}
        for s in spikes.keys():
            if SI['SI'][s]>=SI_thr:
                hd_labels[s]=1
            else:
                hd_labels[s]=0

        spikes.set_info(hd=pd.Series(hd_labels), count_ttl=spikes.restrict(wake_ep).count().d[0])
        spikes_hd=spikes.getby_category('hd')[1]
        spikes_non_hd=spikes.getby_category('hd')[0]
        

############################################################################################### 
# GRID AND PLACE CELLS COMPUTING
###############################################################################################
    print('GC computing...')
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
        
    if 'RM' in compute:
        num_curves = len(spikes)
        num_columns = 4
        num_rows = math.ceil(num_curves / num_columns)

        # Size per subplot (width, height) in inches
        width_per_subplot = 5
        height_per_subplot = 4

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
            plt.title(f"Cell: {i+1}")
        #plt.suptitle(f'Recording: {r} - Rate Map with Spike Locations')
        plt.tight_layout()
        plt.savefig(save_fig_path+r'\rate_map.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_fig_path+r'\rate_map.svg', bbox_inches='tight')
        plt.close()

        num_curves = len(spikes)
        num_columns = 4
        num_rows = math.ceil(num_curves / num_columns)

        # Size per subplot (width, height) in inches
        width_per_subplot = 5
        height_per_subplot = 5

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
        #plt.suptitle(f'Recording: {r} - Spatial Autocorrelation Map')
        plt.tight_layout()
        plt.savefig(save_fig_path+r'\auto_corr.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_fig_path+r'\auto_corr.svg', bbox_inches='tight')
        plt.close()
############################################################################################### 
# CELLS CLASSIFICATION
###############################################################################################
    if 'CellClass' in compute:
        if r in r'\B0714-230221':
            with open(path + r'\principalinterneuron_classification.pickle', 'rb') as file:
                rec_pkl = pickle.load(file)

            max_wf=rec_pkl['maxwf']

            ttp_value={}
            ttp_time={}
            for w in max_wf.keys():
                ttp_value[w]=max(max_wf[w])-min(max_wf[w])
                ttp_time[w]=max_wf[w].idxmax()-max_wf[w].idxmin()
            
            fs_labels={}
            ex_labels={}
            dump_labels={}
            rate=spikes.get_info('rate')
            rate_thr=10
            ttp_thr=0.0004
            for s in spikes.keys():
                if spikes.get_info('location')[s]=='AD':
                    fs_labels[s]=0
                    ex_labels[s]=0
                    dump_labels[s]=0
                else:
                    if s in ttp_time.keys():
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
                    else:
                        dump_labels[s]=1
                        fs_labels[s]=0
                        ex_labels[s]=0   
        else:
            with open(os.path.join(path, data.basename + '_mean_wf.pkl'), 'rb') as file:
                mean_wf = pickle.load(file)

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

        spikes_ex=spikes.getby_category('excitatory')[1]
        spikes_in=spikes.getby_category('excitatory')[0]

        print(spikes)

        if 'CC' in compute:
            for s in ['AD', 'MEC-ex','MEC-fs']:
                print('######## '+s+' ########')
                if s=='AD':
                    idx= spikes._metadata[spikes._metadata["location"].str.contains(s)].index.values
                    spikes_tmp=spikes[idx]
                elif s=='MEC-ex':
                    idx= spikes._metadata[spikes._metadata["location"].str.contains('MEC')].index.values
                    spikes_tmp=spikes[idx].getby_category('excitatory')[1]
                elif s=='MEC-fs':
                    idx= spikes._metadata[spikes._metadata["location"].str.contains('MEC')].index.values
                    spikes_tmp=spikes[idx].getby_category('fs')[1]

    ############################################################################################### 
    # CC COMPUTING
    ###############################################################################################
                print('CC computing...')

                ufo_ts_start=ufo_ep['start']
                ufo_ts_start=nap.Ts(ufo_ts_start)

                #names = [s.split("/")[-1] + "_" + str(n) for n in spikes.keys()]
                names = spikes_tmp.keys()

                for e, ep in zip(['Wake', 'REM', 'NREM'], [wake_ep, rem_ep, sws_ep]):            
                    cc = nap.compute_eventcorrelogram(spikes_tmp, ufo_ts_start, 0.01, 0.250, ep, norm=True)
                    #cc.columns = names
                    #cc=zscore(cc.loc[-0.250:0.250])
                    ccs_long[s][e].append(cc)

                    cc = nap.compute_eventcorrelogram(spikes_tmp, ufo_ts_start, 0.001, 0.025, ep, norm=True)
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

            idx_mec= spikes._metadata[spikes._metadata["location"].str.contains('MEC')].index.values
        
            colors = ['green' if grid_labels[idx] == 1  and hd_labels[idx] ==1 else
                      ('green' if grid_labels[idx] == 1 else 
                ('red' if hd_labels[idx] == 1 and grid_labels[idx] == 0 else
                ('blue' if hd_labels[idx] == 0 and ex_labels[idx] == 1 and grid_labels[idx] == 0 else
                ('red' if hd_labels[idx] == 1 and ex_labels[idx] == 1 else
                    ('purple' if fs_labels[idx] == 1 else
                    ('gray' if dump_labels[idx] == 1 else 'black'))))))  
                for idx in SI_2d.index[idx_mec]]
            # Plot setup
            plt.figure(figsize=(7, 4))
            plt.scatter(SI_2d.loc[idx_mec], SI.loc[idx_mec], c=colors, alpha=0.8, edgecolors='w', linewidth=0.5)
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
            plt.ylabel('HD information (bits per spike)')
            plt.title('Cells classification in the MEC')
            plt.legend(handles=legend_elements, loc='upper left')
            plt.savefig(save_fig_path+r'\cell_class.png', dpi=300, bbox_inches='tight')
            plt.savefig(save_fig_path+r'\cell_class.svg', bbox_inches='tight')
            plt.close()   
    
 ############################################################################################### 
# PETH COMPUTING
###############################################################################################     
    if 'PETH' in compute:    
        ufo_peth={}
        counts_ufo={}
        fr_ufo={}
        meanfr_ufo={}
        error_ufo={}

        ctrl_peth={}
        counts_ctrl={}
        fr_ctrl={}
        meanfr_ctrl={}
        error_ctrl={}

        
        min_max=(-0.25,0.25)
        bin_size = 0.01  # 200ms bin size
        step_size = 0.001  # 10ms step size, to make overlapping bins
        winsize = int(bin_size / step_size)  # Window size
        
        for _,j in enumerate(spikes_hd):
            ufo_peth[j]=nap.compute_perievent(spikes_hd[j],ufo_ts,minmax=min_max)
            counts_ufo[j] = ufo_peth[j].count(step_size, time_units='s')
            counts_ufo[j] = (
                counts_ufo[j].as_dataframe()
                .rolling(winsize, win_type="gaussian", min_periods=1, center=True, axis=0)
                .mean(std=0.2 * winsize)
            )
            fr_ufo[j] = (counts_ufo[j] * winsize)/bin_size
            meanfr_ufo[j] = fr_ufo[j].mean(axis=1)
            error_ufo[j] = fr_ufo[j].sem(axis=1)

        z_meanfr_ufo = {j: zscore(meanfr) for j, meanfr in meanfr_ufo.items()}
        
        meanfr_ufo_total[r]=pd.DataFrame.from_dict(z_meanfr_ufo).mean(axis=1)
        error_ufo_total[r]=pd.DataFrame.from_dict(z_meanfr_ufo).sem(axis=1)

        for _,j in enumerate(spikes_non_hd.getby_threshold('count_ttl',10000,'>')):
            ctrl_peth[j]=nap.compute_perievent(spikes_non_hd[j],ufo_ts,minmax=min_max)
            counts_ctrl[j] = ctrl_peth[j].count(step_size, time_units='s')
            counts_ctrl[j] = (
                counts_ctrl[j].as_dataframe()
                .rolling(winsize, win_type="gaussian", min_periods=1, center=True, axis=0)
                .mean(std=0.2 * winsize)
            )
            fr_ctrl[j] = (counts_ctrl[j] * winsize)/bin_size
            meanfr_ctrl[j] = fr_ctrl[j].mean(axis=1)
            error_ctrl[j] = fr_ctrl[j].sem(axis=1)

        z_meanfr_ctrl = {j: zscore(meanfr) for j, meanfr in meanfr_ctrl.items()}
        
        meanfr_ctrl_total[r]=pd.DataFrame.from_dict(z_meanfr_ctrl).mean(axis=1)
        error_ctrl_total[r]=pd.DataFrame.from_dict(z_meanfr_ctrl).sem(axis=1)

if 'PETH' in compute:
    meanfr_ufo_total_mean=pd.DataFrame.from_dict(meanfr_ufo_total).mean(axis=1)
    meanfr_ctrl_total_mean=pd.DataFrame.from_dict(meanfr_ctrl_total).mean(axis=1)
    error_ufo_total_mean=pd.DataFrame.from_dict(meanfr_ufo_total).sem(axis=1)
    error_ctrl_total_mean=pd.DataFrame.from_dict(meanfr_ctrl_total).sem(axis=1)


    plt.figure(figsize=(8, 4))
    #plt.figure(figsize=figsize(0.75))
    colors = [
        "#FFA3C1",  # Enhanced pink
        "#A3D1FF",  # Enhanced light blue
        "#C0FFB3",  # Brighter green
        "#FFD8A8",  # Richer peach
        "#F0A9F0",  # Brighter lilac
        "#FFD1A8",  # Warmer light orange
        "#E6B0C3",  # Deeper mauve
        "#B3E2FF"   # Lighter sky blue
    ]
    cmap = LinearSegmentedColormap.from_list("CustomPastel", colors, N=len(meanfr_ufo_total))
    colors = [cmap(i) for i in range(len(meanfr_ufo_total))]

    # Plot each normalized mean firing rate with SEM
    for (index, meanfr_ufo_total), color in zip(meanfr_ufo_total.items(), colors):
        plt.plot(meanfr_ufo_total.index.values, meanfr_ufo_total.values, color='black',alpha=0.7,linewidth=0.5, label=f"Recording {index}")
        
        plt.fill_between(
            meanfr_ufo_total.index.values,
            meanfr_ufo_total.values - error_ufo_total[index],
            meanfr_ufo_total.values + error_ufo_total[index],
            color='grey',
            alpha=0.1,
        )
    plt.plot(meanfr_ufo_total_mean.index.values, meanfr_ufo_total_mean.values, color='red', label='Mean')
    plt.axvline(0, linewidth=2, color="k", linestyle="--")
    plt.xlabel("Time from UFO event (s)")
    plt.ylabel("Z-score normalized firing rate")
    plt.ylim([-2,7])
    plt.xlim([-0.1,0.1])
    plt.legend(loc="upper right")
    plt.title(f"DTN cells PETH for all the recording")
    plt.savefig(save_fig_path_total+r'\peth_DTN.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_fig_path_total + r'\peth_DTN.svg', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 4))
    #plt.figure(figsize=figsize(0.75))
    colors = [
        "#FFA3C1",  # Enhanced pink
        "#A3D1FF",  # Enhanced light blue
        "#C0FFB3",  # Brighter green
        "#FFD8A8",  # Richer peach
        "#F0A9F0",  # Brighter lilac
        "#FFD1A8",  # Warmer light orange
        "#E6B0C3",  # Deeper mauve
        "#B3E2FF"   # Lighter sky blue
    ]    
    cmap = LinearSegmentedColormap.from_list("CustomPastel", colors, N=len(meanfr_ctrl_total))
    colors = [cmap(i) for i in range(len(meanfr_ctrl_total))]

    # Plot each normalized mean firing rate with SEM
    for (index, meanfr_ctrl_total), color in zip(meanfr_ctrl_total.items(), colors):
        plt.plot(meanfr_ctrl_total.index.values, meanfr_ctrl_total.values, color='black',alpha=0.7,linewidth=0.5, label=f"Recording {index}")
        
        plt.fill_between(
            meanfr_ctrl_total.index.values,
            meanfr_ctrl_total.values - error_ctrl_total[index],
            meanfr_ctrl_total.values + error_ctrl_total[index],
            color='grey',
            alpha=0.1,
        )
    plt.plot(meanfr_ctrl_total_mean.index.values, meanfr_ctrl_total_mean.values, color='red', label='Mean')
    plt.axvline(0, linewidth=2, color="k", linestyle="--")
    plt.xlabel("Time from UFO event (s)")
    plt.ylabel("Z-score normalized firing rate")
    plt.ylim([-2,7])
    plt.xlim([-0.1,0.1])
    plt.legend(loc="upper right")
    plt.title(f"Non DTN cells PETH for all the recording")
    plt.savefig(save_fig_path_total+r'\peth_MEC_ctrl.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_fig_path_total + r'\peth_MEC_ctrl.svg', bbox_inches='tight')
    plt.close()

if 'LFP' in compute:
    event_idx=0
    path = data_directory + r'\B0703-211129'
    data = ntm.load_session(path, 'neurosuite')
    ufo_ep, ufo_ts = loadUFOsV2(path,ufo_filename)
    plot_ufo_lfp_stacked(data,event_idx,ufo_ep,fs,save_fig_path_total)

plt.show()