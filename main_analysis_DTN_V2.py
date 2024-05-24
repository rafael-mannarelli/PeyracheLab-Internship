# Standard library imports
import math
import os
import sys
import time
from itertools import combinations
from multiprocessing import Pool

# Third-party library imports
import matplotlib.pyplot as plt
from matplotlib import rcParams
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

def plot_ufo_lfp(data,event_idx, sign_channels,ufo_ep,frequency):
    data.load_neurosuite_xml(data.path)
    nb_channels=data.nChannels
    filename = data.basename + ".dat"    
    fp, timestep = get_memory_map(os.path.join(data.path, filename), nb_channels)
    plt.figure(figsize=(15, 8))
    event = ufo_ep[event_idx]
    start, end = event['start'], event['end']
    start_idx = int((start-0.01)*frequency)
    end_idx = int((end+0.01)*frequency)

    signal={}
    for c in sign_channels:
        signal[c] = fp[start_idx:end_idx, c]
        times = timestep[start_idx:end_idx]
    
    signal=pd.DataFrame.from_dict(signal)
    plt.plot(times, signal.mean(axis=1), label=f'Channel {c}')

    plt.title(f'UFO Event {event_idx} from {start} to {end} seconds')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (mV)')
    plt.legend()
    plt.show()

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
    pastel_red = (1.0, 0.5, 0.5, 1.0)  # RGBA for pastel red
    pastel_blue = (0.5, 0.5, 1.0, 1.0)  # RGBA for pastel blue
    #colors = plt.cm.get_cmap('bwr', num_channels)  
    for idx, (s, ax) in enumerate(zip(channels, axs)):
        sign_channels = channels[s]
        #color = colors(idx)
        offset = 0  
        offset_increment = 300 
        if s in [3, 4]:
                color = pastel_red
        else:
                color = pastel_blue
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
    plt.savefig(save_fig_path + r'\lfp_DTN.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_fig_path + r'\lfp_DTN.svg', bbox_inches='tight')
    plt.close()

    groups = unique(spikes.get_info('group').values)  
    fig, axs = plt.subplots(len(groups), 1, figsize=(6, 5)) 
    for ax, g in zip(axs, groups):
        tmp = spikes.getby_category('group')[g]
        color = pastel_red if g in [3, 4] else pastel_blue

        # Plot on the corresponding subplot
        for i in tmp.index:
            ax.eventplot(tmp[i].t, color=color)  # Plot each set of event times

        ax.set_xlim([times[0], times[-1]])  
    fig.tight_layout()  # Adjust the layout to make sure there's no overlap
    plt.savefig(save_fig_path + r'\raster_DTN.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_fig_path + r'\raster_DTN.svg', bbox_inches='tight')
    plt.close()

def get_memory_map(filepath, nChannels, frequency=20000):
    """Summary
    
    Args:
        filepath (TYPE): Description
        nChannels (TYPE): Description
        frequency (int, optional): Description
    """
    n_channels = int(nChannels)    
    f = open(filepath, 'rb') 
    startoffile = f.seek(0, 0)
    endoffile = f.seek(0, 2)
    bytes_size = 2      
    n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
    duration = n_samples/frequency
    interval = 1/frequency
    f.close()
    fp = np.memmap(filepath, np.int16, 'r', shape = (n_samples, n_channels))        
    timestep = np.arange(0, n_samples)/frequency

    return fp, timestep

def figsize(scale):
    fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0) / 2           # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    # fig_width = 5
    fig_height = fig_width*golden_mean*0.95         # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # ax.xaxis.set_tick_params(size=6)
    # ax.yaxis.set_tick_params(size=6)

def noaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.xaxis.set_tick_params(size=6)
    # ax.yaxis.set_tick_params(size=6)

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
compute=['HD','CC'] #'HD', 'CC' or 'LFP'
ufo_filename='ufo'

#data_directory = r'D:\PeyracheLab Internship\Data'
data_directory = r'E:\Data PeyracheLab'

datasets = {r'\A4002-200121':['DTN'],
            r'\A4007-200801':['DTN'],
            r'\A4004-200317b':['DTN'],
            r'\A4002-200120b':['DTN']}

fs=20000

ufo_shanks_nb = {r'\A4002-200120b':[3,1], 
            r'\A4002-200121':[3,1],
            r'\A4004-200317b':[1,0],
            r'\A4007-200801':[4,1]} 

save_fig_path_total=r'D:\PeyracheLab Internship\Figure'


meanfr_ufo_total={}
error_ufo_total={}
meanfr_ctrl_total={}
error_ctrl_total={}

for r in datasets.keys():
    print('######## '+r+' ########')
    ### Load session data
    path = data_directory + r
    data = ntm.load_session(path, 'neurosuite')

    if 'DTN' in datasets[r]:
        tracking_data_path = os.path.join(path, 'Analysis', 'Tracking_data.csv')
        
        # Check if the tracking data file exists
        if os.path.exists(tracking_data_path):
            tracking_data = pd.read_csv(tracking_data_path, header=None)
            tracking_data.iloc[:,1:2] = tracking_data.iloc[:,1:2] * 1.6
            tracking_data.iloc[:, -3:] = (np.radians(tracking_data.iloc[:, -3:]) % (2 * np.pi))
            position = nap.TsdFrame(t=tracking_data.iloc[:,0].values,
                                    d=tracking_data.iloc[:,1:].values,
                                    time_support=nap.IntervalSet(start=min(tracking_data.iloc[:,0]), 
                                                                end=max(tracking_data.iloc[:,0])),
                                    columns=['x','z','ry','rz','rx'])
        else:
            print(f"Tracking data file does not exist: {tracking_data_path}")
            # Handle the case where the tracking data file does not exist, e.g., use default data
            position = data.position
    else:
        position = data.position

    ### For saving figure
    save_fig_path=r'D:\PeyracheLab Internship\Figure'+ r
    os.makedirs(save_fig_path, exist_ok=True)

    ufo_ep, ufo_ts = loadUFOsV2(path,ufo_filename)

    data.load_neurosuite_xml(data.path)
    channels = data.group_to_channel
    nb_channels=data.nChannels

    all_ep=nap.IntervalSet(start=data.time_support['start'][0], end=data.time_support['end'][-1])

############################################################################################## 
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

    ttl_ep=nap.IntervalSet(start=position.time_support['start'][0], end=position.time_support['end'][-1])
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

    SI_thr=0.1
    hd_labels={}
    for s in spikes.keys():
        if SI['SI'][s]>=SI_thr:
            hd_labels[s]=1
        else:
            hd_labels[s]=0
    
    spikes.set_info(hd=pd.Series(hd_labels), count_ttl=spikes.restrict(ttl_ep).count().d[0])
    spikes_hd=spikes.getby_category('hd')[1]
    spikes_non_hd=spikes.getby_category('hd')[0]

    if 'HD' in compute:
        ### Plotting
        spike_groups = spikes.getby_threshold('count_ttl',0,'>').get_info('hd')
        # Number of unique groups
        unique_groups = set(spike_groups.values)

        pastel_red = (1.0, 0.5, 0.5, 1.0)  # RGBA for pastel red
        pastel_blue = (0.5, 0.5, 1.0, 1.0)  # RGBA for pastel blue

        # Map each group to a color
        group_to_color = {group: pastel_red if group == 1 else pastel_blue for group in unique_groups}

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
        #plt.figure(figsize=(total_width, total_height))
        plt.figure(figsize=(8, 4))
        
        # Assuming 'tuning_curves' is predefined.
        for (i, n) in enumerate(spikes.getby_threshold('count_ttl',0,'>').keys()):
            ax = plt.subplot(num_rows, num_columns, i + 1, projection='polar')
            ax.plot(smoothcurves[n], color=color.loc[n], linewidth=3)
            ax.set_xticklabels([])  # Remove angle labels.
            ax.set_yticklabels([])  # Remove radius labels.            
            ax.set_yticks([])  # Remove radius ticks.
        handles = [
            mpatches.Patch(facecolor=pastel_red, edgecolor='none', label='DTN-HD'),
            mpatches.Patch(facecolor=pastel_blue, edgecolor='none', label='Unclassified')
        ]
        plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=2)


        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust the spacing between the plots if necessary.
        plt.savefig(save_fig_path + r'\hd_2.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_fig_path + r'\hd_2.svg', bbox_inches='tight')
        plt.close()

        spike_groups = spikes.getby_threshold('count_ttl',0,'>').get_info('hd')
        # Number of unique groups
        unique_groups = set(spike_groups.values)

        pastel_red = (1.0, 0.5, 0.5, 1.0)  # RGBA for pastel red
        pastel_blue = (0.5, 0.5, 1.0, 1.0)  # RGBA for pastel blue

        # Map each group to a color
        group_to_color = {group: pastel_red if group == 1 else pastel_blue for group in unique_groups}

        # Convert group colors to a DataFrame for easy lookup
        color = pd.DataFrame([group_to_color[spike_groups[i]] for i in spike_groups.keys()],
                            index=spike_groups.keys(), columns=['r', 'g', 'b', 'a'])

        #plt.figure(figsize=(5, 3))
        plt.figure(figsize=(3, 2))
        for (i, n) in enumerate(spikes.getby_threshold('count_ttl',0,'>').keys()):
            plt.scatter(SI.loc[n],spikes.get_info('rate')[n],color=color.loc[n])
        plt.ylabel('Cells mean firing rate')
        plt.xlabel('HD info')
        plt.axvline(0.1, linewidth=2, color="k", linestyle="--")
        plt.savefig(save_fig_path + r'\hd_class.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_fig_path + r'\hd_class.svg', bbox_inches='tight')
        plt.close()
    
############################################################################################### 
# PETH COMPUTING
###############################################################################################     
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

############################################################################################### 
# CC and FR COMPUTING
###############################################################################################
    if 'CC' in compute:
        print('CC computing...')
        win_size=0.05
        group_boundaries = []
        spike_groups=spikes.get_info('group')
        for i in range(1, len(spike_groups)):
            if spike_groups[i] != spike_groups[i-1]:
                group_boundaries.append(i)  
        cc = nap.compute_eventcorrelogram(spikes, ufo_ts, 0.001, win_size, None, norm=False)
        cc=zscore(cc.loc[-win_size:win_size])
        

        num_bins = cc.shape[1]
        bin_height = 1
        #plt.figure(figsize=(15, 8))
        plt.figure(figsize=(4, 4))
        im = plt.imshow(cc.T.fillna(0), aspect='auto', cmap='bwr', interpolation='none', extent=[-win_size, win_size, 0, cc.shape[1]], origin='lower')        
        plt.colorbar(label='Normalized firing rate')
        plt.axvline(0, color='white', linestyle='--') 
        for boundary in group_boundaries:
            plt.axhline(y=boundary, color='red', linestyle='--')
        tick_positions = [x + bin_height / 2 for x in range(0, num_bins, max(1, num_bins // 10))]
        tick_step = max(1, num_bins // 10)
        tick_labels = cc.keys()[::tick_step]
        plt.yticks(tick_positions, tick_labels)
        plt.xlabel('Time from UFO event (s)')
        plt.ylabel('Cell index')
        plt.title(f"Recording: {r}")
        plt.savefig(save_fig_path+r'\cc.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_fig_path+r'\cc.svg', bbox_inches='tight')
        plt.close()

        group_boundaries = []
        spike_groups=spikes_hd.get_info('group')
        for i in range(1, len(spike_groups)):
            if spike_groups.iloc[i] != spike_groups.iloc[i-1]:
                group_boundaries.append(i)  
        cc = nap.compute_eventcorrelogram(spikes_hd, ufo_ts, 0.001, win_size, None, norm=False)
        cc=zscore(cc.loc[-win_size:win_size])

        num_bins = cc.shape[1]
        bin_height = 1
        #plt.figure(figsize=(4, 4))
        plt.figure(figsize=(4, 4))
        im = plt.imshow(cc.T.fillna(0), aspect='auto', cmap='turbo', interpolation='none', extent=[-win_size, win_size, 0, cc.shape[1]], origin='lower')        
        plt.colorbar(label='Normalized firing rate')
        plt.axvline(0, color='white', linestyle='--')
        for boundary in group_boundaries:
            plt.axhline(y=boundary, color='red', linestyle='--')
        tick_positions = [x + bin_height / 2 for x in range(0, num_bins, max(1, num_bins // 10))]
        tick_step = max(1, num_bins // 10)
        tick_labels = cc.keys()[::tick_step]+1
        plt.yticks(tick_positions, tick_labels)
        plt.xlabel('Time from UFO event (s)')
        plt.ylabel('Cell index')
        plt.title(f"DTN cells")
        plt.savefig(save_fig_path+r'\cc_1.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_fig_path+r'\cc_1.svg', bbox_inches='tight')
        plt.close()

        group_boundaries = []
        spike_groups=spikes_non_hd.get_info('group')
        for i in range(1, len(spike_groups)):
            if spike_groups.iloc[i] != spike_groups.iloc[i-1]:
                group_boundaries.append(i) 
        cc = nap.compute_eventcorrelogram(spikes_non_hd, ufo_ts, 0.001, win_size, None, norm=False)
        cc=zscore(cc.loc[-win_size:win_size])

        num_bins = cc.shape[1]
        bin_height = 1
        plt.figure(figsize=(4, 4))
        im = plt.imshow(cc.T.fillna(0), aspect='auto', cmap='turbo', interpolation='none', extent=[-win_size, win_size, 0, cc.shape[1]], origin='lower')        
        plt.colorbar(label='Normalized firing rate')
        plt.axvline(0, color='white', linestyle='--') 
        for boundary in group_boundaries:
            plt.axhline(y=boundary, color='red', linestyle='--')
        tick_positions = [x + bin_height / 2 for x in range(0, num_bins, max(1, num_bins // 10))]
        tick_step = max(1, num_bins // 10)
        tick_labels = cc.keys()[::tick_step]+1
        plt.yticks(tick_positions, tick_labels)
        plt.xlabel('Time from UFO event (s)')
        plt.ylabel('Cell index')
        plt.title(f"non DTN cells")
        plt.savefig(save_fig_path+r'\cc_2.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_fig_path+r'\cc_2.svg', bbox_inches='tight')
        plt.close()

############################################################################################### 
# PETH COMPUTING (second part)
###############################################################################################     

if 'HD' in compute:
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
    plt.savefig(save_fig_path_total+r'\peth_DTN_ctrl.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_fig_path_total + r'\peth_DTN_ctrl.svg', bbox_inches='tight')
    plt.close()

if 'LFP' in compute:
    event_idx=191
    path = data_directory + r'\A4002-200120b'
    data = ntm.load_session(path, 'neurosuite')
    ufo_ep, ufo_ts = loadUFOsV2(path,ufo_filename)
    plot_ufo_lfp_stacked(data,event_idx,ufo_ep,fs,save_fig_path_total)

plt.show()


