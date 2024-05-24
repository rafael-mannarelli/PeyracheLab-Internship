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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pycircstat.descriptive import mean as circmean
from scipy import signal, ndimage
from scipy.interpolate import interp1d
from tqdm import tqdm
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
            im = ax.imshow(tmp.T, aspect='auto', cmap='jet',interpolation='kaiser')
            
            x = ccs[r][e].index.values
            ax.set_xticks([0, len(x)//2, len(x)-1])
            ax.set_xticklabels([f'{x[0]:.3f}', '0.000', f'{x[-1]:.3f}'])
            ax.set_yticks(ticks=range(len(ccs[r][e].keys())),labels=ccs[r][e].keys())
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

def zscore(x):
    """Return the z-score of the provided array."""
    return (x - x.mean(axis=0)) / x.std(axis=0)

def parse_ttl_csv(filepath):
    tracking_data=pd.read_csv(filepath, skiprows=6)
    tracking_data=tracking_data.drop('Frame', axis=1)
    tracking_data.columns=['time', 'ry','rx','rz','X','Y','Z']
    tracking_data.iloc[:, 1:4] = (np.radians(tracking_data.iloc[:, 1:4]) % (2 * np.pi))
    position = nap.TsdFrame(t=tracking_data.iloc[:,0].values,
                                    d=tracking_data.iloc[:,1:].values,
                                    time_support=nap.IntervalSet(start=min(tracking_data.iloc[:,0]), 
                                                                end=max(tracking_data.iloc[:,0])),
                                    columns=['ry','rx','rz','X','Y','Z']) 
    return position

############################################################################################### 
# GENERAL INFOS
###############################################################################################
plot=False
#data_directory = r'D:\PeyracheLab Internship\Data'
data_directory = r'E:\Data PeyracheLab'
datasets = {r'\B0714-230221':['AD', 'MEC']}
            #r'\B3205-231031':['AD','TRN']}
            #'\A3024-220915A':['LMN','PoSub','MMB']

ufo_shanks_nb = {r'\B3205-231031':[5,8],
                 r'\B0714-230221':[3,0],
                 r'\A3024-220915A':[99,99]} # channel number for signal with UFO and without (control) 

lfp_channels = {r'\B3205-231031':[29,61,83]}

fs=20000

SI_thr = {
    'AD':0.0, 
    'LMN':0.0,
    'MMB':0.0,
    'PoSub':0.0,
    'MEC':0.0,
    'TRN':0.0
    }


for r in datasets.keys():
    tmp_datasets=[item for item in datasets[r] if item != 'HIP']
    ccs_long = {s:{e:[] for e in ['wake', 'rem', 'sws']} for s in tmp_datasets}
    ccs_short = {s:{e:[] for e in ['wake', 'rem', 'sws']} for s in tmp_datasets}
    print('######## '+r+' ########')
    ### Load session data
    path = data_directory + r
    data = ntm.load_session(path, 'neurosuite')

    if r==r'\B0714-230221':
        ttl_filename= path + r + '_1.csv'
        position=parse_ttl_csv(ttl_filename)

        ep_sample=nap.IntervalSet(data.epochs['wake'][0])
    else:
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

    ufo_ep, ufo_ts = loadUFOsV2(path)

    ### For saving figure
    save_fig_path=r'D:\PeyracheLab Internship\Figure'+ r
    os.makedirs(save_fig_path, exist_ok=True)

############################################################################################### 
# UFOs DETECTIONS
###############################################################################################
    if ufo_ts is None:
        data.load_neurosuite_xml(data.path)
        channels = data.group_to_channel
        nb_channels=data.nChannels
        sign_shanks = channels[ufo_shanks_nb[r][0]]
        ctrl_shanks = channels[ufo_shanks_nb[r][1]]
        filename = data.basename + ".dat"    
        fp, timestep = get_memory_map(os.path.join(data.path, filename), nb_channels)
        clu = np.genfromtxt(os.path.join(path, r.split("\\")[-1]+".clu."+str(ufo_shanks_nb[r][0]+1)), dtype="int")[1:]
        res = np.genfromtxt(os.path.join(path, r.split("\\")[-1]+".res."+str(ufo_shanks_nb[r][0]+1)))
        #ufo_ep, ufo_ts = detect_ufos_v2(fp, sign_shanks, ctrl_shanks, timestep)
        ufo_ep, ufo_ts = detect_ufos_v4(fp, sign_shanks, ctrl_shanks, timestep,clu,res, ep=None)

        # Save in .evt file for Neuroscope
        start = ufo_ep.as_units('ms')['start'].values
        peaks = ufo_ts.as_units('ms').index.values
        ends = ufo_ep.as_units('ms')['end'].values

        datatowrite = np.vstack((start,peaks,ends)).T.flatten()

        n = len(ufo_ep)

        texttowrite = np.vstack(((np.repeat(np.array(['UFO start 1']), n)),
                                (np.repeat(np.array(['UFO peak 1']), n)),
                                (np.repeat(np.array(['UFO stop 1']), n))
                                    )).T.flatten()

        evt_file = os.path.join(path, data.basename + '.evt.py.ufo')
        f = open(evt_file, 'w')
        for t, n in zip(datatowrite, texttowrite):
            f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
        f.close()


    for s in tmp_datasets:
        print('######## '+s+' ########')
        spikes = data.spikes
        idx = spikes._metadata[spikes._metadata["location"].str.contains(s)].index.values
        spikes = spikes[idx]
############################################################################################### 
# CC COMPUTING
###############################################################################################
        print('CC computing...')

        #tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], 120, minmax=(0, 2*np.pi), ep = position.time_support.loc[[0]])
        #tuning_curves = smoothAngularTuningCurves(tuning_curves, 20, 4)

        #SI = nap.compute_1d_mutual_info(tuning_curves, position['ry'], position['ry'].time_support.loc[[0]], minmax=(0,2*np.pi))

        #spikes = spikes[SI[SI['SI']>SI_thr[r]].index.values]
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

    print('#######')
############################################################################################### 
# HEAD-DIRECTIONS CELLS COMPUTING
###############################################################################################
    print('HD computing...')
    
    spikes = data.spikes

    mean_wf, max_ch=data.load_mean_waveforms()

    # Open a file for writing. The 'wb' argument denotes 'write binary'
    with open(os.path.join(path, data.basename + '_mean_wf.pkl'), 'wb') as file:
        pickle.dump(mean_wf, file)

    with open(os.path.join(path, data.basename + '_max_ch.pkl'), 'wb') as file:
        pickle.dump(max_ch, file)

    angle = position['ry']
    ttl_ep=nap.IntervalSet(start=position.time_support['start'][0], end=position.time_support['end'][0])

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
    
    for e, ep in zip(['wake', 'rem', 'sws'], [wake_ep, rem_ep, sws_ep]): 

        ### Tuning curves computations
        tuning_curves = nap.compute_1d_tuning_curves(group=spikes, 
                                                    feature=angle, 
                                                    nb_bins=num_bins, 
                                                    ep = ep,
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
        width_per_subplot = 4
        height_per_subplot = 4

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
        plt.suptitle(f"Recording: {r} - Ep: {e}")
        plt.tight_layout()
        plt.savefig(save_fig_path+r'\\'+str(e)+'_hd.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_fig_path+r'\\'+str(e)+'_hd.svg', bbox_inches='tight')
        plt.close()

############################################################################################### 
# HIP CORRELETATION
###############################################################################################

    if 'HIP' in datasets[r]:
        path = data_directory + r
        data = ntm.load_session(path, 'neurosuite')
        ufo_ep, ufo_ts = loadUFOsV2(path)
        ufo_ts_start=ufo_ep['start']
        ufo_ts_start=nap.Ts(ufo_ts_start)
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        plt.figure(figsize=(15, 8))
        for index, (i, s) in enumerate(zip(lfp_channels[r], datasets[r])):
            print('######## LFP-'+ s +' ########')
            lfp = data.load_lfp(channel=i, extension='.dat', frequency=20000)
            peth = nap.compute_perievent_continuous(lfp, ufo_ts_start, (-0.1, 0.1), ep=None, time_unit="s")
            peth=peth.as_dataframe()
            z_peth=zscore(peth.dropna(axis=1, how='any'))
            mean_peth=z_peth.mean(axis=1)
            error_peth=z_peth.sem(axis=1)
            
            # Select color based on index
            color = colors[index % len(colors)]  # Use modulo to cycle through colors
            
            plt.plot(mean_peth, color=color, label=s)
            plt.fill_between(
                mean_peth.index.values,
                mean_peth.values - error_peth,
                mean_peth.values + error_peth,
                color=color,
                alpha=0.2,
            )
        
        plt.axvline(0, linewidth=2, color="k", linestyle="--")  # Plot a line at t = 0
        plt.xlabel("Time from boundary (s)")
        plt.ylabel("Z score")  
        plt.legend(loc="upper right")
        plt.title(f"Recording: {r}")
        plt.savefig(save_fig_path+r'\hip_corr.svg', bbox_inches='tight')
        plt.savefig(save_fig_path+r'\hip_corr.png',dpi=300, bbox_inches='tight')
        plt.close()

    print(ufo_ep)
    plt.show()