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

def whm_angles_distance(whm_angles,pref_angle):
    start_angle, end_angle = whm_angles    
    # Normalize angles to [0, 2*pi)
    start_angle %= 2 * np.pi
    end_angle %= 2 * np.pi
    pref_angle %= 2 * np.pi

    # Determine the direction and calculate angular distance accordingly
    if start_angle < end_angle:
        if start_angle < pref_angle < end_angle:
            # Clockwise direction
            angular_distance = np.abs(end_angle - start_angle)
        else:
            # Counter-Clockwise direction
            angular_distance = np.abs(end_angle - start_angle + 2 * np.pi)
    else:
        if end_angle < pref_angle < start_angle:
            # Counter-Clockwise direction
            angular_distance = np.abs(end_angle + 2 * np.pi - start_angle) 
        else:
            # Clockwise direction
            angular_distance = np.abs(end_angle - start_angle) 

    whm_distance=min(2*np.pi-angular_distance,angular_distance)
    if whm_distance<0:
        whm_distance=whm_distance+2*np.pi
    return np.degrees(whm_distance)

def find_closest_whm_angles(tuning_curve, pref_angle):
    pref_value = tuning_curve[pref_angle]
    half_pref_value = pref_value / 2

    diff = np.abs(tuning_curve - half_pref_value)
    sorted_diff=diff.sort_values()

    whm_angle=[]

    for idx in sorted_diff.index:
        if not whm_angle:
            whm_angle.append(idx)
        elif np.abs(whm_angle[0] - idx) >= 1 :
            whm_angle.append(idx)
            break

    return sorted(whm_angle)


############################################################################################### 
# GENERAL INFOS
###############################################################################################
plot=False
data_directory = r'D:\PeyracheLab Internship\Data'
datasets = {#'LMN':r'\A3024-220915A',
            #'MMN':r'\A3024-220915A',
            #'PostSUB':r'\A3024-220915A',
            'DTN1':r'\A4002-200120b',
            'DTN2':r'\A4002-200121',
            'DTN3':r'\A4004-200317b',
            'DTN4':r'\A4007-200801'}

ufo_shanks_nb = {'LMN':[1,0],
            'MMN':[1,0],
            'PostSUB':[1,0],
            'DTN1':[3,1], 
            'DTN2':[3,1],
            'DTN3':[1,0],
            'DTN4':[4,1]} # channel number for signal with UFO and without (control) 
ctrl_channels={'DTN1':1,'DTN2':1, 'DTN3':1, 'DTN4':1}

fs=20000


for s in datasets.keys():
    print('######## '+s+' ########')
    ### Load session data
    path = data_directory + datasets[s]
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(data.path)
    channels = data.group_to_channel

    wake_ep = data.epochs['wake']
    sleep_ep = data.epochs['sleep']
    
    spikes = data.spikes
    location=''.join([char for char in s if char.isalpha()])
    spikes = spikes.getby_category("location")[location]

    ### For saving figure
    save_fig_path=r'D:\PeyracheLab Internship\Figure'+datasets[s]
    os.makedirs(save_fig_path, exist_ok=True)

############################################################################################### 
# HEAD-DIRECTIONS CELLS COMPUTING
###############################################################################################
    if 'DTN' in location:
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

    angle = position['ry']
    ttl_epochs=nap.IntervalSet(start=position.time_support['start'][0], end=position.time_support['end'][0])
    
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

    ### Tuning curves computations
    tuning_curves = nap.compute_1d_tuning_curves(group=spikes, 
                                                feature=angle, 
                                                nb_bins=num_bins, 
                                                ep = ttl_epochs,
                                                minmax=(0, 2 * np.pi))
    
    bin_size = 0.5
    ang_velocity = computeAngularVelocity(angle, ttl_epochs, bin_size)
    tuning_curves_angv = nap.compute_1d_tuning_curves(group=spikes, 
                                            feature=ang_velocity, 
                                            nb_bins=num_bins, 
                                            ep = ttl_epochs,
                                            minmax=(-np.pi, np.pi))
    
    #from scipy.stats import skew

    cell_sym={}
    sym_labels={}
    for _,n in enumerate(tuning_curves_angv):
        tmp=tuning_curves_angv[n]
        tuning_curves_angv[n]=tmp[tmp != 0]
        tmp=tuning_curves_angv[n].dropna()
        tmp = (tmp - tmp.mean()) / tmp.std()
        cell_sym[n]=np.mean(tmp[(tmp.index > 0) & (tmp.index < 1)].values) - np.mean(tmp[(tmp.index < 0) & (tmp.index > -1)].values)
        if cell_sym[n] < 0.15 and cell_sym[n] > -0.15:
            sym_labels[n]=1
        else:
            sym_labels[n]=0
    print('Cells Symetry Labels:',sym_labels)

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
    if plot==False:
        plt.figure(figsize=(total_width, total_height))
        for i, n in enumerate(spikes.keys()):
            plt.subplot(num_rows, num_columns, i + 1)
            plt.plot(tuning_curves_angv[n].dropna(), 'o',color=color.loc[n])
            plt.plot(tuning_curves_angv[n].rolling(window=4).mean().dropna(), color='red')
            plt.title(s + '-' + str(n) + '- Cell Symetry:' + str(cell_sym[n])) 
            plt.xlabel("Angular Head Velocity (rad/sec)")
            plt.ylabel("Firing Rate (Hz)")
        plt.tight_layout()
        plt.savefig(save_fig_path+r'\ahv.png', dpi=300, bbox_inches='tight')
        plt.close()
        

    for k in spikes:
        tuning_curves[k]=signal.medfilt(tuning_curves[k],3)

    pref_ang = tuning_curves.idxmax()
    norm = Normalize()  # Normalizes data into the range [0,1]
    color = plt.cm.hsv(norm([i / (2 * np.pi) for i in pref_ang.values]))  # Assigns a colour in the HSV colourmap for each value of preferred angle
    color = pd.DataFrame(index=pref_ang.index, data = color, columns = ['r', 'g', 'b', 'a'])

    ### Smooth curves computations
    smoothcurves = smoothAngularTuningCurvesHD(tuning_curves, sigma=3)

    ### HD cells identification
    hd_labels = {}  # Initialize a dictionary to store HD cell labels
    hd_whm = {}
    hd_polar_distance = {}
    hd_type = {}

    for i in tuning_curves.keys():
        # Calculate the variability of magnitudes in smoothcurves[i] to determine uniformity
        variability = np.std(smoothcurves[i]) / np.mean(smoothcurves[i]) if np.mean(smoothcurves[i]) > 0 else 0

        # Use a threshold to decide if the magnitudes are almost the same (low variability indicates no particular main direction)
        variability_threshold = 0.2  # This threshold is adjustable based on your dataset
        
        if variability < variability_threshold:
            #print(f"{i}: No particular main direction due to low variability in magnitudes.")
            hd_labels[i] = 0  # Label as non-HD cell
            hd_type[i] = 0
            continue 

        magnitudes = max(tuning_curves[i])
        
        # Main direction and magnitude of the resultant vector
        main_direction = pref_ang[i]
        main_magnitude = magnitudes
        
        # Define a minimum amplitude threshold
        min_amplitude = 1
        
        # Check for conditions
        if main_magnitude < 1e-5 or main_magnitude < min_amplitude:
            #print(f"{i}: Not an HD cell due to low magnitude.")
            hd_labels[i] = 0  # Label as non-HD cell
            hd_type[i] = 0
        else:
            #print(f"{i}: HD cell with main direction {np.degrees(main_direction)} degrees and magnitude {main_magnitude}.")
            hd_labels[i] = 1 
            hd_whm[i] = find_closest_whm_angles(smoothcurves[i], main_direction)
            hd_polar_distance[i]=whm_angles_distance(hd_whm[i],main_direction)
            if hd_polar_distance[i] > 150:
                hd_type[i] = 2
            else:
                hd_type[i] = 1
        
    print('HD Labels:',hd_labels)
    print('HD Type:',hd_type)
    print('HD WHM:', hd_whm)
    print('HD WHM Angular Distance:', hd_polar_distance)

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
    if plot==True:
        plt.figure(figsize=(total_width, total_height))
        for i, n in enumerate(spikes.keys()):
            plt.subplot(num_rows, num_columns, i + 1, projection='polar')
            plt.plot(smoothcurves[n], color=color.loc[n])
            plt.plot(tuning_curves[n], color='k', alpha=0.7)
            plt.plot([pref_ang[n], pref_ang[n]], [0, max(max(smoothcurves[n]), max(tuning_curves[n]))], 'r--')  # 'r--' makes the line red and dashed
            if n in hd_labels and hd_labels[n] == 1:
                whm_angles = hd_whm[n]  # Get the WHM angles
                colors_whm = ['g', 'b','y']  # List of colors to cycle through
                for idx, angle in enumerate(whm_angles):
                    whm_firing_rate = smoothcurves[n][angle]
                    color_whm = colors_whm[idx % len(colors_whm)]  # Cycle through the colors list
                    plt.plot(angle, whm_firing_rate, color_whm + 'o')  # Plot with the chosen color
                
                start_angle, end_angle = whm_angles
                start_angle %= 2 * np.pi
                end_angle %= 2 * np.pi
                pref_angle = pref_ang[n] % (2 * np.pi)

                # Find amplitude at start_angle
                start_angle_amplitude = smoothcurves[n].loc[start_angle]

                # Determine the direction for interpolation
                if start_angle < end_angle:
                    if start_angle < pref_angle < end_angle:
                        interpolated_angles = np.linspace(start_angle, end_angle, num=100)
                    else:
                        interpolated_angles = np.linspace(end_angle, start_angle + 2 * np.pi, num=100)
                else:
                    if end_angle < pref_angle < start_angle:
                        interpolated_angles = np.linspace(start_angle, end_angle + 2 * np.pi, num=100)
                    else:
                        interpolated_angles = np.linspace(end_angle, start_angle, num=100)

                interpolated_angles = interpolated_angles % (2 * np.pi)
                
                # Set amplitude to that of the start_angle for all points, checking not to exceed smoothcurves' max
                interpolated_amplitudes = np.full_like(interpolated_angles, start_angle_amplitude)
                max_amplitude = smoothcurves[n].max()
                interpolated_amplitudes = np.minimum(interpolated_amplitudes, max_amplitude)

                plt.plot(interpolated_angles, interpolated_amplitudes, 'm-', linewidth=2)

            plt.title(s + '-' + str(n))  # Assume 's' is your titles dictionary
            plt.xlabel("Angle (rad)")
            plt.ylabel("Firing Rate (Hz)")
        plt.tight_layout()
        plt.savefig(save_fig_path+r'\hd.png', dpi=300, bbox_inches='tight')
        plt.close()

    ### HD group

    hd_group={}

    for i in hd_labels.keys():
        if hd_labels[i]==1:
            hd_group[i]=spikes.get_info('group')[i]

    print('HD Group:',hd_group)
    hd_group_list=np.unique(list(hd_group.values()))
    group_to_delete=[]

    for i in channels.keys():
        if i not in hd_group_list:
            group_to_delete.append(list(channels[i]))
    group_to_delete=[item for sublist in group_to_delete for item in sublist]
    
    
    txt_writing = "[" + ", ".join(str(item) for item in group_to_delete) + "]"
    with open(path +'\\'+ data.basename+ r'_group_to_delete.txt', 'w') as file:
        file.write(txt_writing)

    spikes.set_info(HD=pd.Series(hd_labels))
    spikes.set_info(HD_type=pd.Series(hd_type))
    spikes_hd=spikes.getby_category('HD')[1]

############################################################################################### 
# UFOs DETECTIONS
###############################################################################################
    ufo_ep, ufo_tsd = loadUFOsV2(path)

    if ufo_tsd is None:
        #spikes_hd=spikes.getby_category('HD')[1]
        #tuning_curves = nap.compute_1d_tuning_curves(spikes_hd, position['ry'], 120, minmax=(0, 2*np.pi), ep=position.time_support.loc[[0]])
        data.load_neurosuite_xml(data.path)
        channels = data.group_to_channel
        sign_shanks = channels[ufo_shanks_nb[s][0]]
        ctrl_shanks = channels[ufo_shanks_nb[s][1]]
        filename = data.basename + ".dat"    
        #clu = np.genfromtxt(os.path.join(path, datasets[s].split("\\")[-1]+".clu."+str(ufo_shanks_nb[s][0])), dtype="int")[1:]
        #res = np.genfromtxt(os.path.join(path, datasets[s].split("\\")[-1]+".res."+str(ufo_shanks_nb[s][0])))
        fp, timestep = get_memory_map(os.path.join(data.path, filename), data.nChannels)
        ufo_ep, ufo_tsd = detect_ufos_v2(fp, sign_shanks, ctrl_shanks, timestep)

        # Save in .evt file for Neuroscope
        start = ufo_ep.as_units('ms')['start'].values
        peaks = ufo_tsd.as_units('ms').index.values
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
############################################################################################### 
# RASTER COMPUTING
###############################################################################################     
    ufo_peth={}
    counts_ufo={}
    fr_ufo={}
    meanfr_ufo={}
    error_ufo={}

    min_max=(-0.25,0.25)
    bin_size = 0.01  # 200ms bin size
    step_size = 0.001  # 10ms step size, to make overlapping bins
    winsize = int(bin_size / step_size)  # Window size
    
    for _,j in enumerate(spikes_hd):
        ufo_peth[j]=nap.compute_perievent(spikes_hd[j],ufo_tsd,minmax=min_max)
        counts_ufo[j] = ufo_peth[j].count(step_size, time_units='s')
        counts_ufo[j] = (
            counts_ufo[j].as_dataframe()
            .rolling(winsize, win_type="gaussian", min_periods=1, center=True, axis=0)
            .mean(std=0.2 * winsize)
        )
        fr_ufo[j] = (counts_ufo[j] * winsize)/bin_size
        meanfr_ufo[j] = fr_ufo[j].mean(axis=1)
        error_ufo[j] = fr_ufo[j].sem(axis=1)

    ctrl_peth=nap.compute_perievent(spikes[ctrl_channels[s]],ufo_tsd,minmax=min_max)
    counts_ctrl = ctrl_peth.count(step_size, time_units='s')
    counts_ctrl = (
        counts_ctrl.as_dataframe()
        .rolling(winsize, win_type="gaussian", min_periods=1, center=True, axis=0)
        .mean(std=0.2 * winsize)
    )
    fr_ctrl = (counts_ctrl * winsize)/bin_size
    meanfr_ctrl = fr_ctrl.mean(axis=1)
    error_ctrl = fr_ctrl.sem(axis=1)

    def z_score_normalize(series):
        mean_val = series.mean(axis=0)
        std_val = series.std(axis=0)
        #std_val[std_val==0]=1
        z_scores = (series - mean_val) / std_val
        return z_scores

    # Normalize the firing rates using Z-score normalization
    z_meanfr_ufo = {j: z_score_normalize(meanfr) for j, meanfr in meanfr_ufo.items()}
    z_meanfr_ctrl = z_score_normalize(meanfr_ctrl)


    if plot==True:
        num_subplots = len(spikes_hd) + 1  # One for each cell in spikes_hd, plus one for ctrl_peth

        # Parameters for dynamic figure sizing
        height_per_subplot = 2 # Adjust based on desired height for each subplot
        width = 20  # Fixed width for the figure
        total_height = height_per_subplot * num_subplots  # Total height based on the number of subplots

        plt.figure(figsize=(width, total_height))
        # Loop through spikes_hd to plot each cell's PETH
        for i,(index, peth) in enumerate(ufo_peth.items()):
            for j, n in enumerate(peth):
                plt.subplot(num_subplots, 1, i+1)  # Dynamic subplot positioning
                plt.plot(
                    peth[n].as_units("s").fillna(j),  # Assuming you can convert to units of seconds; adjust as necessary
                    "|",
                    color='blue',
                    markersize=4,
                )
                plt.axvline(0, linewidth=1, color="k", linestyle="--")
                plt.xlabel("Time from UFO event (s)")
                plt.ylabel("UFO Events")  # Label with cell index or identifier
                plt.title((f"Cell {index}"))

        # Adding the ctrl_peth in the last subplot position
        for j, n in enumerate(ctrl_peth):
            plt.subplot(num_subplots, 1, num_subplots)
            plt.plot(
                ctrl_peth[n].as_units("s").fillna(j),  # Adjust as necessary
                "|",
                color='green',
                markersize=4,
            )
            plt.axvline(0, linewidth=1, color="k", linestyle="--")
            plt.xlabel("Time from event (s)")
            plt.ylabel("UFO Events")  # Label with cell index or identifier
            plt.title((f"Control cell {ctrl_channels[s]}"))
        plt.subplots_adjust(hspace=1.5)
        plt.suptitle(f"Recording: {s}{datasets[s]}")
        plt.savefig(save_fig_path+r'\raster.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(15, 8))

        cmap = plt.cm.get_cmap('jet', len(meanfr_ufo))
        colors = [cmap(i) for i in range(len(meanfr_ufo))]

        for (index, meanfr), color in zip(meanfr_ufo.items(), colors):
            # Plot mean firing rate
            plt.plot(meanfr, color=color, label=f"Cell {index}")
            
            # Plot SEM
            plt.fill_between(
                meanfr.index.values,
                meanfr.values - error_ufo[index],
                meanfr.values + error_ufo[index],
                color=color,
                alpha=0.2,
            )

        # Plot mean firing rate
        plt.plot(meanfr_ctrl, color='green', label="Control Cell")
        
        # Plot SEM
        plt.fill_between(
            meanfr_ctrl.index.values,
            meanfr_ctrl.values - error_ctrl,
            meanfr_ctrl.values + error_ctrl,
            color=color,
            alpha=0.2,
        )

        plt.axvline(0, linewidth=2, color="k", linestyle="--")
        plt.xlabel("Time from UFO event (s)")
        plt.ylabel("Firing rate (Hz)")
        plt.legend(loc="upper right")
        plt.title(f"Recording: {s}{datasets[s]}")
        plt.savefig(save_fig_path+r'\firing_rate.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(15, 8))

        # Generate colors dynamically based on the number of items
        cmap = plt.cm.get_cmap('jet', len(z_meanfr_ufo))
        colors = [cmap(i) for i in range(len(z_meanfr_ufo))]

        # Plot each normalized mean firing rate with SEM
        for (index, z_meanfr), color in zip(z_meanfr_ufo.items(), colors):
            plt.plot(z_meanfr.index.values, z_meanfr.values, color=color, label=f"Cell {index}")
            
            #plt.fill_between(
            #    z_meanfr.index.values,
            #    z_meanfr.values - error_ufo[index],
            #    z_meanfr.values + error_ufo[index],
            #    color=color,
            #    alpha=0.2,
            #)

        plt.plot(z_meanfr_ctrl, color='green', label="Control Cell")
        
        #plt.fill_between(
        #    z_meanfr_ctrl.index.values,
        #    z_meanfr_ctrl.values - error_ctrl,
        #    z_meanfr_ctrl.values + error_ctrl,
        #    color=color,
        #    alpha=0.2,
        #)

        plt.axvline(0, linewidth=2, color="k", linestyle="--")
        plt.xlabel("Time from UFO event (s)")
        plt.ylabel("Z-score normalized firing rate")
        plt.legend(loc="upper right")
        plt.title(f"Recording: {s}{datasets[s]}")
        plt.savefig(save_fig_path+r'\firing_rate_z_score.png', dpi=300, bbox_inches='tight')
        plt.close()

############################################################################################### 
# PETH COMPUTING
###############################################################################################    

    group_boundaries = []
    spike_groups=spikes.get_info('group')
    for i in range(1, len(spike_groups)):
        if spike_groups[i] != spike_groups[i-1]:
            group_boundaries.append(i-1)  
    cc = nap.compute_eventcorrelogram(spikes, ufo_tsd, 0.001, 0.1, None, norm=False)
    z_cc=z_score_normalize(cc)
    z_ccs=z_cc.loc[-0.1:0.1]

    cc_hd = nap.compute_eventcorrelogram(spikes_hd, ufo_tsd, 0.001, 0.1, None, norm=False)
    z_cc_hd=z_score_normalize(cc_hd)
    z_ccs_hd=z_cc_hd.loc[-0.1:0.1]
    sorted_hd_types = spikes_hd.get_info('HD_type').sort_values()
    sorted_indices = sorted_hd_types.index
    z_ccs_hd_sorted = z_ccs_hd.loc[:, sorted_indices]
    #limit_index = sorted_hd_types[sorted_hd_types == 1].index[-1]

    print("Spikes group: ", np.unique(spike_groups.values))
    print("Sorted HD Type:", sorted_hd_types)

    if plot==True:
        plt.figure(figsize=(15, 8))
        im = plt.imshow(z_ccs.T.fillna(0), aspect='auto', cmap='jet', interpolation='none', extent=[-0.1, 0.1, 0, z_ccs.shape[1]])        
        plt.colorbar(label='Normalized firing rate')
        plt.axvline(0, color='white', linestyle='--') 
        for boundary in group_boundaries:
            plt.axhline(y=boundary, color='red', linestyle='--')
        plt.yticks(range(0, int(z_ccs.shape[1])+1, max(1, int(z_ccs.shape[1]) // 10)))
        plt.xlabel('Time from UFO event (s)')
        plt.ylabel('Cell index')
        plt.title(f"Recording: {s}{datasets[s]}")
        plt.savefig(save_fig_path+r'\cross_corr_z_score.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(15, 8))
        im = plt.imshow(z_ccs_hd_sorted.T.fillna(0), aspect='auto', cmap='jet', interpolation='none', extent=[-0.1, 0.1, 0, z_ccs_hd_sorted.shape[1]])
        plt.colorbar(label='Normalized firing rate')
        plt.axvline(0, color='white', linestyle='--')
        # You may adjust the group boundaries visualization as needed
        plt.xlabel('Time from UFO event (s)')
        plt.ylabel('Cell index sorted by HD type')
        plt.yticks(ticks=range(len(sorted_hd_types.index)),labels=sorted_hd_types.index)
        #plt.axhline(y=sorted_hd_types.index.get_loc(limit_index), color='red', linestyle='--')
        plt.title(f"Recording: {s}{datasets[s]}")
        plt.savefig(save_fig_path+r'\firing_rate_hd_sort_z_score.png', dpi=300, bbox_inches='tight')
        plt.close()

############################################################################################### 
# VELOCITY COMPUTING
###############################################################################################    


plt.show()


