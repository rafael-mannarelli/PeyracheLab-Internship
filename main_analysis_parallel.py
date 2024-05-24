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
import _pickle as cPickle  

# Local application/library specific imports
import UFOphysio.python as phy
from UFOphysio.python.functions import *
from UFOphysio.python.ufo_detection import *

# Configuration settings
plt.rcParams.update({"axes.spines.right": False, "axes.spines.top": False})
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5)



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
            'DTN3':r'\A4004-200317b'}
            #'DTN4':r'\A4007-200801'}

ufo_channels_nb = {'LMN':[1,0],
            'MMN':[1,0],
            'PostSUB':[1,0],
            'DTN1':[3,1], 
            'DTN2':[3,1],
            'DTN3':[1,0],
            'DTN4':[1,3]} # channel number for signal with UFO and without (control) 

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

############################################################################################### 
# HEAD-DIRECTIONS CELLS COMPUTING
###############################################################################################
    if 'DTN' in location:
        tracking_data = pd.read_csv(path + r'\Analysis\Tracking_data.csv', header=None)
        tracking_data.iloc[:, -3:] = (np.radians(tracking_data.iloc[:, -3:]) % (2 * np.pi))
        position = nap.TsdFrame(t=tracking_data.iloc[:,0].values,
                                d=tracking_data.iloc[:,-3:].values,
                                time_support=nap.IntervalSet(start=min(tracking_data.iloc[:,0]), 
                                                            end=max(tracking_data.iloc[:,0])),
                                columns=['ry','rz','rx'])
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

    for k in spikes:
        tuning_curves[k]=signal.medfilt(tuning_curves[k],3)

    pref_ang = tuning_curves.idxmax()
    norm = Normalize()  # Normalizes data into the range [0,1]
    color = plt.cm.hsv(norm([i / (2 * np.pi) for i in pref_ang.values]))  # Assigns a colour in the HSV colourmap for each value of preferred angle
    color = pd.DataFrame(index=pref_ang.index, data = color, columns = ['r', 'g', 'b', 'a'])

    ### Smooth curves computations
    smoothcurves = smoothAngularTuningCurvesHD(tuning_curves, sigma=3)


    ### Plotting
    num_curves = len(pref_ang.sort_values().index.values)
    num_columns = 4
    num_rows = math.ceil(num_curves / num_columns)

    # Size per subplot (width, height) in inches
    width_per_subplot = 2
    height_per_subplot = 1

    # Total figure size
    total_width = width_per_subplot * num_columns
    total_height = height_per_subplot * num_rows

    if plot==True:
        plt.figure(figsize=(total_width, total_height))
        for i, n in enumerate(pref_ang.sort_values().index.values):
            plt.subplot(num_rows, num_columns, i + 1, projection='polar')
            plt.plot(smoothcurves[n], color=color.loc[n])
            plt.plot(tuning_curves[n], color='k', alpha=0.7)
            plt.plot([pref_ang[n], pref_ang[n]], [0, max(max(smoothcurves[n]), max(tuning_curves[n]))], 'r--')  # 'r--' makes the line red and dashed
            plt.title(s + '-' + str(n))  # Assume 's' is your titles dictionary
            plt.xlabel("Angle (rad)")
            plt.ylabel("Firing Rate (Hz)")
            plt.xticks([])
        plt.tight_layout()
        plt.show()

    ### HD cells identification
    hd_labels = {}  # Initialize a dictionary to store HD cell labels

    for i in tuning_curves.keys():
        # Calculate the variability of magnitudes in smoothcurves[i] to determine uniformity
        variability = np.std(smoothcurves[i]) / np.mean(smoothcurves[i]) if np.mean(smoothcurves[i]) > 0 else 0

        # Use a threshold to decide if the magnitudes are almost the same (low variability indicates no particular main direction)
        variability_threshold = 0.2  # This threshold is adjustable based on your dataset
        
        if variability < variability_threshold:
            #print(f"{i}: No particular main direction due to low variability in magnitudes.")
            hd_labels[i] = 0  # Label as non-HD cell
            continue 

        magnitudes = max(tuning_curves[i])
        
        # Main direction and magnitude of the resultant vector
        main_direction = pref_ang[i]
        main_magnitude = magnitudes
        
        # Define a minimum amplitude threshold
        min_amplitude = 5
        
        # Check for conditions
        if main_magnitude < 1e-5 or main_magnitude < min_amplitude:
            #print(f"{i}: Not an HD cell due to low magnitude.")
            hd_labels[i] = 0  # Label as non-HD cell
        else:
            #print(f"{i}: HD cell with main direction {np.degrees(main_direction)} degrees and magnitude {main_magnitude}.")
            hd_labels[i] = 1  # Label as HD cell
    print('HD Labels:',hd_labels)

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