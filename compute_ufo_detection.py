import os
import nwbmatic as ntm
import pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
import pynapple as nap 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import Isomap
from scipy.signal import welch
from sklearn.decomposition import PCA
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.lines import Line2D



# Local application/library specific imports
import UFOphysio.python as phy
from UFOphysio.python.functions import *
from UFOphysio.python.ufo_detection import *

##############################################################################################
# FUNCTIONS
##############################################################################################
def zscore(x):
    """Return the z-score of the provided array."""
    return (x - x.mean(axis=0)) / x.std(axis=0)

def save_ufo_events(ufo_ep, ufo_ts, path, basename, file_name):
    """
    Saves UFO events into a Neuroscope-compatible .evt file.

    Parameters:
    - ufo_ep: DataFrame containing the 'start' and 'end' columns for UFO events, indexed in seconds.
    - ufo_ts: DataFrame or Series containing the peak timestamps of UFO events, indexed in seconds.
    - path: Path where the .evt file will be saved.
    - basename: Base name for the .evt file.
    """
    # Convert start, peak, and end times to milliseconds
    start = ufo_ep.as_units('ms')['start'].values
    peaks = ufo_ts.as_units('ms').index.values
    ends = ufo_ep.as_units('ms')['end'].values

    # Prepare the data and text to write into the file
    datatowrite = np.vstack((start, peaks, ends)).T.flatten()
    n = len(ufo_ep)
    texttowrite = np.vstack((np.repeat(['UFO start 1'], n),
                             np.repeat(['UFO peak 1'], n),
                             np.repeat(['UFO stop 1'], n))).T.flatten()

    # Determine the file path and open the file
    evt_file = os.path.join(path, basename + '.evt.py.'+ file_name)
    with open(evt_file, 'w') as f:
        for t, n in zip(datatowrite, texttowrite):
            f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")

def plot_ufo_event(event_idx, sign_channels, fp, timestep, ufo_ep,frequency):
    plt.figure(figsize=(15, 8))
    event = ufo_ep[event_idx]
    start, end = event['start'], event['end']
    start_idx = int((start-0.01)*frequency)
    end_idx = int((end+0.01)*frequency)

    for c in sign_channels:
        signal = zscore(fp[start_idx:end_idx, c])
        times = timestep[start_idx:end_idx]
        plt.plot(times, signal, label=f'Channel {c}')

    plt.title(f'UFO Event {event_idx} from {start} to {end} seconds')
    plt.xlabel('Time (seconds)')
    plt.ylabel('LFP Amplitude')
    plt.legend()
    plt.grid(True)

def on_button_clicked(b):
    global current_event
    if b.description == 'Keep':
        keep_indices.append(current_event)
    elif b.description == 'Delete':
        delete_indices.append(current_event)
    elif b.description == 'Stop':
        global running
        running = False
        return
    
    current_event += 1
    if current_event < len(ufo_ep) and running:
        clear_output(wait=True)
        plot_ufo_event(current_event, sign_channels, fp, timestep, ufo_ep, fs)
        display(button_box)
    else:
        clear_output(wait=True)
        print("Review complete or stopped.")


def next_previous_button(b):
    global current_event
    if b.description == 'Next':
        current_event += 1
    elif b.description == 'Previous':
        current_event -= 1
    elif b.description == 'Stop':
        global running
        running = False
        return

    if current_event < len(ufo_ep) and running:
        clear_output(wait=True)
        plot_ufo_event(current_event, sign_channels, fp, timestep, ufo_ep, fs)
        display(button_box)
    else:
        clear_output(wait=True)
        print("Review complete or stopped.")



class LineDrawer:
    def __init__(self, scatter_plot):
        self.scatter_plot = scatter_plot
        self.ax = scatter_plot.axes
        self.fig = self.ax.figure
        self.line = Line2D([], [], color='red', lw=2)
        self.ax.add_line(self.line)
        self.xs, self.ys = [], []
        self.lines = []
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.cluster_labels = np.zeros(len(scatter_plot.get_offsets()), dtype=int)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'enter':
            self.lines.append((self.xs.copy(), self.ys.copy()))
            self.xs.clear()
            self.ys.clear()
            self.line.set_data(self.xs, self.ys)
            self.fig.canvas.draw()
            self.cluster_data()

    def cluster_data(self):
        all_data = self.scatter_plot.get_offsets()

        for idx, (x_line, y_line) in enumerate(self.lines, start=1):
            path = np.array([x_line, y_line]).T
            is_above = np.ones(len(self.cluster_labels), dtype=bool)

            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]

                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
                intercept = y1 - slope * x1

                if slope != float('inf'):
                    is_above &= all_data[:, 1] >= slope * all_data[:, 0] + intercept
                else:
                    is_above &= all_data[:, 0] >= x1

            self.cluster_labels[is_above] = idx

        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for idx in range(1, max(self.cluster_labels) + 1):
            self.ax.scatter(all_data[self.cluster_labels == idx, 0], all_data[self.cluster_labels == idx, 1], color=colors[(idx - 1) % len(colors)], label=f'Cluster {idx}')

        self.ax.legend()
        self.fig.canvas.draw()

    def get_cluster_labels(self):
        """ Returns the cluster labels similar to k-means """
        return self.cluster_labels
############################################################################################### 
# GENERAL INFOS
###############################################################################################
label_type=['analysis']
cluster_type=['manual']
test_n_neighbors=False

#data_directory = r'E:\Data PeyracheLab'
data_directory = r'D:\PeyracheLab Internship\Data'
datasets = {#r'\B0714-230221':['AD', 'MEC']
            #r'\B0703-211129':['AD','MEC']}
            r'\A4002-200120b':['DTN']}
#            r'\B0702-211111':['AD','MEC']}
ufo_shanks_nb = {r'\B0703-211129':[2,0],
                 r'\B0702-211111':[2,0],
                 r'\B0714-230221':[3,0],
                 r'\A4002-200120b':[3,1]}

fs=20000

for r in datasets.keys():
    path = data_directory + r
    data = ntm.load_session(path, 'neurosuite')
    ufo_ep, ufo_ts = loadUFOsV2(path,'ufo')

    data.load_neurosuite_xml(data.path)
    channels = data.group_to_channel
    nb_channels=data.nChannels
    sign_shanks = channels[ufo_shanks_nb[r][0]]
    ctrl_shanks = channels[ufo_shanks_nb[r][1]]
    filename = data.basename + ".dat"    
    fp, timestep = get_memory_map(os.path.join(data.path, filename), nb_channels)
    
    if ufo_ts is None:
        ufo_ep, ufo_ts = detect_ufos_v2(fp, sign_shanks, ctrl_shanks, timestep)
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

    if 'manual' in label_type:
        sign_channels = sign_shanks
        current_event = 0
        delete_indices = []
        keep_indices = []
        running = True

        keep_button = widgets.Button(description="Keep")
        delete_button = widgets.Button(description="Delete")
        stop_button = widgets.Button(description="Stop")
        keep_button.on_click(on_button_clicked)
        delete_button.on_click(on_button_clicked)
        stop_button.on_click(on_button_clicked)

        button_box = widgets.HBox([keep_button, delete_button, stop_button])
        plot_ufo_event(current_event, sign_channels, fp, timestep, ufo_ep,fs)
        display(button_box)

    if 'analysis' in label_type:
        sign_shanks = channels[ufo_shanks_nb[r][0]]
        c=32 #c=27
        signal={}
        times={}
        for i in ufo_ep.index:
            event = ufo_ep[i]
            start, end = event['start'], event['end']
            start_idx = int((start-0.02)*fs)
            end_idx = int((end+0.02)*fs)
            signal[i] = zscore(fp[start_idx:end_idx, c])
            times[i] = timestep[start_idx:end_idx]
        min_length = min(len(s) for s in signal.values())
        for i in signal:
            signal_length = len(signal[i])
            if signal_length > min_length:
                extra = signal_length - min_length
                start_trim = extra // 2
                end_trim = extra - start_trim
                signal[i] = signal[i][start_trim: signal_length - end_trim]
                times[i] = times[i][start_trim: signal_length - end_trim]
        
        signal_df=pd.DataFrame.from_dict(signal)

        freqs_signal={}
        psd_signal={}
        for i in signal_df.columns:
            freqs, psd = welch(signal_df[i], fs, nperseg=1024)
            freqs_signal[i]=freqs
            psd_signal[i]=psd

        freqs_signal_df=pd.DataFrame.from_dict(freqs_signal)
        psd_signal_df=pd.DataFrame.from_dict(psd_signal)

        features=psd_signal_df

        if test_n_neighbors==True:
            # Range of neighbors to evaluate
            n_neighbors_range = range(2, 100)
            residual_variances = []

            # Compute Isomap residual variances
            for n_neighbors in n_neighbors_range:
                isomap = Isomap(n_components=2, n_neighbors=n_neighbors)
                isomap.fit(features)
                residual_variance = 1-isomap.reconstruction_error()
                residual_variances.append(residual_variance)

            # Plot the residual variances
            plt.figure(figsize=(8, 6))
            plt.plot(n_neighbors_range, residual_variances, marker='o')
            plt.xlabel('Number of Neighbors')
            plt.ylabel('Residual Variance')
            plt.title('Residual Variance vs. Number of Neighbors')
            plt.grid(True)
        
        # Apply Isomap
        isomap = Isomap(n_components=2, n_neighbors=10)
        features_isomap = isomap.fit_transform(features.T)

        # Plot the Isomap results
        plt.figure(figsize=(10, 7))
        plt.scatter(features_isomap[:, 0], features_isomap[:, 1], c='blue', label='UFO Episodes')
        plt.xlabel('Isomap Dimension 1')
        plt.ylabel('Isomap Dimension 2')
        plt.title('Isomap Visualization of LFP Features')
        plt.legend()

        # Apply PCA
        pca = PCA(n_components=2)  # Adjust the number of components as required
        features_pca = pca.fit_transform(features.T)
        
        # Plot the PCA results
        plt.figure(figsize=(10, 7))
        plt.scatter(features_pca[:, 0], features_pca[:, 1])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('LFP PCA: First Two Principal Components')

        features_to_cluster=features_pca

        if 'manual' in cluster_type:
            # Scatter plot
            fig, ax = plt.subplots()
            scatter = ax.scatter(features_pca[:, 0], features_pca[:, 1])

            # Initialize the LineDrawer class
            line_drawer = LineDrawer(scatter)
            plt.show()

            clusters = line_drawer.get_cluster_labels()
            print("Cluster Labels:", clusters)
        else:
            kmeans = KMeans(n_clusters=2, random_state=42)
            clusters = kmeans.fit_predict(features_to_cluster)

        # Calculate the silhouette score to evaluate clustering quality
        silhouette_avg = silhouette_score(features_to_cluster, clusters)
        print(f"Silhouette Score: {silhouette_avg:.2f}")

        # Visualize the Isomap embedding with clusters
        plt.figure(figsize=(10, 7))
        for cluster in np.unique(clusters):
            mask = clusters == cluster
            plt.scatter(features_to_cluster[mask, 0], features_to_cluster[mask, 1], label=f'Cluster {cluster}')

        plt.xlabel('Isomap Dimension 1')
        plt.ylabel('Isomap Dimension 2')
        plt.title('Isomap Visualization of LFP Features with Clusters')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        win_size=0.05
        clusters_of_interest = np.where(clusters == 0)[0]
        print('Length clusters:',len(clusters_of_interest))
        ufo_ts_cluster = [ufo_ep['start'][i] for i in clusters_of_interest]
        ufo_ts_cluster=nap.Ts(ufo_ts_cluster)
        spikes=data.spikes

        cc = nap.compute_eventcorrelogram(spikes, ufo_ts_cluster, 0.001, win_size, norm=False)
        cc=zscore(cc)

        num_bins = cc.shape[1]
        bin_height = 1
        plt.figure(figsize=(15, 8))
        im = plt.imshow(cc.T.fillna(0), aspect='auto', cmap='turbo', interpolation='none', extent=[-win_size, win_size, 0, cc.shape[1]], origin='lower')        
        plt.colorbar(label='Normalized firing rate')
        plt.axvline(0, color='white', linestyle='--') 

        tick_positions = [x + bin_height / 2 for x in range(0, num_bins, max(1, num_bins // 10))]
        tick_step = max(1, num_bins // 10)
        tick_labels = cc.keys()[::tick_step]
        plt.yticks(tick_positions, tick_labels)
        plt.xlabel('Time from UFO event (s)')
        plt.ylabel('Cell index')
        plt.title(f"Recording: {r}")

        clusters_of_interest = np.where(clusters == 1)[0]
        print('Length clusters:',len(clusters_of_interest))
        ufo_ts_cluster = [ufo_ep['start'][i] for i in clusters_of_interest]
        ufo_ts_cluster=nap.Ts(ufo_ts_cluster)

        cc = nap.compute_eventcorrelogram(spikes, ufo_ts_cluster, 0.001, win_size, norm=False)
        cc=zscore(cc)

        num_bins = cc.shape[1]
        bin_height = 1
        plt.figure(figsize=(15, 8))
        im = plt.imshow(cc.T.fillna(0), aspect='auto', cmap='turbo', interpolation='none', extent=[-win_size, win_size, 0, cc.shape[1]], origin='lower')        
        plt.colorbar(label='Normalized firing rate')
        plt.axvline(0, color='white', linestyle='--') 

        tick_positions = [x + bin_height / 2 for x in range(0, num_bins, max(1, num_bins // 10))]
        tick_step = max(1, num_bins // 10)
        tick_labels = cc.keys()[::tick_step]
        plt.yticks(tick_positions, tick_labels)
        plt.xlabel('Time from UFO event (s)')
        plt.ylabel('Cell index')
        plt.title(f"Recording: {r}")
        plt.show()
