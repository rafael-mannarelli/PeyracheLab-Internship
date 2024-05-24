import os
import nwbmatic as ntm
import pickle as pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import Isomap
from scipy.signal import welch
from sklearn.decomposition import PCA


# Local application/library specific imports
import UFOphysio.python as phy
from UFOphysio.python.functions import *
from UFOphysio.python.ufo_detection import *




import jax.numpy as jnp
from jax import jit
from jax.scipy.signal import stft as jax_stft
import matplotlib.pyplot as plt
import pynapple as nap
import numpy as np

@jit
def spectrogram(signal, fs, n_fft=2048, hop_length=512, window='hann', time_units='s'):
    """
    Computes the spectrogram of a signal using the JAX library for performance efficiency.

    Parameters
    ----------
    signal : jax.numpy.array
        The input signal from which to compute the spectrogram.
    fs : int
        The sampling rate of the signal in Hz.
    n_fft : int, optional
        The number of points used in the FFT window.
    hop_length : int, optional
        The number of samples between successive frames.
    window : str, optional
        The type of window function (default is 'hann').
    time_units : str, optional
        The time units of the hop_length and n_fft parameters ('s' for seconds [default], 'ms' for milliseconds, 'us' for microseconds).

    Returns
    -------
    tuple
        A tuple containing the frequency bins array, time bins array, and the STFT magnitude in dB as a JAX numpy array.

    Raises
    ------
    ValueError
        If the time_units parameter is not one of 's', 'ms', 'us'.
    """
    if time_units not in ['s', 'ms', 'us']:
        raise ValueError("time_units must be 's', 'ms', or 'us'.")

    # Compute STFT
    f, t, Zxx = jax_stft(signal, fs=fs, window=window, nperseg=n_fft, noverlap=n_fft-hop_length)
    
    # Convert to decibels
    Zxx_dB = 20 * jnp.log10(jnp.abs(Zxx))
    return (f, t, Zxx_dB)


def compute_spectrogram(fp, timestep, channel, frequency=20000):
    """
    Computes the spectrograms for specified channels within the signal data, leveraging the JAX backend for efficient computation.

    Parameters
    ----------
    fp : numpy.memmap
        The signal data from which the spectrogram is computed, typically a memory-mapped file for handling large datasets.
    timestep : numpy.array
        The time indices corresponding to the signal data, used to set the timestamps for the spectrogram.
    channel : int
        The specific channel index to compute the spectrogram for. This allows for selective computation on multi-channel data.
    frequency : int, optional
        The sampling rate of the signal in Hz. Default is 20000 Hz.

    Returns
    -------
    list of tuples
        A list of tuples where each tuple contains (frequency bins, time bins, STFT magnitude in dB) for each computed batch of the signal.

    """

    nap.nap_config.set_backend("jax")

    batch_size = frequency * 500

    # Calculate starting points for each batch
    starts = np.arange(0, len(timestep), batch_size)
    spectrograms = []

    total_batches = len(starts)  # Total number of batches to process
    print(f"Total batches to process: {total_batches}")

    cumulative_duration = 0  # Keep track of the cumulative duration to adjust time bins
    
    for idx, s in enumerate(starts):
        print(f"Processing batch {idx+1}/{total_batches}...")
        # Define the batch for the current channel
        batch_lfp = nap.Tsd(t=timestep[s:s+batch_size], d = jnp.array(fp[s:s+batch_size,channel][:]))

        # Compute the STFT using JAX's stft function
        f, t, Zxx_dB = spectrogram(batch_lfp.d, fs=frequency)
        # Append the result as a tuple
        spectrograms.append((f, t, Zxx_dB))
    
    f = spectrograms[0][0]

    # Initialize empty lists for collecting time bins and dB magnitudes
    cumulative_time = 0
    all_t = []
    all_Zxx_dB = []

    # Accumulate time bins and magnitude arrays from each batch
    for _, t_batch, Zxx_dB_batch in spectrograms:
        adjusted_t_batch = t_batch + cumulative_time  # Adjust time batch
        all_t.append(adjusted_t_batch)
        all_Zxx_dB.append(Zxx_dB_batch)
        cumulative_time += t_batch[-1] + (t_batch[1] - t_batch[0])  # Update cumulative time

    # Concatenate all time bins and dB magnitudes along the appropriate axes
    all_t = jnp.concatenate(all_t)
    all_Zxx_dB = jnp.concatenate(all_Zxx_dB, axis=1)  # Concatenate along the time axis

    return (f,all_t,all_Zxx_dB)

def plot_spectrogram(f,t,Zxx_dB, start_time=None, end_time=None):
    """
    Plots a combined spectrogram from a list of spectrogram data tuples using JAX arrays, 
    optionally limiting the plot to a specified time interval.

    Parameters
    ----------
    spectrogram_data : list of tuples
        Each tuple contains (frequency bins, time bins, STFT magnitude in dB) from the compute_spectrogram function.
        This list represents multiple batches of spectrogram computations.
    start_time : float, optional
        The start time for the interval of the spectrogram to be plotted. If None, plotting starts from the beginning.
    end_time : float, optional
        The end time for the interval of the spectrogram to be plotted. If None, plotting goes until the end.

    Returns
    -------
    None
        The function creates and shows a plot but does not return any value.
    """

    # Determine indices for the specified time interval
    if start_time is not None and end_time is not None:
        time_mask = (t >= start_time) & (t <= end_time)
        t_plot = t[time_mask]
        Zxx_dB_plot = Zxx_dB[:, time_mask]
    else:
        t_plot = t
        Zxx_dB_plot = Zxx_dB

    # Plotting the selected interval of the combined spectrogram
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t_plot, f, Zxx_dB_plot, shading='gouraud')
    plt.title('Combined Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    plt.show()



import numpy as np
import jax.numpy as jnp

data_directory = r'E:\Data PeyracheLab'
datasets = {r'/A5026/A5026-210725A':['lmn','thl']}
            #r'/A5026/A5026-210727A':['lmn','thl']}

fs=20000

for r in datasets.keys():
    path = data_directory + r
    data = ntm.load_session(path, 'neurosuite')
    ufo_ep, ufo_ts = loadUFOsV2(path,'ufo')

    data.load_neurosuite_xml(data.path)
    channels = data.group_to_channel
    nb_channels=data.nChannels
    filename = data.basename + ".dat"    
    fp, timestep = get_memory_map(os.path.join(data.path, filename), nb_channels)

    f,t,Zxx_dB=compute_spectrogram(fp, timestep, channels[1][1], frequency=20000)
    plot_spectrogram(f,t,Zxx_dB,start_time=7000, end_time=7010)