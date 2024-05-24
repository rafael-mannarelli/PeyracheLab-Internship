import os
import nwbmatic as ntm
import pickle as pickle  

data_directory = r'E:\Data PeyracheLab'
datasets = {r'\B0714-230221':['AD', 'MEC'],
            r'\B0703-211129':['AD','MEC'],
            r'\B0702-211111':['AD','MEC']}


for r in datasets.keys():
    path = data_directory + r
    data = ntm.load_session(path, 'neurosuite')

    mean_wf, max_ch=data.load_mean_waveforms()

    # Open a file for writing. The 'wb' argument denotes 'write binary'
    with open(os.path.join(path, data.basename + '_mean_wf.pkl'), 'wb') as file:
        pickle.dump(mean_wf, file)

    with open(os.path.join(path, data.basename + '_max_ch.pkl'), 'wb') as file:
        pickle.dump(max_ch, file)