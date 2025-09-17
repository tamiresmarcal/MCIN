# Env
import nibabel as nib
from nilearn import datasets, input_data, plotting, connectome, image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, zscore
import pandas as pd
import os
import json
import glob
import time
import gc
import time
from joblib import Parallel, delayed

# Functions
def load_image(sub, movie):
    path = os.path.join(f'/home/tamires/projects/rpp-aevans-ab/tamires/data/fmri_datasets/ds002837/derivatives/sub-{sub}/func', 
                        f'sub-{sub}_task-{movie}_bold_blur_no_censor_ica.nii.gz')
    return nib.load(path, mmap=True)


def save_correlation_data(networks_data, sub, movie): 
    # format dataframe
    df = pd.DataFrame(networks_data, columns=['network', 'start', 'end', 
                                              'net_mean', 'net_median','net_std','net_mean_abs','net_median_abs','net_std_abs',
                                              'fc_mean_abs','fc_median_abs','fc_std_abs'])
    df['sub'] = sub
    df['movie'] = movie
    
    # file name
    base_filename = f'/home/tamires/projects/rpp-aevans-ab/tamires/data/fmri_derived/networks_data/sub-{sub}_task-{movie}_networks'
    version = 1
    filename = f"{base_filename}_v{version}.parquet"
    
    # check if there is a previous version
    while os.path.exists(filename):
        version += 1
        filename = f"{base_filename}_v{version}.parquet"
        
    df.to_parquet(filename)


def brain_networks_extraction(df_networks, network_name, fmri_img, radius = 6):
    # Coordinates
    peak_coords = df_networks[df_networks.name == network_name][['x', 'y', 'z']].values
    # Create 6mm spheres around these coordinates
    spheres_masker = input_data.NiftiSpheresMasker(seeds=peak_coords, radius=radius, standardize=True)
    # Extract time series data for each sphere
    time_series = spheres_masker.fit_transform(fmri_img) 
    return time_series


def network_statistic(time_series):
    net_stats = [np.mean(time_series), 
                 np.median(time_series), 
                 np.std(time_series),
                 np.mean(abs(time_series)), 
                 np.median(abs(time_series)), 
                 np.std(abs(time_series))]
    return net_stats
    

def functional_conectivity_statistic(time_series):
    # Compute the functional connectome using ConnectivityMeasure
    correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    fc_stats = [np.mean(abs(correlation_matrix)), 
                np.median(abs(correlation_matrix)), 
                np.std(abs(correlation_matrix))]
    return fc_stats


def process_time_intervals(img_original, network_list, start, end, step, radius=2):
    """ Extract data of interval using brain_networks_extraction, network_statistic and functional_conectivity_statistic """
    
    networks_data = []
    for t in range(start, end, step):
        img = img_original.dataobj[:,:,:,t:t+step]
        img = nib.Nifti1Image(img, img_original.affine, img_original.header)
        
        for network_name in network_list:
            time_series = brain_networks_extraction(df_networks, network_name=network_name, fmri_img=img, radius=radius)
            net_stats = network_statistic(time_series)
            fc_stats = functional_conectivity_statistic(time_series)
            networks_data.append([network_name, t, t+step] + net_stats + fc_stats) 
            #print(f"{network_name} | clip start: {t} | clip end: {t+step}")
    
    return networks_data


def process_participant(sub, movie, network_list, end, duration, step):
    print(f"Initializing processing participant {sub} | {movie}")
    
    # Load: 153 ms
    img_original = load_image(sub, movie)
    
    # Process: 2s per interval per network
    start = end - duration
    networks_data = process_time_intervals(img_original, network_list, start=start, end=end, step=step)
    
    # Save: 600ms
    save_correlation_data(networks_data, sub, movie)
    print(f"File saved participant {sub} | {movie}")


# Data
df_networks = pd.read_csv("/home/tamires/projects/rpp-aevans-ab/tamires/data/fmri_derived/mni_space_of_networks.csv")
participants = pd.read_csv("/home/tamires/projects/rpp-aevans-ab/tamires/data/fmri_datasets/ds002837/participants.tsv", sep='\t')
participants['sub'] = range(1,87)
participants = participants[participants['sub'] != 49] # ta corrompido
participants['end'] = [load_image(sub=row['sub'], movie=row['task']).shape[3] for index, row in participants.iterrows()]


# Parameters
participants_test = participants.iloc[40:]
network_list = df_networks.name.unique()
duration = 30 * 60
step = 10

# Set the number of jobs (parallel workers)
n_jobs = 30  # Adjust this number based on your system's capacity
os.environ["OMP_NUM_THREADS"] = "1"  # Ensure thread limiting if needed
results = Parallel(n_jobs=n_jobs)(
    delayed(process_participant)(sub, movie, network_list, (end-(30*60)), duration, step)
    for sub, movie, end in participants_test[['sub', 'task','end']].values
)
