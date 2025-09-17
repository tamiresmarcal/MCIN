## Env

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
import gc
from joblib import Parallel, delayed
from tqdm import tqdm

## Functions

def process_sliding_windows_dfc(img_original, sub, movie, start, step, end, overlap_factor = 4):
    """ Dynamic Functional connectivity with slide windows """
    list_df = []
    for t in tqdm(range(start, end, step), desc=f' Processing dFC sub {sub} {movie}'):
        for i in range(overlap_factor): 
            # update the time for windows overlap
            t = round(t+i*step/overlap_factor) 
            if t+step <= end:
                # slice img
                img = img_original.dataobj[:,:,:,t:t+step]
                img = nib.Nifti1Image(img, img_original.affine, img_original.header)
                # get fc of the slice
                correlation_matrix = extract_fc(img)
                # format dataframe
                df = pd.DataFrame(correlation_matrix)
                df['node'] = df.index
                df['sub'] = sub
                df['movie'] = movie
                df['start'] = t #start do intervalo
                df['step'] = step
                list_df.append(df)
                # Clear intermediate variables to free memory
                del img, correlation_matrix, df
                gc.collect()
    # save
    final_df = pd.concat(list_df, ignore_index=True)
    return final_df


def load_image(sub, movie):
    path = os.path.join(f'/home/tamires/projects/rpp-aevans-ab/tamires/data/fmri_datasets/ds002837/derivatives/sub-{sub}/func', 
                        f'sub-{sub}_task-{movie}_bold_blur_censor_ica.nii.gz')
    return nib.load(path, mmap=True)


def extract_fc(fmri_img):
    # Use a predefined atlas from Nilearn (e.g., Harvard-Oxford atlas)
    atlas = datasets.fetch_atlas_basc_multiscale_2015(resolution=122,version='asym')
    atlas_img = atlas.maps
    # Define the masker to extract time series from ROIs and get time series
    masker = input_data.NiftiLabelsMasker(labels_img=atlas_img, standardize=True)
    time_series = masker.fit_transform(fmri_img)
    # Compute the correlation matrix
    correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    # Clear intermediate variables to free memory
    del masker, time_series, correlation_measure
    gc.collect()
    return correlation_matrix


def save_data(df, sub, movie): 
    # file name
    base_filename = f'/home/tamires/projects/rpp-aevans-ab/tamires/data/fmri_derived/dfcs/sub-{sub}_task-{movie}'
    version = 1
    filename = f"{base_filename}_v{version}.parquet"
    # check if there is a previous version, duplicate if there is a previous version
    while os.path.exists(filename):
        version += 1
        filename = f"{base_filename}_v{version}.parquet"
    # save file
    df.to_parquet(filename)
    

def process_participant(sub, movie, start, step, end):
    """
    start: onde vai comecar a ser processado
    step: os intervalos/janelas de processamento
    end: onde para de processar
    """
    # Load: 153 ms
    print(f"Initializing processing participant {sub} | {movie} | Estimated processing time: {round(16*4*((end-start)/step)/60,0)} minutes")
    img_original = load_image(sub, movie)
    # Process: 10s per interval per interval
    dfc = process_sliding_windows_dfc(img_original, sub, movie, start, step, end)
    # Save: 600ms
    save_data(dfc, sub, movie)
    print(f"File saved participant {sub} | {movie}")
    # Clear all variables to free memory after saving
    del img_original, dfc
    gc.collect()
    print(f"Memory cleared for participant {sub} | {movie}")

# Data
participants = pd.read_csv("/home/tamires/projects/rpp-aevans-ab/tamires/data/fmri_datasets/ds002837/participants.tsv", sep='\t')
participants['sub'] = range(1,87)
participants = participants.rename(columns={'task':'movie'})
participants = participants[participants['sub'] != 49] # ta corrompido
participants['end_movie'] = [load_image(sub=row['sub'], movie=row['movie']).shape[3] for index, row in participants.iterrows()]

# Parameters
participants_test = participants.loc[[36,68,85]]
start = 0
step = 300

# Set the number of jobs (parallel workers)
n_jobs = 10  # Adjust this number based on your system's capacity
os.environ["OMP_NUM_THREADS"] = "1"  # Ensure thread limiting if needed

# Process
results = Parallel(n_jobs=n_jobs)(
    delayed(process_participant)(sub, movie, start, step, end) 
    for sub, movie, end in participants_test[['sub', 'movie','end_movie']].values
)