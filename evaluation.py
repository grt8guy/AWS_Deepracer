"""evaluation module used to score contestants

All submissions will be placed inside evaluation/compressed/, then we will run this script.

# To view a contestant:
>>> with open(eval_results / 'results.json', 'r') as json_file:
>>>     results = json.load(json_file)

>>> from IPython.display import display
>>> display(pd.DataFrame(results['firstname-lastname']['stats']))
>>> display(pd.DataFrame(results['firstname-lastname']['metrics']))
"""

# Intra-Project Imports
import utils

# Apply sklearn intel acceleration patch.
import platform
if 'Intel' in platform.processor():
    if utils.is_package_installed('scikit-learn-intelex'):
        from sklearnex import patch_sklearn
        patch_sklearn()

# Standard Library Imports
import copy
import json
import math
import tarfile
from pathlib import Path

# 3rd Party Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


# Get absolute path to evaluation dir
eval_compressed = Path("__file__").parent.resolve() / "evaluation" / "compressed"
eval_expanded = Path("__file__").parent.resolve() / "evaluation" / "expanded"
eval_results = Path("__file__").parent.resolve() / "evaluation" / "results"

# Define the track name for the analysis
track_name='2024_american_hills_speedway'

# Define the number of episodes per iteration
episode_per_iter = 20

# Get the waypoints for the specified track
waypoints = utils.get_track_waypoints(track_name)

# extract track boundaries from waypoints
center_line = waypoints[:,0:2] 
inner_border = waypoints[:,2:4]
outer_border = waypoints[:,4:6]

# Walk through eval compressed directory
for entry in tqdm(eval_compressed.iterdir(), total=len(list(eval_compressed.iterdir())), desc='expansion'):
    if entry.is_file() and entry.suffix == '.gz':
        # Extract all tars
        with tarfile.open(entry.absolute(), 'r:gz') as tar:
            # Extract all contents to the specified directory
            tar.extractall(path=eval_expanded / entry.stem.strip('.tar'))

# Walk through eval expanded directory
# Initialize data dictionary to store metrics
data = {}
for entry in tqdm(eval_expanded.iterdir(), total=len(list(eval_expanded.iterdir())), desc='processing results'):
    if entry.is_dir():

        # Grab contestant name
        contestant_name = entry.name.strip('-submission')

        # Find and open the model metadata JSON file
        with open(next(entry.rglob('*model_metadata.json')).as_posix()) as json_file:
            model_metadata = json.load(json_file)
        algorithm = model_metadata['training_algorithm']

        with open(next(entry.rglob('*evaluation-*.json')).as_posix()) as json_file:
            evaluation_json = json.load(json_file)

        # Flag to handle the first pass
        initial_pass = True
        entry_names = []

        data[entry.name.strip('-submission')] = {}

        # Get all simulation trace CSV files
        sim_trace_csvs = [next(entry.rglob('*0-iteration.csv')).as_posix()]

        # Path for merged simulation trace file
        merged_simtrace_path = eval_expanded / entry.name / "merged_simtrace.csv"
        utils.merge_csv_files(merged_simtrace_path, sim_trace_csvs)

        # Read the merged simulation trace CSV into a DataFrame
        df = pd.read_csv(merged_simtrace_path)
        
        # Calculate iteration array based on the maximum episode
        iteration_arr = np.arange(math.ceil(df.episode.max() / episode_per_iter)+ 1) * episode_per_iter
        df['iteration'] = np.digitize(df.episode, iteration_arr)
        df = df.rename(columns={'X': 'x', 'Y': 'y', 'tstamp': 'timestamp'})

        # Convert continuous data to discrete buckets
        _, df = utils.continuous_to_discrete(copy.copy(model_metadata), df, num_angle_buckets=5, num_speed_buckets=4)            

        # Scale the reward values using MinMaxScaler
        min_max_scaler = MinMaxScaler()
        scaled_vals = min_max_scaler.fit_transform(df['reward'].values.reshape(df['reward'].values.shape[0], 1))
        df['reward'] = pd.DataFrame(scaled_vals.squeeze())

        # Parse episodes and actions from the DataFrame
        action_dict, episode_dict, sorted_episodes = utils.episode_parser(df)
        sorted_episodes.sort()

        # Collect metrics for each episode:
        #   - Distance to center statistics
        stats = [utils.get_distance_to_center_stats(episode_dict[episode], center_line) for episode in sorted_episodes]
        #   - Reward
        for episode in sorted_episodes:
            stats[episode]['reward'] = df[df['episode'] == episode]['reward'].sum()
        
        # Store the collected metrics in the data dictionary
        data[entry.name.strip('-submission')] = {'stats': stats, 'metrics': evaluation_json['metrics']}

with open(eval_results / 'results.json', 'w+') as json_file:
    json.dump(data, json_file)