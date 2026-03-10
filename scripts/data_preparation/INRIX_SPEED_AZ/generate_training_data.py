import json
import os
import shutil

import numpy as np
import pandas as pd

# Hyperparameters
dataset_name = 'INRIX_SPEED_AZ'
data_file_path = f'datasets/raw_data/{dataset_name}/processed/processed.csv'
graph_file_path = f'datasets/raw_data/{dataset_name}/processed/adj_mx.pkl'
output_dir = f'datasets/{dataset_name}'
target_channel = [0]  # Target traffic flow channel
add_time_of_day = True  # Add time of day as a feature
add_day_of_week = True  # Add day of the week as a feature
add_day_of_month = False  # Add day of the month as a feature
add_day_of_year = False  # Add day of the year as a feature
add_month_of_year = True # Add month of the year as a feature
steps_per_day = 288  # Number of time steps per day
frequency = 1440 // steps_per_day # data points every 5 min
domain = 'traffic speed'
feature_description = [domain, 'time of day', 'day of week', 'month of year']
regular_settings = {
    'INPUT_LEN': 12,
    'OUTPUT_LEN': 12,
    'TRAIN_VAL_TEST_RATIO': [0.8, 0.1, 0.1],
    'NORM_EACH_CHANNEL': False,
    'RESCALE': True,
    'METRICS': ['MAE', 'RMSE', 'MAPE'],
    'NULL_VAL': 0.0
}

def load_and_preprocess_data():
    '''Load and preprocess raw data, selecting the specified channel(s).'''
    df = pd.read_csv(data_file_path, index_col=0, parse_dates=True)
    # sort all the columns by name to ensure the same order as the graph file
    # df = df.reindex(sorted(df.columns), axis=1)
    
    data = np.expand_dims(df.values, axis=-1)
    data = data[..., target_channel]
    print(f'Raw time series shape: {data.shape}')
    return data, df

def add_temporal_features(data, df):
    '''Add time of day and day of week as features to the data.'''
    _, n, _ = data.shape
    feature_list = [data]

    if add_time_of_day:
        time_of_day = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        time_of_day_tiled = np.tile(time_of_day, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(time_of_day_tiled)

    if add_day_of_week:
        day_of_week = df.index.dayofweek / 7
        day_of_week_tiled = np.tile(day_of_week, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(day_of_week_tiled)

    if add_day_of_month:
        # numerical day_of_month
        day_of_month = (df.index.day - 1 ) / 31 # df.index.day starts from 1. We need to minus 1 to make it start from 0.
        day_of_month_tiled = np.tile(day_of_month, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(day_of_month_tiled)

    if add_day_of_year:
        # numerical day_of_year
        day_of_year = (df.index.dayofyear - 1) / 366 # df.index.month starts from 1. We need to minus 1 to make it start from 0.
        day_of_year_tiled = np.tile(day_of_year, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(day_of_year_tiled)
    
    if add_month_of_year:
    # numerical month_of_year
        month_of_year = (df.index.month - 1) / 13
        month_of_year_tiled = np.tile(month_of_year, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(month_of_year_tiled)

    data_with_features = np.concatenate(feature_list, axis=-1)  # L x N x C
    return data_with_features

def save_data(data):
    '''Save the preprocessed data to a binary file.'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, 'data.dat')
    fp = np.memmap(file_path, dtype='float32', mode='w+', shape=data.shape)
    fp[:] = data[:]
    fp.flush()
    del fp
    print(f'Data saved to {file_path}')

def save_timestamps(df):
    '''Save timestamp index to CSV for calendar-aware splits.'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp_path = os.path.join(output_dir, 'timestamps.csv')
    df.index.to_series().dt.strftime('%Y-%m-%d %H:%M:%S').to_csv(
        timestamp_path, index=False, header=['timestamp']
    )
    print(f'Timestamps saved to {timestamp_path}')

def save_graph():
    '''Save the adjacency matrix to the output directory.'''
    output_graph_path = os.path.join(output_dir, 'adj_mx.pkl')
    shutil.copyfile(graph_file_path, output_graph_path)
    print(f'Adjacency matrix saved to {output_graph_path}')

def save_description(data, df):
    '''Save a description of the dataset to a JSON file.'''
    description = {
        'name': dataset_name,
        'domain': domain,
        'shape': data.shape,
        'num_time_steps': data.shape[0],
        'num_nodes': data.shape[1],
        'num_features': data.shape[2],
        'feature_description': feature_description,
        'has_graph': graph_file_path is not None,
        'frequency (minutes)': frequency,
        'timestamp_file': 'timestamps.csv',
        'start_datetime': df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
        'end_datetime': df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
        'regular_settings': regular_settings
    }
    description_path = os.path.join(output_dir, 'desc.json')
    with open(description_path, 'w') as f:
        json.dump(description, f, indent=4)
    print(f'Description saved to {description_path}')
    print(description)

def main():
    # Load and preprocess data
    data, df = load_and_preprocess_data()

    # Add temporal features
    data_with_features = add_temporal_features(data, df)

    # Save timestamps for downstream calendar-based datasets
    save_timestamps(df)

    # Save processed data
    save_data(data_with_features)

    # Copy and save adjacency matrix
    save_graph()

    # Save dataset description
    save_description(data_with_features, df)

if __name__ == '__main__':
    main()
