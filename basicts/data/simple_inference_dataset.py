import json
import logging
import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset


def tile_feature(values: Union[pd.Index, np.ndarray], num_nodes: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.tile(values, [1, num_nodes, 1]).transpose((2, 1, 0))


def get_weather_feature(df: pd.DataFrame, num_nodes: int, description: dict, dataset_name: str) -> np.ndarray:
    weather_categories = description.get('weather_categories', [])
    if not weather_categories or not dataset_name:
        return None

    weather_file_path = f'datasets/raw_data/{dataset_name}/processed/weather_condition.csv'
    if not os.path.exists(weather_file_path):
        raise FileNotFoundError(f'Weather file not found: {weather_file_path}')

    weather_df = pd.read_csv(weather_file_path)
    if 'timestamp' not in weather_df.columns or 'weather_condition' not in weather_df.columns:
        raise ValueError('Weather file must contain timestamp and weather_condition columns.')

    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], utc=True).dt.tz_localize(None)
    weather_df = weather_df.drop_duplicates(subset='timestamp', keep='last').set_index('timestamp')
    weather_df = weather_df.reindex(df.index)
    weather_df['weather_condition'] = weather_df['weather_condition'].ffill().bfill()

    if weather_df['weather_condition'].isna().any():
        raise ValueError('Weather data could not be filled after timestamp alignment.')

    weather_codes = pd.Categorical(
        weather_df['weather_condition'],
        categories=weather_categories
    )
    weather_one_hot = pd.get_dummies(weather_codes, dtype=np.float32).to_numpy()
    return np.tile(weather_one_hot[:, None, :], (1, num_nodes, 1))


def get_incident_feature(
    df: pd.DataFrame, num_nodes: int, node_columns: Union[pd.Index, None], description: dict, dataset_name: str
) -> np.ndarray:
    incident_levels = description.get('incident_levels', [])
    if not incident_levels or not dataset_name:
        return None

    incident_file_path = f'datasets/raw_data/{dataset_name}/processed/incidents.csv'
    if not os.path.exists(incident_file_path):
        raise FileNotFoundError(f'Incident file not found: {incident_file_path}')

    if node_columns is None or len(node_columns) != num_nodes:
        raise ValueError('Incident features require node-aligned inference columns.')

    incident_df = pd.read_csv(incident_file_path, index_col=0, parse_dates=True)
    incident_df.index = pd.to_datetime(incident_df.index, utc=True).tz_localize(None)

    if incident_df.shape[1] != len(node_columns):
        raise ValueError('Incident columns must exactly match inference node columns.')

    incident_df = incident_df.reindex(df.index).fillna(0)
    incident_values = incident_df.to_numpy(dtype=np.int8)
    
    unknown_levels = sorted(set(np.unique(incident_values)) - {0, *incident_levels})
    if unknown_levels:
        raise ValueError(f'Unknown incident levels found: {unknown_levels}')

    return np.stack(
        [(incident_values == level).astype(np.float32) for level in incident_levels],
        axis=-1
    )


def _build_features(data: np.ndarray, df: pd.DataFrame, description: dict, dataset_name: str) -> np.ndarray:
    _, num_nodes, _ = data.shape
    feature_description = description.get('feature_description', [])

    # Fall back to the original simple temporal layout when description metadata is unavailable.
    if not feature_description:
        feature_description = ['value', 'time of day', 'day of week', 'day of month', 'day of year']

    feature_map = {
        'time of day': tile_feature((df.index.hour * 60 + df.index.minute) / (24 * 60), num_nodes),
        'day of week': tile_feature(df.index.dayofweek / 7, num_nodes),
        'day of month': tile_feature((df.index.day - 1) / 31, num_nodes),
        'day of year': tile_feature((df.index.dayofyear - 1) / 366, num_nodes),
        'month of year': tile_feature((df.index.month - 1) / 13, num_nodes),
    }
    has_weather_features = any(name.startswith('weather: ') for name in feature_description)
    has_incident_features = any(name.startswith('incident level: ') for name in feature_description)

    weather_feature = (
        get_weather_feature(df, num_nodes, description, dataset_name)
        if has_weather_features else None
    )
    node_columns = df.columns if len(df.columns) == num_nodes else None
    incident_feature = (
        get_incident_feature(df, num_nodes, node_columns, description, dataset_name)
        if has_incident_features else None
    )

    feature_list = [data]
    weather_offset = 0
    incident_offset = 0

    for feature_name in feature_description[1:]:
        if feature_name in feature_map:
            feature_list.append(feature_map[feature_name])
        elif feature_name.startswith('weather: ') and weather_feature is not None:
            feature_list.append(weather_feature[..., weather_offset:weather_offset + 1])
            weather_offset += 1
        elif feature_name.startswith('incident level: ') and incident_feature is not None:
            feature_list.append(incident_feature[..., incident_offset:incident_offset + 1])
            incident_offset += 1

    data_with_features = np.concatenate(feature_list, axis=-1).astype('float32')
    data_set_shape = description['shape']
    return data_with_features[..., range(data_set_shape[2])]


class TimeSeriesInferenceDataset(BaseDataset):
    """
    A dataset class for time series inference tasks, where the input is a sequence of historical data points
    
    Attributes:
        description_file_path (str): Path to the JSON file containing the description of the dataset.
        description (dict): Metadata about the dataset, such as shape and other properties.
        data (np.ndarray): The loaded time series data array.
        raw_data (str): The raw data path or data list of the dataset.
        last_datetime (pd.Timestamp): The last datetime in the dataset. Used to generate time features of future data.
    """

    # pylint: disable=unused-argument
    def __init__(self, dataset_name:str, dataset: Union[str, list], input_len: int, output_len: int, memmap: bool = False,
                 logger: logging.Logger = None, **kwargs) -> None:
        """
        Initializes the TimeSeriesInferenceDataset by setting up paths, loading data, and 
        preparing it according to the specified configurations.

        Args:
            dataset_name (str): The name of the dataset. If dataset_name is None, the dataset is expected to be passed directly.
            dataset(str or array): The data path of the dataset or data itself.
            input_len(str): The length of the input sequence (number of historical points).
            output_len(str): The length of the output sequence (number of future points to predict).
            logger (logging.Logger): logger.
        """
        train_val_test_ratio: List[float] = []
        mode: str = 'inference'
        super().__init__(dataset_name, train_val_test_ratio, mode, memmap)
        self.input_len = input_len
        self.output_len = output_len
        self.logger = logger

        self.description = {}
        if dataset_name:
            self.description_file_path = f'datasets/{dataset_name}/desc.json'
            self.description = self._load_description()

        self.last_datetime:pd.Timestamp = pd.Timestamp.now()
        self._raw_data = dataset
        self.data = self._load_data()

    def _load_description(self) -> dict:
        """
        Loads the description of the dataset from a JSON file.

        Returns:
            dict: A dictionary containing metadata about the dataset, such as its shape and other properties.

        Raises:
            FileNotFoundError: If the description file is not found.
            json.JSONDecodeError: If there is an error decoding the JSON data.
        """
        try:
            with open(self.description_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Description file not found: {self.description_file_path}') from e
        except json.JSONDecodeError as e:
            raise ValueError(f'Error decoding JSON file: {self.description_file_path}') from e

    def _load_data(self) -> np.ndarray:
        """
        Loads the time series data from a file or list and processes it according to the dataset description.
        Returns:
            np.ndarray: The data array for the specified mode (train, validation, or test).

        Raises:
            ValueError: If there is an issue with loading the data file or if the data shape is not as expected.
        """

        if isinstance(self._raw_data, str):
            df = pd.read_csv(self._raw_data, header=None)
        else:
            df = pd.DataFrame(self._raw_data)

        df_index = pd.to_datetime(df[0].values, format='%Y-%m-%d %H:%M:%S').to_numpy()
        df = df[df.columns[1:]]
        df.index = pd.Index(df_index)
        df = df.astype('float32')
        self.last_datetime = df.index[-1]

        data = np.expand_dims(df.values, axis=-1)
        data = data[..., [0]]

        # if description is not provided, we assume the data is already in the correct shape.
        if not self.dataset_name:
            # calc frequency form df
            freq = int((df.index[1] - df.index[0]).total_seconds() / 60)  # convert to minutes
            if freq <= 0:
                raise ValueError('Frequency must be a positive number.')
            self.description = {
                'shape': data.shape,
                'frequency (minutes)': freq,
            }

        data_with_features = self._add_temporal_features(data, df)

        data_set_shape = self.description['shape']
        _, n, c = data_with_features.shape
        if data_set_shape[1] != n or data_set_shape[2] != c:
            raise ValueError(f'Error loading data. Shape mismatch: expected {data_set_shape[1:]}, got {[n,c]}.')

        return data_with_features

    def _add_temporal_features(self, data, df) -> np.ndarray:
        '''
        Add time of day and day of week as features to the data.

        Args:
            data (np.ndarray): The data array.
            df (pd.DataFrame): The dataframe containing the datetime index.
        
        Returns:
            np.ndarray: The data array with added time of day and day of week features.
        '''

        _, n, _ = data.shape
        feature_list = [data]

        # numerical time_of_day
        tod = (df.index.hour*60 + df.index.minute) / (24*60)
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

        # numerical day_of_week
        dow = df.index.dayofweek / 7
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

        # numerical day_of_month
        dom = (df.index.day - 1) / 31 # df.index.day starts from 1. We need to minus 1 to make it start from 0.
        dom_tiled = np.tile(dom, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dom_tiled)

        # numerical day_of_year
        doy = (df.index.dayofyear - 1) / 366 # df.index.month starts from 1. We need to minus 1 to make it start from 0.
        doy_tiled = np.tile(doy, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(doy_tiled)

        data_with_features = np.concatenate(feature_list, axis=-1).astype('float32')  # L x N x C

        # Remove extra features
        data_set_shape = self.description['shape']
        data_with_features = data_with_features[..., range(data_set_shape[2])]

        return data_with_features

    def append_data(self, new_data: np.ndarray) -> None:
        """
        Append new data to the existing data

        Args:
            new_data (np.ndarray): The new data to append to the existing data.
        """

        freq = self.description['frequency (minutes)']
        l, _, _ = new_data.shape

        data_with_features, datetime_list = self._gen_datetime_list(new_data, self.last_datetime, freq, l)
        self.last_datetime = datetime_list[-1]

        self.data = np.concatenate([self.data, data_with_features], axis=0)

    def _gen_datetime_list(self, new_data: np.ndarray, start_datetime: pd.Timestamp, freq: int, num_steps: int) -> Tuple[np.ndarray, List[pd.Timestamp]]:
        """
        Generate a list of datetime objects based on the start datetime, frequency, and number of steps.

        Args:
            start_datetime (pd.Timestamp): The starting datetime for the sequence.
            freq (int): The frequency of the data in minutes.
            num_steps (int): The number of steps in the sequence.

        Returns:
            List[pd.Timestamp]: A list of datetime objects corresponding to the sequence.
        """
        datetime_list = [start_datetime]
        for _ in range(num_steps):
            datetime_list.append(datetime_list[-1] + pd.Timedelta(minutes=freq))
        new_index = pd.Index(datetime_list[1:])
        new_df = pd.DataFrame()
        new_df.index = new_index
        data_with_features = self._add_temporal_features(new_data, new_df)

        return data_with_features, datetime_list

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample from the dataset, considering both the input and output lengths.
        For inference, the input data is the last 'input_len' points in the dataset, and the output data is the next 'output_len' points.

        Args:
            index (int): The index of the desired sample in the dataset.

        Returns:
            dict: A dictionary containing 'inputs' and 'target', where both are slices of the dataset corresponding to
                  the historical input data and future prediction data, respectively.
        """
        history_data = self.data[-self.input_len:]

        freq = self.description['frequency (minutes)']
        _, n, _ = history_data.shape
        future_data = np.zeros((self.output_len, n, 1))

        data_with_features, _ = self._gen_datetime_list(future_data, self.last_datetime, freq, self.output_len)
        return {'inputs': history_data, 'target': data_with_features}

    def __len__(self) -> int:
        """
        Calculates the total number of samples available in the dataset.
        For inference, there is only one valid sample, as the input data is the last 'input_len' points in the dataset.

        Returns:
            int: The number of valid samples that can be drawn from the dataset, based on the configurations of input and output lengths.
        """
        return 1

class TimeSeriesFullInferenceDataset(BaseDataset):
    """
    A dataset class for time series inference tasks, where the input is a sequence of historical data points
    
    Attributes:
        description_file_path (str): Path to the JSON file containing the description of the dataset.
        description (dict): Metadata about the dataset, such as shape and other properties.
        data (np.ndarray): The loaded time series data array.
        raw_data (str): The raw data path or data list of the dataset.
        last_datetime (pd.Timestamp): The last datetime in the dataset. Used to generate time features of future data.
    """

    # pylint: disable=unused-argument
    def __init__(self, dataset_name:str, dataset: Union[str, list], input_len: int, output_len: int, memmap: bool = False,
                 logger: logging.Logger = None, **kwargs) -> None:
        """
        Initializes the TimeSeriesInferenceDataset by setting up paths, loading data, and 
        preparing it according to the specified configurations.

        Args:
            dataset_name (str): The name of the dataset. If dataset_name is None, the dataset is expected to be passed directly.
            dataset(str or array): The data path of the dataset or data itself.
            input_len(str): The length of the input sequence (number of historical points).
            output_len(str): The length of the output sequence (number of future points to predict).
            logger (logging.Logger): logger.
        """
        train_val_test_ratio: List[float] = []
        mode: str = 'inference'
        super().__init__(dataset_name, train_val_test_ratio, mode, memmap)
        self.input_len = input_len
        self.output_len = output_len
        self.logger = logger

        self.description = {}
        if dataset_name:
            self.description_file_path = f'datasets/{dataset_name}/desc.json'
            self.description = self._load_description()

        self.last_datetime:pd.Timestamp = pd.Timestamp.now()
        self.first_datetime:pd.Timestamp = pd.Timestamp.now()
        self._raw_data = dataset
        self.data = self._load_data()

    def _load_description(self) -> dict:
        """
        Loads the description of the dataset from a JSON file.

        Returns:
            dict: A dictionary containing metadata about the dataset, such as its shape and other properties.

        Raises:
            FileNotFoundError: If the description file is not found.
            json.JSONDecodeError: If there is an error decoding the JSON data.
        """
        try:
            with open(self.description_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Description file not found: {self.description_file_path}') from e
        except json.JSONDecodeError as e:
            raise ValueError(f'Error decoding JSON file: {self.description_file_path}') from e

    def _load_data(self) -> np.ndarray:
        """
        Loads the time series data from a file or list and processes it according to the dataset description.
        Returns:
            np.ndarray: The data array for the specified mode (train, validation, or test).

        Raises:
            ValueError: If there is an issue with loading the data file or if the data shape is not as expected.
        """

        if isinstance(self._raw_data, str):
            df = pd.read_csv(self._raw_data, header=None)
        else:
            df = pd.DataFrame(self._raw_data)

        df_index = pd.to_datetime(df[0].values, format='%Y-%m-%d %H:%M:%S').to_numpy()
        df = df[df.columns[1:]]
        df.index = pd.Index(df_index)
        df = df.astype('float32')
        self.last_datetime = df.index[-1]
        self.first_datetime = df.index[0]

        data = np.expand_dims(df.values, axis=-1)
        data = data[..., [0]]

        # if description is not provided, we assume the data is already in the correct shape.
        if not self.dataset_name:
            # calc frequency form df
            freq = int((df.index[1] - df.index[0]).total_seconds() / 60)  # convert to minutes
            if freq <= 0:
                raise ValueError('Frequency must be a positive number.')
            self.description = {
                'shape': data.shape,
                'frequency (minutes)': freq,
            }

        data_with_features = self._add_temporal_features(data, df)

        data_set_shape = self.description['shape']
        _, n, c = data_with_features.shape
        if data_set_shape[1] != n or data_set_shape[2] != c:
            raise ValueError(f'Error loading data. Shape mismatch: expected {data_set_shape[1:]}, got {[n,c]}.')

        return data_with_features

    def _add_temporal_features(self, data, df) -> np.ndarray:
        '''
        Add temporal and auxiliary features to the data.

        Args:
            data (np.ndarray): The data array.
            df (pd.DataFrame): The dataframe containing the datetime index.
        
        Returns:
            np.ndarray: The data array with added time of day and day of week features.
        '''

        return _build_features(data, df, self.description, self.dataset_name)

    def append_data(self, new_data: np.ndarray) -> None:
        """
        Append new data to the existing data

        Args:
            new_data (np.ndarray): The new data to append to the existing data.
        """

        freq = self.description['frequency (minutes)']
        l, _, _ = new_data.shape

        data_with_features, datetime_list = self._gen_datetime_list(new_data, self.last_datetime, freq, l)
        self.last_datetime = datetime_list[-1]

        self.data = np.concatenate([self.data, data_with_features], axis=0)

    def _gen_datetime_list(self, new_data: np.ndarray, start_datetime: pd.Timestamp, freq: int, num_steps: int) -> Tuple[np.ndarray, List[pd.Timestamp]]:
        """
        Generate a list of datetime objects based on the start datetime, frequency, and number of steps.

        Args:
            start_datetime (pd.Timestamp): The starting datetime for the sequence.
            freq (int): The frequency of the data in minutes.
            num_steps (int): The number of steps in the sequence.

        Returns:
            List[pd.Timestamp]: A list of datetime objects corresponding to the sequence.
        """
        datetime_list = [start_datetime]
        for _ in range(num_steps):
            datetime_list.append(datetime_list[-1] + pd.Timedelta(minutes=freq))
        new_index = pd.Index(datetime_list[1:])
        new_df = pd.DataFrame()
        new_df.index = new_index
        data_with_features = self._add_temporal_features(new_data, new_df)

        return data_with_features, datetime_list

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample from the dataset, considering both the input and output lengths.
        For inference, the input data is the last 'input_len' points in the dataset, and the output data is the next 'output_len' points.

        Args:
            index (int): The index of the desired sample in the dataset.

        Returns:
            dict: A dictionary containing 'inputs' and 'target', where both are slices of the dataset corresponding to
                  the historical input data and future prediction data, respectively.
        """
        history_data = self.data[index:index+self.input_len].astype('float32')

        # freq = self.description['frequency (minutes)']
        _, n, feature_num = history_data.shape
        future_data = self.data[index+self.input_len:index+self.input_len+self.output_len].astype('float32')

        # data_with_features, _ = self._gen_datetime_list(future_data, self.last_datetime, freq, self.output_len)
        return {'inputs': history_data, 'target': future_data}

    def __len__(self) -> int:
        """
        Calculates the total number of samples available in the dataset.
        For inference, there is only one valid sample, as the input data is the last 'input_len' points in the dataset.

        Returns:
            int: The number of valid samples that can be drawn from the dataset, based on the configurations of input and output lengths.
        """
        return max(0, self.data.shape[0] - self.input_len - self.output_len + 1)
