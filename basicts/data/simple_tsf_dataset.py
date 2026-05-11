import inspect
import json
import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset


class TimeSeriesForecastingDataset(BaseDataset):
    """
    A dataset class for time series forecasting problems, handling the loading, parsing, and partitioning
    of time series data into training, validation, and testing sets based on provided ratios.
    
    This class supports configurations where sequences may or may not overlap, accommodating scenarios
    where time series data is drawn from continuous periods or distinct episodes, affecting how
    the data is split into batches for model training or evaluation.
    
    Attributes:
        data_file_path (str): Path to the file containing the time series data.
        description_file_path (str): Path to the JSON file containing the description of the dataset.
        data (np.ndarray): The loaded time series data array, split according to the specified mode.
        description (dict): Metadata about the dataset, such as shape and other properties.
    """

    def __init__(self, dataset_name: str, train_val_test_ratio: List[float], mode: str, input_len: int,
                 output_len: int, memmap: bool = False, overlap: bool = False, logger: logging.Logger = None) -> None:
        """
        Initializes the TimeSeriesForecastingDataset by setting up paths, loading data, and 
        preparing it according to the specified configurations.

        Args:
            dataset_name (str): The name of the dataset.
            train_val_test_ratio (List[float]): Ratios for splitting the dataset into train, validation, and test sets.
                Each value should be a float between 0 and 1, and their sum should ideally be 1.
            mode (str): The operation mode of the dataset. Valid values are 'train', 'valid', or 'test'.
            input_len (int): The length of the input sequence (number of historical points).
            output_len (int): The length of the output sequence (number of future points to predict).
            overlap (bool): Flag to determine if training/validation/test splits should overlap. 
                Defaults to False for strictly non-overlapping periods. Set to True to allow overlap.
            logger (logging.Logger): logger.

        Raises:
            AssertionError: If `mode` is not one of ['train', 'valid', 'test'].
        """
        assert mode in ['train', 'valid', 'test'], f"Invalid mode: {mode}. Must be one of ['train', 'valid', 'test']."
        super().__init__(dataset_name, train_val_test_ratio, mode, memmap)
        self.input_len = input_len
        self.output_len = output_len
        self.overlap = overlap
        self.logger = logger

        self.data_file_path = f'datasets/{dataset_name}/data.dat'
        self.description_file_path = f'datasets/{dataset_name}/desc.json'
        self.description = self._load_description()
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
        Loads the time series data from a file and splits it according to the selected mode.

        Returns:
            np.ndarray: The data array for the specified mode (train, validation, or test).

        Raises:
            ValueError: If there is an issue with loading the data file or if the data shape is not as expected.
        """

        try:
            data = np.memmap(self.data_file_path, dtype='float32', mode='r', shape=tuple(self.description['shape']))
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f'Error loading data file: {self.data_file_path}') from e

        total_len = len(data)
        valid_len = int(total_len * self.train_val_test_ratio[1])
        test_len = int(total_len * self.train_val_test_ratio[2])
        train_len = total_len - valid_len - test_len

        # Automatically configure the overlap parameter
        minimal_len = self.input_len + self.output_len
        if minimal_len > {'train': train_len, 'valid': valid_len, 'test': test_len}[self.mode]:
            self.overlap = True  # Enable overlap when the train, validation, or test set is too short
            current_frame = inspect.currentframe()
            file_name = inspect.getfile(current_frame)
            line_number = current_frame.f_lineno - 7
            dataset = {'train': 'Training', 'valid': 'Validation', 'test': 'Test'}[self.mode]
            if self.logger is not None:
                self.logger.info(f'{dataset} dataset is too short, enabling overlap. See details in {file_name} at line {line_number}.')
            else:
                print(f'{dataset} dataset is too short, enabling overlap. See details in {file_name} at line {line_number}.')

        if self.mode == 'train':
            offset = self.output_len if self.overlap else 0
            seg = data[:train_len + offset]
        elif self.mode == 'valid':
            offset_left = self.input_len - 1 if self.overlap else 0
            offset_right = self.output_len if self.overlap else 0
            seg = data[train_len - offset_left : train_len + valid_len + offset_right]
        else:  # self.mode == 'test'
            offset = self.input_len - 1 if self.overlap else 0
            seg = data[train_len + valid_len - offset:]

        if not self.memmap:
            seg = seg.copy()
        return seg

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample from the dataset at the specified index, considering both the input and output lengths.

        Args:
            index (int): The index of the desired sample in the dataset.

        Returns:
            dict: A dictionary containing 'inputs' and 'target', where both are slices of the dataset corresponding to
                  the historical input data and future prediction data, respectively.
        """
        history_data = self.data[index:index + self.input_len]
        future_data = self.data[index + self.input_len:index + self.input_len + self.output_len]
        if self.memmap:
            history_data = history_data.copy()
            future_data = future_data.copy()
        return {'inputs': history_data, 'target': future_data}

    def __len__(self) -> int:
        """
        Calculates the total number of samples available in the dataset, adjusted for the lengths of input and output sequences.

        Returns:
            int: The number of valid samples that can be drawn from the dataset, based on the configurations of input and output lengths.
        """
        return len(self.data) - self.input_len - self.output_len + 1


class CalendarSplitDataset(BaseDataset):
    """
    A dataset class for time series forecasting with timestamp-based calendar splits.

    This class mirrors TimeSeriesForecastingDataset behavior for sample generation and overlap,
    but determines train/valid/test boundaries from calendar timestamps instead of ratios.
    """

    def __init__(self, dataset_name: str, train_val_test_ratio: List[float], mode: str, input_len: int,
                output_len: int, timestamp_file: Optional[str] = None,
                holdout_last_days_of_month: Optional[int] = None,
                memmap: bool = False,
                overlap: bool = False, logger: logging.Logger = None) -> None:
        """
        Initializes the TimeSeriesForecastingDataset by setting up paths, loading data, and 
        preparing it according to the specified configurations.

        Args:
            dataset_name (str): The name of the dataset.
            train_val_test_ratio (List[float]): Ratios for splitting the dataset into train, validation, and test sets.
                Each value should be a float between 0 and 1, and their sum should ideally be 1.
            mode (str): The operation mode of the dataset. Valid values are 'train', 'valid', or 'test'.
            input_len (int): The length of the input sequence (number of historical points).
            output_len (int): The length of the output sequence (number of future points to predict).
            overlap (bool): Flag to determine if training/validation/test splits should overlap. 
                Defaults to False for strictly non-overlapping periods. Set to True to allow overlap.
            logger (logging.Logger): logger.

        Raises:
            AssertionError: If `mode` is not one of ['train', 'valid', 'test'].
        """
        assert mode in ['train', 'valid', 'test'], f"Invalid mode: {mode}. Must be one of ['train', 'valid', 'test']."
        super().__init__(dataset_name, train_val_test_ratio, mode, memmap)
        self.input_len = input_len
        self.output_len = output_len
        self.overlap = overlap
        self.logger = logger
        self.holdout_last_days_of_month = holdout_last_days_of_month
        self.sample_starts = None

        self.data_file_path = f'datasets/{dataset_name}/data.dat'
        self.description_file_path = f'datasets/{dataset_name}/desc.json'
        self.description = self._load_description()
        self.timestamp_file_path = self._resolve_timestamp_path(timestamp_file)
        self.timestamps = self._load_timestamps()
        self.data = self._load_data()

    def _resolve_timestamp_path(self, timestamp_file: Optional[str]) -> str:
        """Resolve timestamp CSV path from arg or dataset description."""
        timestamp_name = timestamp_file or self.description.get('timestamp_file', 'timestamps.csv')
        return f'datasets/{self.dataset_name}/{timestamp_name}'

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

    def _load_timestamps(self) -> pd.DatetimeIndex:
        """Load timestamps from CSV."""
        try:
            df = pd.read_csv(self.timestamp_file_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Timestamp file not found: {self.timestamp_file_path}') from e

        if len(df.columns) < 1:
            raise ValueError(f'Timestamp file is empty or malformed: {self.timestamp_file_path}')

        ts = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        if ts.isna().any():
            raise ValueError(f'Timestamp file contains invalid datetime values: {self.timestamp_file_path}')
        return pd.DatetimeIndex(ts)

    def _load_data(self) -> np.ndarray:
        """
        Loads the time series data from a file and splits it according to the selected mode.

        Returns:
            np.ndarray: The data array for the specified mode (train, validation, or test).

        Raises:
            ValueError: If there is an issue with loading the data file or if the data shape is not as expected.
        """

        try:
            data = np.memmap(self.data_file_path, dtype='float32', mode='r', shape=tuple(self.description['shape']))
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f'Error loading data file: {self.data_file_path}') from e

        total_len = len(data)
        if len(self.timestamps) != total_len:
            raise ValueError(
                f'Timestamp length mismatch: data has {total_len} rows, '
                f'timestamps has {len(self.timestamps)} rows.'
            )

        use_calendar_holdout = self.holdout_last_days_of_month is not None
        minimal_len = self.input_len + self.output_len

        if use_calendar_holdout:
            holdout_days = self.holdout_last_days_of_month
            if holdout_days <= 0:
                raise ValueError('Holdout length must be a positive integer.')

            # Hold out the last N days of every month for test split.
            last_day_cutoff = self.timestamps.days_in_month - holdout_days + 1
            test_mask = self.timestamps.day >= last_day_cutoff
            test_indices = np.flatnonzero(test_mask)
            pretest_indices = np.flatnonzero(~test_mask)

            train_valid_ratio_sum = self.train_val_test_ratio[0] + self.train_val_test_ratio[1]
            if train_valid_ratio_sum <= 0:
                raise ValueError('train+valid ratio must be positive when using calendar holdout.')

            valid_ratio_in_pretest = self.train_val_test_ratio[1] / train_valid_ratio_sum
            valid_count = int(len(pretest_indices) * valid_ratio_in_pretest)
            train_count = len(pretest_indices) - valid_count

            train_indices = pretest_indices[:train_count]
            valid_indices = pretest_indices[train_count:]

            selected_indices = {
                'train': train_indices,
                'valid': valid_indices,
                'test': test_indices,
            }[self.mode]

            seg = data[selected_indices]
            self.timestamps = self.timestamps[selected_indices]

            # Build valid sample starts that do not cross removed gaps.
            if len(selected_indices) < minimal_len:
                self.sample_starts = np.array([], dtype=np.int64)
            else:
                if minimal_len == 1:
                    self.sample_starts = np.arange(len(selected_indices), dtype=np.int64)
                else:
                    contiguous_steps = (np.diff(selected_indices) == 1).astype(np.int32)
                    rolling = np.convolve(
                        contiguous_steps,
                        np.ones(minimal_len - 1, dtype=np.int32),
                        mode='valid'
                    )
                    self.sample_starts = np.flatnonzero(rolling == (minimal_len - 1))
        else:
            valid_len = int(total_len * self.train_val_test_ratio[1])
            test_len = int(total_len * self.train_val_test_ratio[2])
            train_len = total_len - valid_len - test_len

            # Automatically configure the overlap parameter
            if minimal_len > {'train': train_len, 'valid': valid_len, 'test': test_len}[self.mode]:
                self.overlap = True  # Enable overlap when the train, validation, or test set is too short
                current_frame = inspect.currentframe()
                file_name = inspect.getfile(current_frame)
                line_number = current_frame.f_lineno - 7
                dataset = {'train': 'Training', 'valid': 'Validation', 'test': 'Test'}[self.mode]
                if self.logger is not None:
                    self.logger.info(f'{dataset} dataset is too short, enabling overlap. See details in {file_name} at line {line_number}.')
                else:
                    print(f'{dataset} dataset is too short, enabling overlap. See details in {file_name} at line {line_number}.')

            if self.mode == 'train':
                offset = self.output_len if self.overlap else 0
                seg_start = 0
                seg_end = train_len + offset
                seg = data[seg_start:seg_end]
            elif self.mode == 'valid':
                offset_left = self.input_len - 1 if self.overlap else 0
                offset_right = self.output_len if self.overlap else 0
                seg_start = train_len - offset_left
                seg_end = train_len + valid_len + offset_right
                seg = data[seg_start:seg_end]
            else:  # self.mode == 'test'
                offset = self.input_len - 1 if self.overlap else 0
                seg_start = train_len + valid_len - offset
                seg_end = total_len
                seg = data[seg_start:seg_end]

            # Keep timestamps aligned with the loaded split segment.
            self.timestamps = self.timestamps[seg_start:seg_end]
            self.sample_starts = np.arange(max(0, len(seg) - minimal_len + 1), dtype=np.int64)

        if not self.memmap:
            seg = seg.copy()
        return seg

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample from the dataset at the specified index, considering both the input and output lengths.

        Args:
            index (int): The index of the desired sample in the dataset.

        Returns:
            dict: A dictionary containing 'inputs' and 'target', where both are slices of the dataset corresponding to
                  the historical input data and future prediction data, respectively.
        """
        start = int(self.sample_starts[index]) if self.sample_starts is not None else index
        history_data = self.data[start:start + self.input_len]
        future_data = self.data[start + self.input_len : start + self.input_len + self.output_len]
        if self.memmap:
            history_data = history_data.copy()
            future_data = future_data.copy()
        return {'inputs': history_data, 'target': future_data}

    def __len__(self) -> int:
        """
        Calculates the total number of samples available in the dataset, adjusted for the lengths of input and output sequences.

        Returns:
            int: The number of valid samples that can be drawn from the dataset, based on the configurations of input and output lengths.
        """
        if self.sample_starts is not None:
            return len(self.sample_starts)
        
        return len(self.data) - self.input_len - self.output_len + 1


class EventAwareCalendarSplitDataset(CalendarSplitDataset):
    """
    Calendar-split dataset variant that removes event-overlapping training samples
    using event channels already present in the loaded data.
    """

    DEFAULT_EVENT_KEYWORDS = ('event', 'incident')

    def __init__(self, dataset_name: str, train_val_test_ratio: List[float], mode: str, input_len: int,
                output_len: int, timestamp_file: Optional[str] = None,
                holdout_last_days_of_month: Optional[int] = None,
                event_buffer_minutes: int = 0,
                memmap: bool = False,
                overlap: bool = False, logger: logging.Logger = None) -> None:
        self.event_feature_indices = None
        self.event_buffer_minutes = event_buffer_minutes
        self.event_mask = None

        super().__init__(
            dataset_name=dataset_name,
            train_val_test_ratio=train_val_test_ratio,
            mode=mode,
            input_len=input_len,
            output_len=output_len,
            timestamp_file=timestamp_file,
            holdout_last_days_of_month=holdout_last_days_of_month,
            memmap=memmap,
            overlap=overlap,
            logger=logger,
        )

    def _load_data(self) -> np.ndarray:
        """Load calendar-split data, then filter training windows by events."""
        seg = super()._load_data()

        if self.mode != 'train':
            return seg

        self.event_feature_indices = self.resolve_event_feature_indices(seg)
        self.event_mask = self.build_event_mask_from_data(seg)
        self.event_mask = self.apply_event_buffer(self.event_mask)
        self.filter_event_overlapping_samples()
        return seg

    def resolve_event_feature_indices(self, data: np.ndarray) -> List[int]:
        """Infer and validate event feature indices against the data feature axis."""
        num_features = data.shape[-1]
        feature_description = self.description.get('feature_description', [])
        event_feature_indices = [
            idx
            for idx, name in enumerate(feature_description)
            if any(keyword in str(name).lower() for keyword in self.DEFAULT_EVENT_KEYWORDS)
        ]

        if len(event_feature_indices) == 0:
            raise ValueError(
                'No event feature indices were inferred from desc.json. '
                f'Expected feature_description entries containing one of {self.DEFAULT_EVENT_KEYWORDS}.'
            )

        normalized_indices = []
        for index in event_feature_indices:
            if index < 0 or index >= num_features:
                raise ValueError(
                    f'Inferred event feature index {index} is out of bounds for {num_features} features.'
                )
            normalized_indices.append(index)
        return normalized_indices

    def build_event_mask_from_data(self, data: np.ndarray) -> np.ndarray:
        """Create a time-level event mask from event feature channels."""
        event_values = np.take(data, indices=self.event_feature_indices, axis=-1)
        event_values = np.nan_to_num(np.asarray(event_values), nan=0.0)
        if event_values.ndim == 1:
            return event_values != 0
        return np.any(event_values != 0, axis=tuple(range(1, event_values.ndim)))

    def apply_event_buffer(self, event_mask: np.ndarray) -> np.ndarray:
        """Expand event timestamps by event_buffer_minutes on both sides."""
        if self.event_buffer_minutes <= 0 or not event_mask.any():
            return event_mask

        step_minutes = self.get_frequency_minutes()
        buffer_steps = int(np.ceil(self.event_buffer_minutes / step_minutes))
        if buffer_steps <= 0:
            return event_mask

        event_counts = np.concatenate(([0], np.cumsum(event_mask.astype(np.int64))))
        indices = np.arange(len(event_mask), dtype=np.int64)
        starts = np.clip(indices - buffer_steps, 0, len(event_mask))
        ends = np.clip(indices + buffer_steps + 1, 0, len(event_mask))
        return (event_counts[ends] - event_counts[starts]) > 0

    def get_frequency_minutes(self) -> float:
        """Read timestamp frequency in minutes from the dataset description."""
        if 'frequency (minutes)' not in self.description:
            raise ValueError('Dataset description must include "frequency (minutes)" to apply an event buffer.')

        frequency_minutes = float(self.description['frequency (minutes)'])
        if frequency_minutes <= 0:
            raise ValueError('Dataset frequency must be positive to apply an event buffer.')
        return frequency_minutes

    def filter_event_overlapping_samples(self) -> None:
        """Remove training sample starts whose configured window overlaps an event."""
        if self.sample_starts is None:
            minimal_len = self.input_len + self.output_len
            self.sample_starts = np.arange(max(0, len(self.data) - minimal_len + 1), dtype=np.int64)

        if len(self.sample_starts) == 0:
            return

        window_len = self.input_len + self.output_len
        starts = self.sample_starts
        ends = starts + window_len
        event_counts = np.concatenate(([0], np.cumsum(self.event_mask.astype(np.int64))))
        keep_mask = (event_counts[ends] - event_counts[starts]) == 0

        original_len = len(self.sample_starts)
        self.sample_starts = self.sample_starts[keep_mask]
        removed = original_len - len(self.sample_starts)
        if removed > 0:
            self.log_info(
                f'Removed {removed} event-overlapping training samples '
                f'from {self.dataset_name} using event features {self.event_feature_indices}.'
            )

    def log_info(self, message: str) -> None:
        """Log with the dataset logger when available."""
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)
