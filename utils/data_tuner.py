#  Copyright (c) 2022 Andrew
#  Email: andrewlee1807@gmail.com

from sklearn.model_selection import train_test_split
import numpy as np


def pattern(datapack: np.array, kernel_size: int, gap=7):
    # sequence.shape = [num_record, timeseries, feature] , example (1093, 24, 1)
    head_kernel_size = tail_kernel_size = kernel_size // 2

    # Padding
    padding = np.zeros((gap + tail_kernel_size, datapack.shape[-1]))  # [padding, feature]

    # padding = np.zeros(gap + tail_kernel_size)

    def generate_index(ix):
        for i in range(0, head_kernel_size):  # gen index from head
            list_ix.append(ix + i)
        for j in range(0, tail_kernel_size):  # gen index from tail
            list_ix.append(ix + gap + j)

    # ix_padding = len(sequence) - (gap + tail_kernel_size)
    new_datapack = []
    # align sequence
    for time_pattern in datapack:
        new_sequence = np.concatenate([time_pattern, padding])
        list_ix = []
        for node_index in range(0, len(time_pattern)):
            generate_index(node_index)
        # new_sequence = new_sequence[list_ix]
        new_datapack.append(new_sequence[list_ix])

    return np.asarray(new_datapack)


class TimeSeriesGenerator:
    """
    This class only support to prepare training (backup to TSF class)
    """

    def __init__(
            self,
            data: np.ndarray,
            config,
            shift=1,
            shuffle=False,
            normalize_type=1
    ):
        """
        :normalize: The mean and standard deviation should only be computed using the training data so that the models
        have no access to the values in the validation and test sets.
            1: MinMaxScaler,
            2: StandardScaler,
            3: RobustScaler,
            4: PowerTransformer
            None: no normalization # should not be used
        return:
        data_train,
        data_valid,
        data_test,
        function: inverse_scale_transform
        """
        self.data_train = None
        self.data_test = None
        self.scaler_x = None
        self.scaler_y = None
        self.raw_data = data
        self.input_width = config['input_width']
        self.output_length = config['output_length']
        self.shift = shift
        self.shuffle = shuffle

        """
        The procedure of data preparation:
        1. Split data into TRAIN and TEST
            [num_record, timeseries-past, feature, timeseries-label]
                example [16752, X] -> [15076, X], [1676, X]
        2. Building the Time series data type: 
            [num_record, timeseries-past, feature, timeseries-label]
                example (1093, 168, 1, 7)
        3. Normalize data
        4. Split train data into TRAIN and VALID
        5. Normalize data
        """

        self.X_train, self.X_test = self.split_data(data, config['train_ratio'])  # [16752, X] -> [15076, X], [1676, X]

        # ASSUME TRAIN AND TEST DATASET HAVE THE SAME DISTRIBUTION
        self.scaler_engine = None  # This is for the normalization of the TRAIN dataset but apply to test and valid
        if normalize_type is not None:
            self.X_train, self.scaler_engine = self.normalize_dataset(self.X_train, standardization_type=normalize_type)
            if self.X_test is not None:
                self.X_test, _ = self.normalize_dataset(self.X_test, standardization_type=normalize_type,
                                                        scaler=self.scaler_engine)

        # Split train and valid dataset for TRAINING PROCESS. The distribution of train and valid dataset is the same.
        self.X_train, self.X_valid = self.split_data(self.X_train, 0.9)
        if normalize_type is not None:
            self.X_valid, _ = self.normalize_dataset(self.X_valid, standardization_type=normalize_type,
                                                     scaler=self.scaler_engine)

        self.data_train = self.build_tsd(self.X_train)  # (13568, X) -> [(13399, 168, X), (13399, 1, X)]
        self.data_valid = self.build_tsd(self.X_valid)  # (1508, X) -> [(1339, 168, X), (1339, 1, X)]
        if self.X_test is not None:
            self.data_test = self.build_tsd(self.X_test)
        else:
            self.data_test = None

        self.data_train_adjustment = None
        self.data_valid_adjustment = None
        self.data_test_adjustment = None
        # self.normalize_data()

    def normalize_dataset(self, dataset, standardization_type, scaler=None):
        """

        :param scaler:
        :param dataset: [Number of records, Number of features]
        :param standardization_type:
        :return:
        """
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import PowerTransformer
        from sklearn.preprocessing import RobustScaler
        from sklearn.preprocessing import StandardScaler

        standardization_methods = {
            1: MinMaxScaler,
            2: StandardScaler,  # Gaussian distribution
            3: RobustScaler,
            4: PowerTransformer,
        }
        assert standardization_type in standardization_methods.keys()

        standardization_method = standardization_methods[standardization_type]

        # There are some NaN values in the dataset, but the Normalization ignores them (not effectively results)
        if scaler is None:
            scaler = standardization_method()

        scaled_data = scaler.fit_transform(dataset)

        return scaled_data, scaler

    # def split_data(self, dataset, ratio):
    #     X_test = None  # No testing, using whole data to train
    #     X_train = self.raw_data
    #     if ratio is not None:
    #         X_train, X_test = train_test_split(
    #             self.raw_data, train_size=ratio, shuffle=self.shuffle
    #         )
    #     X_train, X_valid = train_test_split(
    #         X_train, train_size=0.9, shuffle=self.shuffle
    #     )
    #
    #     return X_train, X_valid, X_test

    def split_data(self, dataset, ratio):
        X_test = None  # No testing, using whole data to train
        X_train = dataset
        if ratio is not None:
            X_train, X_test = train_test_split(dataset, train_size=ratio, shuffle=self.shuffle)
        return X_train, X_test

    def inverse_scale_transform(self, y_predicted):
        """
        un-scale predicted output
        """
        if self.scaler_engine is not None:
            return self.scaler_engine.inverse_transform(y_predicted)
        return y_predicted

    def re_arrange_sequence(self, config):
        """Arranges the input sequence to support Model1 training"""
        self.data_train_adjustment = (pattern(datapack=self.data_train[0],
                                   kernel_size=config['kernel_size'],
                                   gap=config['gap']),
                                   self.data_train[1])

        self.data_valid_adjustment = (pattern(datapack=self.data_valid[0],
                                   kernel_size=config['kernel_size'],
                                   gap=config['gap']),
                                   self.data_valid[1])
        if self.data_test is not None:
            self.data_test_adjustment = (pattern(datapack=self.data_test[0],
                                      kernel_size=config['kernel_size'],
                                      gap=config['gap']),
                              self.data_test[1])

    def normalize_data(self, standardization_type=1):
        """The mean and standard deviation should only be computed using the training data so that the models
        have no access to the values in the validation and test sets.
        1: MinMaxScaler, 2: StandardScaler, 3: RobustScaler, 4: PowerTransformer
        """
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import PowerTransformer
        from sklearn.preprocessing import RobustScaler
        from sklearn.preprocessing import StandardScaler

        standardization_methods = {
            1: MinMaxScaler,
            2: StandardScaler,
            3: RobustScaler,
            4: PowerTransformer,
        }
        standardization_method = standardization_methods[standardization_type]
        scaler_x = standardization_method()
        scaler_x.fit(self.data_train[0])
        scaler_y = standardization_method()
        scaler_y.fit(self.data_train[1])
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

        self.data_train = (
            scaler_x.transform(self.data_train[0]),
            scaler_y.transform(self.data_train[1]),
        )
        # converting into L.S.T.M format
        self.data_train = self.data_train[0][..., np.newaxis], self.data_train[1]
        self.data_valid = (
            scaler_x.transform(self.data_valid[0]),
            scaler_y.transform(self.data_valid[1]),
        )
        self.data_valid = self.data_valid[0][..., np.newaxis], self.data_valid[1]
        if self.data_test is not None:
            self.data_test = (
                scaler_x.transform(self.data_test[0]),
                scaler_y.transform(self.data_test[1]),
            )
            self.data_test = self.data_test[0][..., np.newaxis], self.data_test[1]

    def build_tsd_test(self, data):
        """
        Build time series dataset ==> (VALUES_, LABELS_)
        This function is used to build the time series dataset for training dataset, validation dataset and testing dataset
        :param data: [Number of records, Number of features]
        :return: [Number of records, INPUT_WIDTH, INPUT_DIMENSION], [Number of records, OUTPUT_LENGTH, OUTPUT_DIMENSION]
        """
        X_data, y_label = [], []
        if self.input_width >= len(data) - self.output_length - self.input_width:
            raise ValueError(
                f"Cannot devide sequence with length={len(data)}. The dataset is too small to be used input_length= {self.input_width}. Please reduce your input_length"
            )

        for i in range(self.input_width, len(data) - self.output_length):
            X_data.append(data[i - self.input_width: i])
            y_label.append(data[i: i + self.output_length])

        X_data, y_label = np.array(X_data), np.array(y_label)

        return X_data, y_label

    def build_tsd(self, data):
        """
        Build time series dataset ==> (VALUES_, LABELS_)
        This function is used to build the time series dataset for training dataset, validation dataset and testing dataset
        :param data: [Number of records, Number of features]
        :return: [Number of records, INPUT_WIDTH, INPUT_DIMENSION], [Number of records, OUTPUT_LENGTH, OUTPUT_DIMENSION]
        """
        X_data, y_label = [], []
        if self.input_width >= len(data) - self.output_length - self.input_width:
            raise ValueError(
                f"Cannot devide sequence with length={len(data)}. The dataset is too small to be used input_length= {self.input_width}. Please reduce your input_length"
            )

        for i in range(self.input_width, len(data) - self.output_length):
            X_data.append(data[i - self.input_width: i])
            y_label.append(data[i: i + self.output_length])

        X_data, y_label = np.array(X_data), np.array(y_label)

        return X_data, y_label


from utils.datasets import *


def get_all_data_supported():
    return list(CONFIG_PATH.keys())


class Dataset:
    """
    Dataset class hold all the dataset via dataset name
    :function:
    - Load dataset
    """

    def __init__(self, dataset_name):
        dataset_name = dataset_name.upper()
        if dataset_name not in get_all_data_supported():
            raise f"Dataset name {dataset_name} isn't supported"
        self.dataset_name = dataset_name
        # DataLoader
        self.dataloader = self.__load_data()

    def __load_data(self):
        if self.dataset_name == cnu_str:
            return CNU()
        elif self.dataset_name == comed_str:
            return COMED()
        elif self.dataset_name == france_household_hour_str:
            return FRANCEHOUSEHOLD()
        elif self.dataset_name == gyeonggi_str:
            return GYEONGGI()
        elif self.dataset_name == spain_str:
            return SPAIN()

        return None
