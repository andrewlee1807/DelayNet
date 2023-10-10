#  Copyright (c) 2022 Andrew
#  Email: andrewlee1807@gmail.com

import numpy as np
from sklearn.model_selection import train_test_split


def pattern(datapack: np.array, kernel_size: int, gap=7, delay_factor=1):
    # sequence.shape = [num_record, timeseries, feature] , example (1093, 24, 1)

    # Padding
    padding = np.zeros((delay_factor * gap + delay_factor * kernel_size, datapack.shape[-1]))  # [padding, feature]

    def generate_kernel_order():
        ixs = []
        for d in range(0, delay_factor + 1):
            for k in range(0, kernel_size):
                ixs.append(gap * d + k)
        return np.asarray(ixs)  # example: [0,1,2,5,6,7,10,11,12]

    kernel_order = generate_kernel_order()

    def generate_index(seq_len):
        idx = np.asarray([], dtype=np.int32)
        for node_index in range(0, seq_len):
            idx = np.concatenate((idx, kernel_order + node_index))
        return idx

    input_len = datapack.shape[1]  # get input length
    index_sample = generate_index(input_len)

    new_datapack = []
    for time_pattern in datapack:
        new_sequence = np.concatenate([time_pattern, padding])
        new_datapack.append(new_sequence[index_sample])

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
        self.scaler_engine = None  # This is for the normalization of the TRAIN dataset but apply to test and valid

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
        data_bk = data
        # data_gr = [data[:2900], data[2900:7500], data[7500:]]
        self.X_train = []
        self.X_valid = []
        self.X_test = []
        # for data in data_gr:
        X_train, X_valid, X_test = self.split_norm_data(data, config['train_ratio'], normalize_type)
        self.X_train = self.X_train + list(X_train)
        self.X_valid = self.X_valid + list(X_valid)
        self.X_test = self.X_test + list(X_test)
        self.X_train = np.asarray(self.X_train)
        self.X_valid = np.asarray(self.X_valid)
        self.X_test = np.asarray(self.X_test)

        # (13568, X) -> [(13399, 168, X), (13399, 1, prediction_step)]
        self.data_train = self.build_tsd(self.X_train,
                                         config["features"].index(config["prediction_feature"]))
        # (1508, X) -> [(1339, 168, X), (1339, prediction_step)]
        self.data_valid = self.build_tsd(self.X_valid,
                                         config["features"].index(config["prediction_feature"]))
        if self.X_test is not None:
            self.data_test = self.build_tsd(self.X_test,
                                            config["features"].index(config["prediction_feature"]))
        else:
            self.data_test = None

        self.data_train_gen = None
        self.data_valid_gen = None
        self.data_test_gen = None

    def __split_2_set__(self, dataset, ratio):
        X_test = None  # No testing, using whole data to train
        X_train = dataset
        if ratio is not None:
            X_train, X_test = train_test_split(dataset, train_size=ratio, shuffle=self.shuffle)
        return X_train, X_test

    def split_norm_data(self, data, ratio, normalize_type=None):
        X_train, X_test = self.__split_2_set__(data, ratio)  # [16752, X] -> [15076, X], [1676, X]

        # ASSUME TRAIN AND TEST DATASET HAVE THE SAME DISTRIBUTION
        if normalize_type is not None:
            X_train = self.normalize_dataset(X_train,
                                             standardization_type=normalize_type)
            if X_test is not None:
                X_test = self.normalize_dataset(X_test, standardization_type=normalize_type)

        # Split train and valid dataset for TRAINING PROCESS. The distribution of train and valid dataset is the same.
        X_train, X_valid = self.__split_2_set__(X_train, 0.9)
        if normalize_type is not None:
            X_valid = self.normalize_dataset(X_valid, standardization_type=normalize_type)

        return X_train, X_valid, X_test

    def normalize_dataset(self, dataset, standardization_type):
        """

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
        if self.scaler_engine is None:
            self.scaler_engine = standardization_method()
            scaled_data = self.scaler_engine.fit_transform(dataset)
        else:
            scaled_data = self.scaler_engine.transform(dataset)

        return scaled_data

    def inverse_scale_transform(self, y_predicted):
        """
        un-scale predicted output
        """
        if self.scaler_engine is not None:
            return self.scaler_engine.inverse_transform(y_predicted)
        return y_predicted

    def re_arrange_sequence(self, config):
        """Arranges the input sequence to support Model1 training"""
        self.data_train_gen = (pattern(datapack=self.data_train[0],
                                       kernel_size=config['kernel_size'],
                                       gap=config['gap'],
                                       delay_factor=config['delay_factor']),
                               self.data_train[1])

        self.data_valid_gen = (pattern(datapack=self.data_valid[0],
                                       kernel_size=config['kernel_size'],
                                       gap=config['gap'],
                                       delay_factor=config['delay_factor']),
                               self.data_valid[1])
        if self.data_test is not None:
            self.data_test_gen = (pattern(datapack=self.data_test[0],
                                          kernel_size=config['kernel_size'],
                                          gap=config['gap'],
                                          delay_factor=config['delay_factor']),
                                  self.data_test[1])
        config['input_width'] = self.data_train_gen[0].shape[1]
        # # saving data_train, data_valid, data_test as a numpy file to use in next time
        # saving_file_pkl(f'{config["output_dir"]}/{config["dataset_name"]}_data_train.pkl', self.data_train)
        # saving_file_pkl(f'{config["output_dir"]}/{config["dataset_name"]}_data_valid.pkl', self.data_valid)
        # if self.data_test is not None:
        #     saving_file_pkl(f'{config["output_dir"]}/{config["dataset_name"]}_data_test.pkl', self.data_test)

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

    def build_tsd(self, data, feature_order):
        """
        Build time series dataset ==> (VALUES_, LABELS_)
        This function is used to build the time series dataset for training dataset, validation dataset and testing dataset
        :param data: [Number of records, Number of features]
        :return: [Number of records, INPUT_WIDTH, INPUT_DIMENSION], [Number of records, OUTPUT_LENGTH, OUTPUT_DIMENSION]

        Args:
            feature_order: Which feature will be predicted
        """
        X_data, y_label = [], []
        if self.input_width >= len(data) - self.output_length - self.input_width:
            raise ValueError(
                f"Cannot devide sequence with length={len(data)}. The dataset is too small to be used input_length= {self.input_width}. Please reduce your input_length"
            )

        for i in range(self.input_width, len(data) - self.output_length):
            X_data.append(data[i - self.input_width: i])
            y_label.append(data[i: i + self.output_length][::, feature_order])

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
        if self.dataset_name == cnu_str or \
                self.dataset_name == cnu_str_engineering_7:
            return CNU(data_name=self.dataset_name)

        elif self.dataset_name == comed_str:
            return COMED()

        elif self.dataset_name == france_household_hour_str:
            return FRANCEHOUSEHOLD()

        elif self.dataset_name == gyeonggi_str or \
                self.dataset_name == gyeonggi2955_str or \
                self.dataset_name == gyeonggi9654_str or \
                self.dataset_name == gyeonggi6499_str:
            return GYEONGGI(data_name=self.dataset_name)

        elif self.dataset_name == spain_str:
            return SPAIN()

        return None
