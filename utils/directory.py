#  Copyright (c) 2023 Andrew
#  Email: andrewlee1807@gmail.com

import os
from datetime import datetime
import pickle


def create_file(filename):
    if os.path.exists(filename):
        # create new file with datetime suffix
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        basename, ext = os.path.splitext(filename)
        new_filename = f"{basename}_{timestamp}{ext}"
        return new_filename
    return filename


def saving_file_pkl(file_path, data):
    if os.path.exists(file_path):
        file_path = create_file(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def open_file_pkl(file_path):
    if os.path.exists(file_path):
        f = open(file_path, 'rb')
        data_loaded = pickle.load(f)
        f.close()
        return data_loaded
    return None
