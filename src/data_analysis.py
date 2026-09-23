import os
import numpy as np
import utilities.reading_hdf5 as reading_hdf5

class DataAnalysis:

    def __init__(self, folder_path, pattern='.hdf5'):
        self.folder_path = folder_path
        self.pattern = pattern

        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"The folder path {self.folder_path} does not exist.")

        self.hdf5_files = [
            f for f in os.listdir(self.folder_path)
            if f.endswith(self.pattern)
        ]

        if not self.hdf5_files:
            raise FileNotFoundError(f"No HDF5 files with pattern {self.pattern} found in {self.folder_path}.")



    @staticmethod
    def get_hdf5_group_data(file_path, group_path):
        reader = reading_hdf5.open_hdf5_wrapper(file_path)
        group_data = reader.group_to_dict(group_path)
        reader.close()
        return group_data

    @staticmethod
    def get_image_from_file(file_path, camera_name=None):
        reader = reading_hdf5.open_hdf5_wrapper(file_path)
        image = reader.get_image(camera_name)
        reader.close()
        return image

    def create_hdf5_dict(self, group_path):
        hdf5_dict = {}
        for file_name in self.hdf5_files:
            file_path = os.path.join(self.folder_path, file_name)
            main_data = self.get_hdf5_group_data(file_path, group_path)
            hdf5_dict[file_name] = main_data
        return hdf5_dict

    def get_image_dictionary(self, camera_name=None):
        image_dict = {}
        for file_name in self.hdf5_files:
            file_path = os.path.join(self.folder_path, file_name)
            image_dict[file_name] = self.get_image_from_file(file_path, camera_name)
        return image_dict


def start_analysis(folder_path):
    return DataAnalysis(folder_path)