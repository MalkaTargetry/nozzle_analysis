import h5py
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
from numpy.array_api import uint8


class HDF5Wrapper:

    ######################################################################################################
    ############# wrapper class to provide dot notation access to HDF5 groups and datasets ###############
    ######################################################################################################

    def __init__(self, hdf_obj):
        self._hdf_obj = hdf_obj

    def close(self):
        try:
            self._hdf_obj.file.close()
        except ValueError:
            print('HDF5 file is not open or already closed.')

    @staticmethod
    def _convert(value):
        if isinstance(value, (np.generic, np.ndarray)) and value.size == 1:
            value = value.item()
        elif isinstance(value, bytes):
            try:
                value = pickle.loads(value)  # Attempt to deserialize byte strings
                if isinstance(value, (float, int, np.ndarray)):
                    return value
            except (pickle.UnpicklingError, ValueError, UnicodeDecodeError):
                try:
                    value = np.frombuffer(value, dtype=np.uint8)  # Try converting raw bytes to numpy array
                    if value.size > 1:
                        side_length = int(np.sqrt(value.size))
                        return value[:side_length * side_length].reshape((side_length, side_length))
                except ValueError:
                    pass
        else:
            print('Unsupported data type for _convert.')
        return value

    @staticmethod
    def _decode_scalar(v):
        """Convert common HDF5 scalar types to normal Python types."""
        # numpy scalar -> python scalar
        if isinstance(v, np.generic):
            v = v.item()

        # bytes -> str (best effort)
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode("utf-8")
            except UnicodeDecodeError:
                return v  # keep raw bytes if not utf-8

        return v

    def _display_image(self, camera_name = None):
        image = self.get_image(camera_name)
        if image is None:
            print('No image to display.')
            return

        if len(image.shape) == 2:
            print('displaying image')
            cv2.imshow('Grayscale Image', image)

            # wait for a key press indefinitely or for a specified time (0 means indefinitely)
            cv2.waitKey(0)

            # close all opencv windows
            cv2.destroyAllWindows()



        # # olde method:
        # if isinstance(data, np.ndarray):
        #     if len(data.shape) == 2:
        #         print('displaying image')
        #         cv2.imshow('Grayscale Image', data)
        #
        #         # wait for a key press indefinitely or for a specified time (0 means indefinitely)
        #         cv2.waitKey(0)
        #
        #         # close all opencv windows
        #         cv2.destroyAllWindows()
        #     elif len(data.shape) == 1 and data.size > 1:
        #         side_length = int(np.sqrt(data.size))
        #         if side_length * side_length == data.size:
        #             data = data.reshape((side_length, side_length))
        #             plt.imshow(data, cmap='gray', interpolation='nearest')
        #             plt.title("HDF5 Dataset Visualization")
        #             plt.colorbar()
        #             plt.show()


    @staticmethod
    def _find_dataset(group):
        """Recursively find the first Dataset inside a Group, or return None."""
        for name, item in group.items():
            if isinstance(item, h5py.Dataset):
                print(item)
                return item
            if isinstance(item, h5py.Group):
                ds = HDF5Wrapper._find_dataset(item)
                if ds is not None:
                    return ds
        return None

    def _read_path(self, path, default=None):
        """Read a dataset at an absolute or relative path and convert scalar types."""
        try:
            obj = self._hdf_obj[path]
        except KeyError as e:
            raise KeyError(
                f"HDF5 path not found: {path!r}. "
            ) from e
        except (TypeError, ValueError) as e:
            raise
        except OSError as e:
            # file closed or I/O issues; usually you *do* want to know about this
            raise OSError(f"Failed to access HDF5 object at {path!r}. Is the file open?") from e

        if isinstance(obj, h5py.Dataset):
            data = obj[()]
            # data = self._convert(data)
            return self._decode_scalar(data)

        if isinstance(obj, h5py.Group):
            return HDF5Wrapper(obj)

        return default

    def group_to_dict(self, group_path, *, include_datasets=True, include_groups=True):
        """
        Recursively convert a group subtree into a nested dict of python values.
        not for the cameras group!
        """
        grp = self._read_path(group_path)
        if not isinstance(grp, HDF5Wrapper):
            return {}

        def rec(h5group):
            d = {}
            for name, item in h5group.items():
                if isinstance(item, h5py.Dataset) and include_datasets:
                    v = item[()]
                    v = self._decode_scalar(v)
                    d[name] = v
                elif isinstance(item, h5py.Group) and include_groups:
                    d[name] = rec(item)
            return d

        return rec(grp._hdf_obj)

    def camera_list(self):
        """Returns a list of camera names in the HDF5 file."""
        if 'cameras' in self._hdf_obj:
            cameras_group = self._hdf_obj['cameras']
            return list(cameras_group.keys())
        else:
            return []

    def get_image(self, camera_name = None):
        """Retrieves the image data from the specified camera."""
        camera_list = self.camera_list()
        if not camera_list:
            print('No cameras found in HDF5 file.')
            return None
        elif camera_name is None:
            if len(self.camera_list()) == 1:
                camera_name = camera_list[0]
            else:
                print('Multiple cameras found. Please specify a camera name.')
                return None

        elif camera_name not in camera_list:
            print(f'Camera "{camera_name}" not found in HDF5 file.')
            return None

        buffered_data = self._hdf_obj['cameras'][camera_name]['data'][()]

        if buffered_data.dtype != np.uint8:
            print('The data is not in the expected uint8 format.')
            return None

        blob = buffered_data.tobytes()
        image_data = pickle.loads(blob)[0]

        if image_data is None:
            print(f'No dataset found for camera "{camera_name}".')
            return None
        elif type(image_data) != np.ndarray:
            print(f'Camera "{camera_name}" data is not a numpy array.')
            return None
        elif image_data.dtype != np.uint16:
            print(f'Camera "{camera_name}" data is not in uint16 format.')
            return None

        image = self._convert(image_data)
        if not isinstance(image, np.ndarray):
            print(f'Camera "{camera_name}" data is not a valid image format.')
            return None

        return image

    def save_image(self, camera_name, file_name, folder_path = None):
        """Saves the image from the specified camera to a PNG file."""
        image = self.get_image(camera_name)

        # default folder path
        if folder_path is None:
            folder_path = 'C:/Users/vmlab/Documents/data/images/'
        file_path = folder_path + file_name + '.png'
        if len(image.shape) == 2:
            plt.imsave(str(file_path), image, cmap='gray', vmin=0, vmax=image.max())
            # cv2.imwrite(file_path, image)
            print(f'Image from camera "{camera_name}" saved to {file_path}')
        elif len(image.shape) == 1 and image.size > 1:
            side_length = int(np.sqrt(image.size))
            if side_length * side_length == image.size:
                image = image.reshape((side_length, side_length))
                cv2.imwrite(file_path, image)
                print(f'Image from camera "{camera_name}" saved to {file_path}')

    def __getattr__(self, item):

        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            try:
                hdf_item = self._hdf_obj[item]
                if isinstance(hdf_item, h5py.Group):
                    return HDF5Wrapper(hdf_item)
                elif isinstance(hdf_item, h5py.Dataset):
                    data = hdf_item[()]  # Retrieve the dataset contents
                    # self._display_image(self._convert(data))
                    return self._decode_scalar(data)
            except KeyError:
                raise AttributeError(f"No such attribute: {item}")

    def __dir__(self):
        return list(self._hdf_obj.keys())


# function to open the HDF5 file and return a wrapper
def open_hdf5_wrapper(file_path):
    hdf_file = h5py.File(file_path, 'r')
    return HDF5Wrapper(hdf_file)
