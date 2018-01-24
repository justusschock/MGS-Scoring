import h5py
import numpy as np

with h5py.File("mouse_dataset2.hdf5", "r") as file:
    labels = file['labels'][:]
    images = file['images'][:]

print('')