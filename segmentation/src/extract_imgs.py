import os
import cv2
import numpy as np
import imageio
from multiprocessing import Pool


def extract_frames(file_path, base_path):
    path, folder = os.path.split(file_path)
    out_path = os.path.join(base_path, os.path.split(path)[-1], folder.rsplit(".", maxsplit=1)[0])
    os.makedirs(out_path, exist_ok=True)

    print("Extracting frames from \n %s \n \t to \n %s" % (file_path, out_path))

    reader = imageio.get_reader(file_path)
    for i, im in enumerate(reader):
        imageio.imwrite(os.path.join(out_path, "frame_%d.png" % i), im)
        print(i)


root_dir = "/home/temp/schock/MGS/Videos"

base_out_dir = "/home/temp/schock/MGS/ExtractedFrames/Whole"


subdirs = [os.path.join(root_dir, x) for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]
subdirs.sort()
movie_files = []
for subdir in subdirs:
    movie_files += [os.path.join(subdir, x) for x in os.listdir(subdir) if (os.path.isfile(os.path.join(subdir, x)) and x.endswith(".MOV"))]
movie_files.sort()

print(movie_files)
#for mvfile in movie_files:
#    extract_frames(mvfile, base_out_dir)