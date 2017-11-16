import os
import numpy as np
from PIL import Image
import sys
import re
import json

if sys.platform == 'linux':
    base_path = "/home/temp/schock"
else:
    base_path = "K:/"


selection_file = os.path.join(base_path, "MGS/Videos/wahl.txt")
source_dir = os.path.join(base_path, "MGS/ExtractedFrames/Separated")
dest_dir = os.path.join(base_path, "MGS/TrainsetVGG1500")
num_frames_per_vid = 1500
os.makedirs(dest_dir, exist_ok=True)

selected_movies = []
with open(selection_file) as f:
    content = f.readlines()

for line in content[1:]:
    selected_movies.append(line.strip("\n"))

selected_movies.sort()


sub_dirs = [os.path.join(source_dir, x) for x in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, x))]

movie_dirs = []

for sub_dir in sub_dirs:
    movie_dirs += [os.path.join(sub_dir, x) for x in os.listdir(sub_dir) if os.path.isdir(os.path.join(sub_dir, x))]

movie_dirs.sort()

dirs_selected = []
for movie in selected_movies:
    print(movie)
    regex = re.compile(".*(%s).*" % movie)
    _tmp_list = [m.group(0) for l in movie_dirs for m in [regex.search(l)] if m]
    dirs_selected.append(_tmp_list[0])

dirs_selected.sort()

data_list = []
dictionary = {}

print("Start Indexing Frames")

for idx, directory in enumerate(dirs_selected):
    print("\t %s" % directory)
    files = [os.path.join(directory, x) for x in os.listdir(directory)
             if (os.path.isfile(os.path.join(directory, x)) and x.endswith(".png"))]
    files = np.random.choice(np.asarray(files), num_frames_per_vid)
    dictionary[os.path.split(directory)[-1]] = idx
    labels = [idx for x in files]

    data_list += list(zip(files, labels))

print("Finished Indexing Frames \n writing Files")

with open(os.path.join(dest_dir, "data_dict.json"), "w") as f:
    json.dump(dictionary, f)


for idx, data in enumerate(data_list):
    Image.open(data[0]).convert('RGB').save(os.path.join(dest_dir, "img_%06d.png" % idx))
    label = [0 for x in range(len(dictionary))]
    label[data[1]] = 1
    np.save(os.path.join(dest_dir, "img_%06d.npy" % idx), np.asarray(label))
