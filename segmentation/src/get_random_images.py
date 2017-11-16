import os
import sys
import shutil
import re
import random
import numpy as np

if sys.platform == 'linux':
    base_path = "/home/temp/schock"
else:
    base_path = "K:/"


selection_file = os.path.join(base_path, "MGS/Videos/wahl.txt")
source_dir = os.path.join(base_path, "MGS/ExtractedFrames/Separated")
dest_dir = os.path.join(base_path, "MGS/SubsetArne25")

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


for directory in dirs_selected:
    print(directory)
    tmp, mv_name = os.path.split(directory)
    tmp, week = os.path.split(tmp)

    out_dir = os.path.join(dest_dir, week, mv_name)
    os.makedirs(out_dir, exist_ok=True)

    first_frames = [os.path.join(directory, x) for x in os.listdir(directory) if
                    (x.endswith("_1.png") and os.path.isfile(os.path.join(directory, x)))]

    print("Listed Frames")

    number_of_whole_frames = len(first_frames)

    integers = np.asarray(list(range(number_of_whole_frames)))

    frame_nbrs = np.random.choice(integers, 25)

    for number in frame_nbrs:
        for idx in [1, 2, 3, 4]:
            shutil.copy2(src=os.path.join(directory, "frame_%d_%d.png" % (number, idx)),
                         dst=os.path.join(out_dir, "frame_%d_%d.png" % (number, idx)))
