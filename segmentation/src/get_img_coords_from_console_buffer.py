import os
import json
import numpy as np
import sys


def get_vid_points(vid_path, base_path, tmp_list):
    path, folder = os.path.split(vid_path)
    out_path = os.path.join(base_path, os.path.split(path)[-1], folder.rsplit(".", maxsplit=1)[0])

    upper_left_x = tmp_list[0]
    upper_left_y = tmp_list[1]

    middle_x = tmp_list[2]
    middle_y = tmp_list[3]

    bottom_right_x = tmp_list[4]
    bottom_right_y = tmp_list[5]

    tmp_dict = {'Upper Left X': upper_left_x,
                'Upper Left Y': upper_left_y,
                'Middle X': middle_x,
                'Middle Y': middle_y,
                'Bottom Right X': bottom_right_x,
                'Bottom Right Y': bottom_right_y}

    return tmp_dict


tmp_file = "K:/MGS/ExtractedFrames/tmp.txt"

if sys.platform == 'linux':
    base_path = "/home/temp/schock"
else:
    base_path = "K:/"

root_dir = os.path.join(base_path, "MGS/Videos")

base_out_dir = os.path.join(base_path, "MGS/ExtractedFrames/Whole")


subdirs = [os.path.join(root_dir, x) for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]
subdirs.sort()
movie_files = []
for subdir in subdirs:
    movie_files += [os.path.join(subdir, x) for x in os.listdir(subdir) if (os.path.isfile(os.path.join(subdir, x)) and x.endswith(".MOV"))]
movie_files.sort()


movie_dict = {}

coordinate_list = []
with open(tmp_file) as f:
    content = f.readlines()

for idx in range(0, len(content), 6):
    _tmp = []
    for i in range(0, 6):
        _tmp.append(int(content[idx+i].strip("\n").rsplit(":")[-1]))
    coordinate_list.append(_tmp)


for idx, mv_file in enumerate(movie_files):
    if (idx+1) % 5 == 0:
        print("%d of %d" % (idx+1, len(movie_files)))
    movie_dict[os.path.split(mv_file)[-1].rsplit(".", maxsplit=1)[0]] = get_vid_points(mv_file, base_out_dir, coordinate_list[idx])

with open(os.path.join(base_out_dir, "movie_box_points.json"), "w") as f:
    json.dump(movie_dict, f)