import os
import cv2
import numpy as np
import imageio
from multiprocessing import Pool, freeze_support
from functools import partial
import json
from PIL import Image
import matplotlib
import sys
import threading
import re

if sys.platform == 'linux':
    matplotlib.use('Agg')

from matplotlib import pyplot as plt


def extract_frames(file_path, base_path):
    path, folder = os.path.split(file_path)
    out_path = os.path.join(base_path, os.path.split(path)[-1], folder.rsplit(".", maxsplit=1)[0])
    os.makedirs(out_path, exist_ok=True)

    print("Extracting frames from \n %s \n \t to \n %s" % (file_path, out_path))

    reader = imageio.get_reader(file_path)
    for i, im in enumerate(reader):
        imageio.imwrite(os.path.join(out_path, "frame_%d.png" % i), im)

    print("Finished Extracting %s" % file_path)


def get_vid_points(vid_path, base_path):
    path, folder = os.path.split(vid_path)
    out_path = os.path.join(base_path, os.path.split(path)[-1], folder.rsplit(".", maxsplit=1)[0])

    if not os.path.exists(out_path):
        extract_frames(vid_path, base_path)

    image = plt.imread(os.path.join(out_path, "frame_1.png"))

    plt.imshow(image)
    plt.show()

    upper_left_x = int(input("\tX-Coordinate of upper left corner:"))
    upper_left_y = int(input("\tY-Coordinate of upper left corner:"))

    plt.imshow(image)
    plt.show()

    middle_x = int(input("\tX-Coordinate of middle point:"))
    middle_y = int(input("\tY-Coordinate of middle point:"))

    plt.imshow(image)
    plt.show()

    bottom_right_x = int(input("\tX-Coordinate of bottom right corner:"))
    bottom_right_y = int(input("\tY-Coordinate of bottom right corner:"))

    tmp_dict = {'Upper Left X': upper_left_x,
                'Upper Left Y': upper_left_y,
                'Middle X': middle_x,
                'Middle Y': middle_y,
                'Bottom Right X': bottom_right_x,
                'Bottom Right Y': bottom_right_y}

    return tmp_dict


def extract_box(frame, boxes_coord_list, separated_out_dir):
    img = np.array(Image.open(frame).convert('RGB'))

    for idx, coords in enumerate(boxes_coord_list):
        box = img[coords[0][0]:coords[0][1], coords[1][0]:coords[1][1], :]
        Image.fromarray(box).save(os.path.join(separated_out_dir,
                                               os.path.split(frame)[-1].rsplit(".", maxsplit=1)[0] +
                                               "_%d.png" % (idx + 1)))


def extract_boxes_from_frames(selected_movie, base_dir):
    selected_movie = os.path.split(selected_movie.rsplit(".", maxsplit=1)[0])
    week = os.path.split(selected_movie[0])[-1]
    selected_movie = selected_movie[-1]
    current_dict = movie_dict[selected_movie]

    upper_left_box_coordinates = ((current_dict["Upper Left Y"], current_dict["Middle Y"]),
                                  (current_dict["Upper Left X"], current_dict["Middle X"]))

    upper_right_box_coordinates = ((current_dict["Upper Left Y"], current_dict["Middle Y"]),
                                   (current_dict["Middle X"], current_dict["Bottom Right X"]))

    bottom_left_box_coordinates = ((current_dict["Middle Y"], current_dict["Bottom Right Y"]),
                                   (current_dict["Upper Left X"], current_dict["Middle X"]))

    bottom_right_box_coordinates = ((current_dict["Middle Y"], current_dict["Bottom Right Y"]),
                                    (current_dict["Middle X"], current_dict["Bottom Right X"]))

    movie_dir = os.path.join(base_dir, week, selected_movie)

    frames = [os.path.join(movie_dir, x) for x in os.listdir(movie_dir) if (x.endswith(".png") and
                                                                            os.path.isfile(os.path.join(movie_dir, x)))]
    frames.sort()

    separated_out_dir = os.path.join(os.path.split(base_dir)[0], "Separated", week, selected_movie)
    os.makedirs(separated_out_dir, exist_ok=True)

    boxes_coord_list = [
        upper_left_box_coordinates,
        upper_right_box_coordinates,
        bottom_left_box_coordinates,
        bottom_right_box_coordinates
    ]

    with Pool(20) as p:
        p.map(partial(extract_box, boxes_coord_list=boxes_coord_list, separated_out_dir=separated_out_dir), frames)


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


with open(os.path.join(base_out_dir, "movie_box_points.json")) as f:
    movie_dict = json.load(f)

# for idx, mv_file in enumerate(movie_files):
#     if (idx+1) % 5 == 0:
#         print("%d of %d" % (idx+1, len(movie_files)))
#     movie_dict[os.path.split(mv_file)[-1].rsplit(".", maxsplit=1)[0]] = get_vid_points(mv_file, base_out_dir)

# with open(os.path.join(base_out_dir, "movie_box_points.json"), "w") as f:
#     json.dump(movie_dict, f)

selection_file = os.path.join(base_path, "MGS/Videos/wahl.txt")

selected_movies = []
with open(selection_file) as f:
    content = f.readlines()


for line in content[1:]:
    selected_movies.append(line.strip("\n"))

selected_movies.sort()

files_not_selected = []
files_selected = []

for movie in selected_movies:
    print(movie)
    regex = re.compile(".*(%s).*" % movie)
    _tmp_list = [m.group(0) for l in movie_files for m in [regex.search(l)] if m]
    files_selected.append(_tmp_list[0])

for mv in movie_files:
    if mv not in files_selected:
        files_not_selected.append(mv)

for mv in files_not_selected:
    extract_boxes_from_frames(mv, base_out_dir)
