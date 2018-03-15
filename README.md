## Installation

* ``git clone https://git.lfb.rwth-aachen.de/schock/mouse-project.git --branch classification --single-branch``
* ``cd mouse-project``
* ``conda env create -f linux-env-gpu.yaml``
* ``source activate mouse_classification``

## Setup Datatsets 

* Unpack "ear_images.zip" and "eye_images.zip"
* Change Paths and Train options in "main_kfold.py"
..* "param_dir" should be a directory containing the file "params_tutorialNet_CIFAR10.pth"
..* trained parameters will also be saved to "param_dir"

## Training

* specify CUDA_VISIBLE_DEVICES in "main_kfold.py"
* run script "main_kfold.py" 