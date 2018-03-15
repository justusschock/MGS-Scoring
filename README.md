## Installation

* ``git clone https://git.lfb.rwth-aachen.de/schock/mouse-project.git``
* ``cd mouse-project``
* ``git fetch && git checkout segmentation``
* ``conda env create -f gpu_env_linux_segmentation_mgs.yaml``
* ``source activate mgs_segmentation``

## Setup Datatsets 

* Unpack "Trainsets.zip"
* ``cd src``
* Choose network type to train
* ``cd YOUR_NETWORK_TYPE``
* Change Paths and Train options in "seg_config.yaml"

## Training

* specify CUDA_VISIBLE_DEVICES in main.py
* ``python main.py``