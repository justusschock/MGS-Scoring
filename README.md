# Server setup
The following commands should be executed on a server with graphics hardware and preferably within the same network as the client.

## Installation

* ``git clone https://git.lfb.rwth-aachen.de/schock/mouse-project.git --branch backend --single-branch``
* ``cd mouse-project``
* ``conda env create -f gpu_env_linux_backend_mgs.yaml``
* ``source activate mgs_backend``

## Start server
* ``cd src``
* Start the server: ``python zmq_server.py [-p YOUR_PORT]``

# Client setup
Note that this setup should work on an windows client.

## Installation

* ``git clone https://git.lfb.rwth-aachen.de/schock/mouse-project.git --branch gui --single-branch``
* ``cd mouse-project``
* ``conda env create -f maus_gui.yaml``
* ``source activate maus_gui``

## Start GUI
* Start the GUI: ``python main.py``

## Optional: Label tool
* Start the label tool: ``python label_tool/main.py``

## Usage
* Enter the server credentials (hostname and port) in the corresponding inputs
* Open a video from the file dialog
* Select one or more boxes (again, this can be done using the file dialog)
* Start the video and watch the plots