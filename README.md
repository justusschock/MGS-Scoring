## Installation

* ``git clone https://git.lfb.rwth-aachen.de/schock/mouse-project.git``
* ``cd mouse-project``
* ``git fetch && git checkout backend``
* ``conda env create -f gpu_env_linux_backend_mgs.yaml``
* ``source activate mgs_backend``

## Start Backend
* If you want to use the default port (5555):
	``python zmq_server.py``
* Else:
	``python zmq_server.py -p YOUR_PORT``