import zmq
import sys
from PIL import Image
import time

port = "5555"

context = zmq.Context()
print("Connecting to server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://kepler:%s" % port)


img = Image.open("K:/MGS/Eyes/Images_Boxes/W2_MGS_MVI_0297_126_1.png")

t_start = time.time()
socket.send_pyobj(img)

message = socket.recv_pyobj()

t_end = time.time()
print("Received reply ", "[", message, "]")
print("Time needed: ", (t_end-t_start))