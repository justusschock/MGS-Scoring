#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import time
import zmq
from backend import Backend

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:6666")
backend = Backend()

print("Started Server")


while True:
    #  Wait for next request from client

    message = socket.recv_pyobj()
    t_start = time.time()
    pred = backend.predict(message)
    t_end = time.time()
    socket.send_pyobj(pred)
    print("Time needed: ", (t_end-t_start))