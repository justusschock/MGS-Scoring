
import time
import zmq
from backend import Backend
import argparse

parser = argparse.ArgumentParser(description='Parse Port from command line')
parser.add_argument("-p", "--port", type=int, help="Backend communication port", default=5555)

args = parser.parse_args()

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:" + str(args.port))
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