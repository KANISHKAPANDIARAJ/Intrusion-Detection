import socket
import time

target = "127.0.0.1"
port = 8080

for i in range(500):
    try:
        s = socket.socket()
        s.connect((target, port))
        s.close()
    except:
        pass
    time.sleep(0.01)
