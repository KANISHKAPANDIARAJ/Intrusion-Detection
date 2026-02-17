import socket
import time

target = "172.16.50.224"
port = 8080

for i in range(200):
    try:
        s = socket.socket()
        s.connect((target, port))
        s.close()
    except:
        pass
    time.sleep(0.01)
