import socket
import time 
HOST = "0.0.0.0"   # listen on all interfaces
PORT = 8080

s = socket.socket()
s.bind((HOST, PORT))
s.listen(100)

print("[+] Victim server listening on port 8080")

while True:
    conn, addr = s.accept()
    timestamp = time.time()
    print(f"[{timestamp}] Connection from {addr[0]}:{addr[1]}")
    conn.close()
