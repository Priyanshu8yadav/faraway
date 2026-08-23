from dashboard.app import start_mjpeg_server
import time

server = start_mjpeg_server(8081)
print(f"Server started: {server}")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
