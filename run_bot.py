import sys
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os

class ReloadHandler(FileSystemEventHandler):
    def __init__(self, script):
        self.script = script
        self.process = None
        self.start_process()

    def start_process(self):
        if self.process:
            self.process.kill()
        self.process = subprocess.Popen([sys.executable, self.script])

    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            print("Detected change in:", event.src_path)
            self.start_process()

if __name__ == "__main__":
    script = "full.py"  # <-- replace with your file name
    event_handler = ReloadHandler(script)
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
