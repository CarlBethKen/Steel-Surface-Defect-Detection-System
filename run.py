#!/usr/bin/env python3

import sys
import time
import signal
import subprocess
import webbrowser
from pathlib import Path


class Launcher:
    def __init__(self):
        self.root = Path(__file__).parent
        self.host = "127.0.0.1"
        self.port = 8000
        self.proc = None

    def check_files(self):
        required = [
            "app.py",
            "templates/system.html",
            "models"
        ]
        for item in required:
            if not (self.root / item).exists():
                print(f"Missing: {item}")
                return False
        return True

    def start_server(self):
        self.proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "app:app",
                "--host", self.host,
                "--port", str(self.port),
                "--reload"
            ],
            cwd=self.root
        )
        time.sleep(2)

    def open_browser(self):
        webbrowser.open(f"http://{self.host}:{self.port}/system")

    def shutdown(self, *_):
        if self.proc:
            self.proc.terminate()
        sys.exit(0)

    def run(self):
        if not self.check_files():
            return

        self.start_server()
        self.open_browser()

        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

        while True:
            time.sleep(1)


if __name__ == "__main__":
    Launcher().run()
