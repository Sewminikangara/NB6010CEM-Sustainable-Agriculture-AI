import subprocess
import time
import os
import sys

dataset = sys.argv[1]
output_file = sys.argv[2]

def download():
    attempt = 1
    while True:
        print(f"Attempt {attempt}...")
        try:
            cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", "."]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end='', flush=True)
                with open("/tmp/kaggle_current.log", "a") as f:
                    f.write(line)
            
            process.wait()
            if process.returncode == 0:
                print("Download successful!")
                return True
            else:
                print(f"Download failed with return code {process.returncode}")
        except Exception as e:
            print(f"Error during download: {e}")
            
        print(f"Waiting 15 seconds before retry...")
        time.sleep(15)
        attempt += 1

if __name__ == "__main__":
    download()
