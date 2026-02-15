import requests
import os
import sys

# URL for a sample video (Intel IoT DevKit sample)
VIDEO_URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
SAVE_PATH = "data/sample_video.mp4"

def download_file(url, save_path):
    print(f"Downloading from {url}...")
    try:
        if not os.path.exists("data"):
            os.makedirs("data")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)
                
        print(f"Download complete! Saved to {save_path}")
        print(f"File size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"Error downloading file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_file(VIDEO_URL, SAVE_PATH)
