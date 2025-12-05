import gdown
import os

# Link Google Drive model terkompresi
url = "https://drive.google.com/file/d/1M15AVq0DG8_h1hHzvAWDAIFWXCRlGELd/view?usp=drive_link"  # ganti FILE_ID
output = "model.pkl"

if not os.path.exists(output):
    print("Downloading model...")
    gdown.download(url, output, quiet=False)
else:
    print("Model already exists.")
