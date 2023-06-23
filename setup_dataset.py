import subprocess
from tqdm import tqdm

# Create directories for the CelebA dataset
subprocess.run(["mkdir", "-p", "datasets/celeba"], check=True)

# Download the CelebA dataset
subprocess.run(["kaggle", "datasets", "download", "-d", "jessicali9530/celeba-dataset"], check=True)

# Unzip the dataset into the 'celeba' directory
with tqdm(total=100, desc="Unzipping dataset") as pbar:
    subprocess.run(["unzip", "-d", "datasets/celeba", "celeba-dataset.zip"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pbar.update(100)
