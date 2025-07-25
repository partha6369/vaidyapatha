import os
import gdown
import pandas as pd

def download_file_if_missing(file_name, file_id):
    """
    Downloads a file from Google Drive if it doesn't already exist locally.

    Parameters:
        file_name (str): Path to save the file locally.
        file_id (str): Google Drive file ID.
    """
    # Ensure the directory exists
    dir_path = os.path.dirname(file_name)
    if dir_path != "/data":  # Avoid trying to create /data
        os.makedirs(dir_path, exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(file_name):
        print(f"📥 Downloading {file_name}...")
        gdown.download(url, file_name, quiet=False)
    else:
        print(f"✅ {file_name} already exists. Skipping download.")
        
def download_and_load_csv(file_name, file_id):
    download_file_if_missing(file_name, file_id)
    try:
        df = pd.read_csv(file_name)
        print(f"{file_name} loaded successfully.")
        print("Shape:", df.shape)
        return df
    except Exception as e:
        print(f"Error reading {file_name}:", e)
        return None

def lookup_version_file_id(data_file_name, version_file_ids):
    try:
        return version_file_ids[data_file_name]
    except KeyError:
        raise ValueError(f"❌ No version file ID mapped for: {data_file_name}")
        
def get_local_file_id(file_name):
    fileid_path = file_name + ".fileid"
    if os.path.exists(fileid_path):
        with open(fileid_path, "r") as f:
            return f.read().strip()
    return None

def write_local_file_id(file_name, file_id):
    fileid_path = file_name + ".fileid"
    with open(fileid_path, "w") as f:
        f.write(file_id)

def check_and_update(data_file_name, data_file_id, version_file_ids):
    version_file_id = lookup_version_file_id(data_file_name, version_file_ids)
    current_file_id = get_local_file_id(data_file_name)

    if current_file_id != version_file_id:
        print(f"🔄 Detected new version (or not tracked). Downloading from version_file_id: {version_file_id}")
        gdown.download(id=version_file_id, output=data_file_name, quiet=False, fuzzy=True)
        write_local_file_id(data_file_name, version_file_id)
    elif not os.path.exists(data_file_name):
        print(f"📁 File missing. Downloading using original file_id: {data_file_id}")
        download_file_if_missing(data_file_name, data_file_id)
        write_local_file_id(data_file_name, data_file_id)
    else:
        print(f"✅ File is up to date: {data_file_name}")