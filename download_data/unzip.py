import zipfile

path = "/data/mars/data.zip"
target_path = "/data/mars/"
with zipfile.ZipFile(path, 'r') as zip_ref:
    zip_ref.extractall(target_path)
