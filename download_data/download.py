import wget
import sys
import os


def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


newpath = r'/data/mars/'
if not os.path.exists(newpath):
    os.makedirs(newpath)

print("directory created")

url = "https://ordo.open.ac.uk/ndownloader/articles/7352270/versions/1"
save_path = "/data/mars/data.zip"
wget.download(url, save_path, bar=bar_progress)

print("download completed")
