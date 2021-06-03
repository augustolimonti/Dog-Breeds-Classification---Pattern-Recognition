import shutil
import os

source = "/Users/augustolimonti/Desktop/753_Project_Final/Dog_Dataset"
destination_train = "/Users/augustolimonti/Desktop/753_Project_Final/Dog_Images/Training"
destination_test = "/Users/augustolimonti/Desktop/753_Project_Final/Dog_Images/Testing"
folders = os.listdir(source)

for files in folders:
    path = os.path.join(source, files)
    for root, dirs, files in os.walk(path):
        for filename in files:
            filename = path + "/" + filename
            shutil.copy(filename, "/Users/augustolimonti/Desktop/753_Project_Final/Dog_Images")
