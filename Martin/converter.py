import glob
import os

from PIL import Image

datasetPath = ""  # Enter your path
for file in sorted(glob.glob(datasetPath + "\\*\\*.png")):
    if (file[file.__len__() - 4:] == ".png"):
        print("File name is : " + file)
        ima = Image.open(file)
        rgb_im = ima.convert('RGB')
        fileToDelete = file
        file = file[:file.__len__() - 4]
        rgb_im.save(file + ".jpg")
        print("delete " + fileToDelete)
        os.remove(fileToDelete)