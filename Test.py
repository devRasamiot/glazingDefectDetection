import RasamIPAlgo as algo
import cv2
import json
from PIL import Image as pilImage


def loadConfig(addr = "./RCIP.json"):
    jsonFile =  open(addr)
    data = json.load(jsonFile)
    return data

img = cv2.imread("in2.jpeg")
# img = cv2.resize(img,(1920,1920))
pic,cp,dp=algo.ImageProcess(img,loadConfig(),algo.loadConfig(),debugFlag = True)

im= pilImage.fromarray(pic)
im.save("out.jpeg")
cv2.waitKey()
