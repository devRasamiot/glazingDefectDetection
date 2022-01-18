import RasamIPAlgo as algo
import RasamAnDAlgo as randAlgo
import cv2
import json
from PIL import Image as pilImage


def loadConfig(addr = "./RCIP.json"): 
    jsonFile =  open(addr)
    data = json.load(jsonFile)
    return data

img = cv2.imread("in9.jpg")
# img = cv2.resize(img,(1920,1920))
# pic,cp,dp=algo.ImageProcess(img,loadConfig(),algo.loadConfig(),debugFlag = True)
anggle=randAlgo.AngelDetectionAlgo(img,loadConfig(),randAlgo.loadConfig(),debugFlag = True,persCalibrationmode=False,cropCalibrationmode=False)
# im= pilImage.fromarray(pic)
# im.save("out.jpeg")

cv2.waitKey()
