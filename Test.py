import RasamIPAlgo as algo
import RasamAnDAlgo as randAlgo
import cv2
import json
from RasamIPUtils import RasamIPUtils


def loadConfig(addr = "./RCIP.json"): 
    jsonFile =  open(addr)
    data = json.load(jsonFile)
    return data

utils = RasamIPUtils()
# camera = utils.cameraInit()
# img = utils.cameraCVCapture()

img = cv2.imread("333.jpeg")
# img = cv2.resize(img,(1920,1920))
# pic,cp,dp=algo.ImageProcess(img,loadConfig(),algo.loadConfig(),debugFlag = True)
anggle=randAlgo.AngelDetectionAlgo(img,loadConfig(),randAlgo.loadConfig(),debugFlag = False,persCalibrationmode=False,cropCalibrationmode=False)
# im= pilImage.fromarray(pic)
# im.save("out.jpeg")

cv2.waitKey()
