import RasamIPAlgo as algo
import cv2
import json

def loadConfig(addr = "./RCIP.json"):
    jsonFile =  open(addr)
    data = json.load(jsonFile)
    return data


img = cv2.imread("./test.jpeg")
pic,cp,dp=algo.ImageProcess(img,loadConfig(),algo.loadConfig())
showpic = cv2.resize(pic,(1024,768))
cv2.imshow("finalimage",showpic)
cv2.waitKey()