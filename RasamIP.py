import threading, Queue 
from datetime import datetime
from RasamIPUtils import RasamIPUtils
import time
import RasamIPAlgo as algo

#Init
utils = RasamIPUtils()
Trigger,sensor = utils.sensorInit()
camera = utils.cameraInit()

# Q cration
imageProcessQ = Queue.Queue()
imageSaveQ = Queue.Queue()
algoConfig = algo.loadConfig()

def imageCaptureBySensor() :
    while True:
        if not Trigger(sensor):
            cvImage,_ = utils.cameraCVCapture()
            imageProcessQ.put((datetime.now(),cvImage))
            print("capture at ="+str(datetime.now()))
        while not Trigger(sensor):
            time.sleep(0.2)

def processImage() :
    while True:
        captureTime,cvImage = imageProcessQ.get()
        print("cptrd at:"+str(captureTime)+" prc start at:"+str(datetime.now()))
        processedImage,ceramicPercent,defectPercent = algo.ImageProcess(cvImage,utils.getConfig(),algoConfig,True)
        print("cptrd at:"+str(captureTime)+" prc end at:"+str(datetime.now()))
        imageSaveQ.put((captureTime,processedImage))
        imageProcessQ.task_done()

def saveImage():
    while True:
        captureTime,processedImage = imageSaveQ.get()
        print("cptrd at:"+str(captureTime)+" save start at:"+str(datetime.now()))
        utils.saveImage(processedImage,captureTime)
        print("cptrd at:"+str(captureTime)+" save end at:"+str(datetime.now()))
        imageSaveQ.task_done()

imageCaptureThread = threading.Thread(target=imageCaptureBySensor)
processImageThread = threading.Thread(target=processImage)
saveImageThread = threading.Thread(target=saveImage)

imageCaptureThread.start()
processImageThread.start()
saveImageThread.start()