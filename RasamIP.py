import threading, Queue
from datetime import datetime
from RasamIPUtils import RasamIPUtils
import time
import RasamIPAlgo as algo
import traceback
import RasamAnDAlgo as randAlgo
#Init
utils = RasamIPUtils()
camera = utils.cameraInit()

# Q creation
imageProcessQ = Queue.Queue()
imageSaveQ = Queue.Queue()
algoConfig = algo.loadConfig()
randAlgoConfig = randAlgo.loadConfig()

def imageCaptureBySensor(channel) :
    try:
        cvImage = utils.cameraCVCapture()
        imageProcessQ.put((datetime.now(),cvImage))
        print("capture at ="+str(datetime.now()))
        imageSaveQ.put((datetime.now(),cvImage))
    except:
        traceback.print_exc()

def processImage():
    while True:
        captureTime,cvImage = imageProcessQ.get()
        try:
            print("cptrd at:"+str(captureTime)+" prc start at:"+str(datetime.now()))
            # processedImage,ceramicPercent,defectPercent = algo.ImageProcess(cvImage,utils.getConfig(),algoConfig,True)
            processedImage,_ = randAlgo.AngelDetectionAlgo(cvImage,utils.getConfig(),randAlgoConfig)
            print("cptrd at:"+str(captureTime)+" prc end at:"+str(datetime.now()))
            imageSaveQ.put((captureTime,processedImage))
        except:
            traceback.print_exc()
        imageProcessQ.task_done()

def saveImage():
    while True:
        captureTime,processedImage = imageSaveQ.get()
        try:
            print("cptrd at:"+str(captureTime)+" save start at:"+str(datetime.now()))
            utils.saveImage(processedImage,captureTime)
            print("cptrd at:"+str(captureTime)+" save end at:"+str(datetime.now()))
        except:
                traceback.print_exc()
        imageSaveQ.task_done()

utils.sensorInit(imageCaptureBySensor)

imageCaptureThread = threading.Thread(target=imageCaptureBySensor,args=[1])
processImageThread = threading.Thread(target=processImage)
saveImageThread = threading.Thread(target=saveImage)

imageCaptureThread.start()
processImageThread.start()
saveImageThread.start()

import signal              
signal.signal(signal.SIGINT, utils.signal_handler)
signal.pause()