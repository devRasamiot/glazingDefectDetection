import RPi.GPIO as GPIO
import json
import time
import picamera
import numpy as np
from PIL import Image as pilImage

class RasamIPUtils():
    def __init__(self):
        self.configAddr = "./RCIP.json"
        self.defaultConfig = self.loadConfig()
        self.defaultCamera = None
    
    def getConfig(self):
        return self.defaultConfig
    
    def sensorInit(self,sensor = 18):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(sensor,GPIO.IN)
        print ("IR Sensor Ready.....")
        print (" ")
        return GPIO.input,sensor

    def loadConfig(self,addr = None):
        if addr is None:
            addr = self.configAddr
        jsonFile =  open(addr)
        data = json.load(jsonFile)
        return data

    def cameraInit(self,config = None):
        if config is None:
            config = self.defaultConfig
        camera = picamera.PiCamera()
        camera.resolution = (config["camera"]["resh"], config["camera"]["resw"])
        camera.shutter_speed =config["camera"]["shutter_speed"]
        camera.iso = config["camera"]["iso"]
        # Camera warm-up time
        print ("Camera warmup.....")
        time.sleep(2)
        print ("Camera Ready.....")
        print (" ")
        self.defaultCamera = camera
        return camera

    def cameraCVCapture(self,config = None, camera = None):
        if config is None:
            config = self.defaultConfig
        if camera is None:
            camera = self.defaultCamera
        image = np.empty((config["camera"]["resh"] * config["camera"]["resw"] * 3,), dtype=np.uint8)
        camera.capture(image,"rgb")
        image = image.reshape((config["camera"]["resw"], config["camera"]["resh"], 3))
        # cropped_image = image[config["camera"]["cropy"]+config["camera"]["cropdimy"],config["camera"]["cropx"]+config["camera"]["cropdimx"]]
        cropped_image = image
        return cropped_image,image
    
    def saveImage(self,processedImage,captureTime,config = None):
        if config is None:
            config = self.defaultConfig
        saveAddr = config["export"]["adr"]
        im= pilImage.fromarray(processedImage)
        im.save(saveAddr+"IMG"+str(captureTime)+".jpeg")
