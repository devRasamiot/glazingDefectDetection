# from sklearn.decomposition import PCA as sklearnPCA
import numpy as np
import cv2
import json
from datetime import datetime



def loadConfig(addr = "./RCIPAlgo.json"):
    jsonFile =  open(addr)
    data = json.load(jsonFile)
    return data

def ImageProcess(img,utilCfg,algoCfg,debugFlag = False):
    imgW = utilCfg["camera"]["resw"]
    imgH = utilCfg["camera"]["resh"]
    print("#### start IP at: "+str(datetime.now()))
    if (debugFlag):
        cv2.imshow("image",cv2.resize(img,(1024,768)))
        cv2.waitKey(0)
    # mainDim = applyPCA(img,imgW,imgH,debugFlag)
    mainDim = applyGrayscale(img,debugFlag)
    print("#### end grayscale at: "+str(datetime.now()))
    croppedCeramic = applyCrop (img,mainDim,algoCfg,imgW,imgH,debugFlag)
    print("#### end Crop at: "+str(datetime.now()))
    defectetionCanvas,ceramicPercent,defectPercent = applyDetect(croppedCeramic,algoCfg,debugFlag)
    print("#### end Defection  at: "+str(datetime.now()))
    return defectetionCanvas,ceramicPercent,defectPercent



# def applyPCA(imgRaw,imgW,imgH,debugFlag = False):
#     img = imgRaw.reshape((imgW*imgH,3))
#     print("#### start PCA at: "+str(datetime.now()))
#     pca = sklearnPCA(n_components=1)
#     pca.fit(img)
#     Main_Dim = pca.transform(img)
#     print("#### start uniforming at: "+str(datetime.now()))
#     minPixel = min(Main_Dim)
#     maxPixel = max(Main_Dim) 
#     Main_Dim = (((Main_Dim - minPixel)/(maxPixel -minPixel))*255.0)
#     Main_Dim = Main_Dim.astype("int")
#     main_img_resize = [item[0] for item in Main_Dim]
#     main_img_resize = np.array(main_img_resize)
#     main_img_resize = main_img_resize.reshape((imgW,imgH))
#     main_img_resize = main_img_resize.astype("uint8")
#     if (debugFlag):
#         cv2.imshow("image",cv2.resize(main_img_resize,(1024,768)))
#         cv2.waitKey(0)
#         print("#### end uniforming at: "+str(datetime.now()))

    return main_img_resize

def applyGrayscale(img,debugFlag = False):
    mainDim = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    mainDim = mainDim.astype("uint8")
    if (debugFlag):
        cv2.imshow("image",cv2.resize(mainDim,(1024,768)))
        cv2.waitKey(0)
    return mainDim 

def applyCrop (img,mainDim,algoCfg,imgW,imgH,debugFlag = False):
    th, thImage = cv2.threshold(
        mainDim,
        algoCfg["mask_threshold"]["min"],
        algoCfg["mask_threshold"]["max"],
        cv2.THRESH_BINARY
        )
    if (debugFlag):
        cv2.imshow("thresholded for Crop",cv2.resize(thImage,(1024,768)))
        cv2.waitKey(0)
    cnts = cv2.findContours(thImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = max(cnts, key = cv2.contourArea)
    mask = np.zeros((imgW,imgH,3), dtype=np.uint8)
    cv2.drawContours(mask, [cnt], 0, (255,255,255), cv2.FILLED)
    result = cv2.bitwise_and(img, mask,mask=None)
    #cutting the ceramic in photo
    x, y, w, h = cv2.boundingRect(cnt)
    # Crop the bounding rectangle out of img
    margin = algoCfg["mask_threshold"]["bounding_margin"]
    out =result[(y-margin):(y+h+margin), (x-margin):(x+w+margin), :].copy()
    if (debugFlag):
        cv2.imshow("cropped",cv2.resize(out,(1024,768)))
        cv2.waitKey(0)
    return out



def applyDetect(cropped,algoCfg,debugFlag = False):
    returnCanvas = cropped.copy()
    edgesImg = cv2.Canny(
        cropped,
        algoCfg["edge_detection"]["canny_min"],
        algoCfg["edge_detection"]["canny_max"]
        )
    th, thresh = cv2.threshold(
        edgesImg,
        algoCfg["edge_detection"]["min_threshold"],
        algoCfg["edge_detection"]["max_threshold"],
        cv2.THRESH_BINARY
        )
    if (debugFlag):
        cv2.imshow("threshold for defects",cv2.resize(thresh,(1024,768)))
        cv2.waitKey(0)
    
    cnts,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    
    
    areaerror = 0
    num=0

    
    if (debugFlag):
        print(hierarchy)
            
    limerror=algoCfg["countur_detection"]["min_area_percent"]
    defect_type=algoCfg["countur_detection"]["defect_type"]
    if defect_type=="ALL":
        cv2.drawContours(returnCanvas,  cnts, -1, (0,255,0), 1, cv2.LINE_AA)
        cnt = max(cnts, key = cv2.contourArea)
        biggestArea =  cv2.contourArea(cnt)
        for c in cnts:
            if c.shape!=cnt.shape:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(returnCanvas, (x, y), (x + w, y + h), (0, 0, 255), 1)
                areaerror=cv2.contourArea(c)+areaerror


    if defect_type=="BIGEST":

        for c in cnts:
            
            if hierarchy[0][num][3] ==-1:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.drawContours(returnCanvas,  [c], -1, (0,255,0), 1, cv2.LINE_AA)
                biggestArea=cv2.contourArea(c)
                
    
                
        if hierarchy[0][num][3] ==1:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.drawContours(returnCanvas,  [c], -1, (255,255,0), 1, cv2.LINE_AA)
            areaerrorfec=cv2.contourArea(c)
            p3=(areaerrorfec/biggestArea)*100
            
            if  p3>limerror:
                
                cv2.rectangle(returnCanvas, (x, y), (x + w, y + h), (0, 255,255), 1)
                areaerror=areaerror+areaerrorfec
                
            

        num=num+1


    areapic=cropped.shape[0]*cropped.shape[1]
    defectPercent=round(areaerror/biggestArea,2)*100
    ceramicPercent=round(biggestArea/areapic,2)*100
    if (debugFlag):
        cv2.imshow("final",cv2.resize(returnCanvas,(1024,768)))
        cv2.waitKey(0)
    
    return returnCanvas,ceramicPercent,defectPercent

     