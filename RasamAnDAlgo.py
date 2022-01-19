import numpy as np
import cv2
import json
from datetime import datetime
import math

def loadConfig(addr = "./RCIPAnD.json"):
    jsonFile =  open(addr)
    data = json.load(jsonFile)
    return data

num=0

###main function###

def AngelDetectionAlgo(img,utilCfg,algoCfg,debugFlag = False,persCalibrationmode=False,cropCalibrationmode=False):

    print("#### startAnD at: "+str(datetime.now()))
    # blur = cv2.blur(img,(3,3))
    # if (debugFlag):
    #     cv2.imshow("BLUR",cv2.resize(blur,(1024,768)))
    #     cv2.waitKey(0)
    # sharpen_kernel=algoCfg["corner_detection"]["sharpen_kernel"]
    # sharpen_kernel=np.array(sharpen_kernel)
    # sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    # if (debugFlag):
    #         cv2.imshow("sharpen",cv2.resize(sharpen,(1024,768)))
    #         cv2.waitKey(0)
    # img = sharpen
    
    
    defisheyeimage=DefisheyeImage(img,algoCfg,debugFlag)
    if(debugFlag):
        cv2.imwrite("undistorted.jpg",defisheyeimage)
    # defisheyeimage=img
    if (persCalibrationmode):
        # pointsList=[[903,468],[1658,367],[891,1172],[1720,1232]]
        # cv2.circle(defisheyeimage,(903,468),5,(0,0,255),cv2.FILLED)
        # cv2.circle(defisheyeimage,(1658,367),5,(0,255,0),cv2.FILLED)
        # cv2.circle(defisheyeimage,(891,1172),5,(255,0,0),cv2.FILLED)
        # cv2.circle(defisheyeimage,(1720,1232),5,(255,255,0),cv2.FILLED)
        # cv2.imshow("deg",cv2.resize(defisheyeimage,(1024,768)))
        # cv2.waitKey(0)
        maskPoint = cuttingconveyor(defisheyeimage)
        result,mask=maskonimage(defisheyeimage,algoCfg,debugFlag,maskPoint = maskPoint)
        approx1,approx2=CornerDetection( result,mask,algoCfg,debugFlag,maskPoint = maskPoint)
        pointsList = findOrderedTileCorners(approx1,approx2)
        # pointsList=[[903,468],[1658,367],[1720,1232],[891,1172]]

        nopers = getPerspectiveTransformMatrix(defisheyeimage,pointsList,algoCfg)
        if(debugFlag):
            cv2.imwrite("unperspective.jpg",nopers)
        return

    
    image=UnpersPective(defisheyeimage,algoCfg,debugFlag)
    image = cv2.resize(image,None,fx=0.5,fy=0.5)
    if(debugFlag):
        cv2.imwrite("unperspective.jpg",image)
    
    if (cropCalibrationmode):
        maskPoint = cuttingconveyor(image)
        print (maskPoint)
        return
   
    result,mask=maskonimage(image,algoCfg,debugFlag)
    approx1,approx2=CornerDetection( result,mask,algoCfg,debugFlag)
    pointsList = findOrderedTileCorners(approx1,approx2)

    angle,resImage=CalculateAngle2(image, pointsList,debugFlag )
    img,ratiodiameter=CalculateDiameter(resImage,pointsList,debugFlag)
    print(angle)
    return img,angle,ratiodiameter



def DefisheyeImage(img,algoCfg,debugFlag = False):  
   
    DIM = algoCfg["defisheye_image"]["DIM"]
    K = algoCfg["defisheye_image"]["K"]
    K=np.array(K)
    D = algoCfg["defisheye_image"]["D"]   
    D=np.array(D)
    
    nk=K.copy()
    nk[0,0]=K[0,0]/2
    nk[1,1]=K[1,1]/2
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3),nk,DIM , cv2.CV_16SC2)

    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    if (debugFlag):
        cv2.imshow("Undistort Image",cv2.resize(undistorted_img,(1024,768)))
        cv2.waitKey(0)
    undistorted_img = undistorted_img.astype("uint8")
    print("######end defisheye"+str(datetime.now()))
    return undistorted_img


def UnpersPective(img,algoCfg,debugFlag = False):
    matrix= algoCfg["unperspectiv_image"]["matrix"] 
    if(debugFlag):
        print(matrix)
        print(type(matrix))
    widthImg= algoCfg["unperspectiv_image"]["widthImg"]
    heightImg= algoCfg["unperspectiv_image"]["heightImg"]  
    matrix = np.array(matrix)    
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    print("######end UnpersPective"+str(datetime.now()))

    if (debugFlag):
        cv2.imshow("unperspective Image",cv2.resize(imgWarpColored,(1024,768)))
        cv2.waitKey(0)
    return imgWarpColored


def cuttingconveyor(img):
    pointsList = []
    winw = int(img.shape[1]/2)
    winh = int(img.shape[0]/2)
    image = img.copy()
    image = cv2.resize(image,(winw,winh))
    # cv2.resize(img,(1256,972))
    
    def mousePoints(event,x,y,flags,params):
        if event == cv2.EVENT_LBUTTONDOWN:
            size = len(pointsList)
            cv2.circle(image,(x,y),5,(0,0,255),cv2.FILLED)
            pointsList.append([x,y])
            global num
            num=num+1
            print(num)

    while True:
        
        cv2.imshow('choose point',image)
        cv2.setMouseCallback('choose point',mousePoints)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            pointsList = []
        
            # img = cv2.imread("3032.jpg")
        if num==4:
            print("ok")
            print(pointsList)
            break
    cv2.destroyAllWindows()
    
    return [[p[0]*2,p[1]*2] for p in pointsList]


def maskonimage(img,algoCfg,debugFlag = False,maskPoint = None):
    if maskPoint is None:
        P1=algoCfg["point_mask"]["P1"] 
        P2=algoCfg["point_mask"]["P2"] 
        P3=algoCfg["point_mask"]["P3"] 
        P4=algoCfg["point_mask"]["P4"] 
    else:
        P1=maskPoint[0] 
        P2=maskPoint[1] 
        P3=maskPoint[2]  
        P4=maskPoint[3] 

    # im1 = img[P1[1]:P2[1],P1[0]:P2[0]]
    # im2=img[pointsList[2][1]:pointsList[3][1],pointsList[2][0]:pointsList[3][0]]
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.rectangle(mask, (P1), (P2), (255, 255, 255), -1)
    cv2.rectangle(mask, (P3), (P4), (255, 255, 255), -1)
    result = cv2.bitwise_and(img, mask,mask=None)
    result[np.all(result == (0, 0, 0), axis=-1)] = (0,255,0)
    print("######end maskonimage"+str(datetime.now()))
    if(debugFlag):
        cv2.imshow("mask",cv2.resize(mask,(1024,768)))
        cv2.waitKey(0)
    return result,mask


def CornerDetection(img,mask,algoCfg,debugFlag = False,maskPoint = None):

    sharpen_kernel=algoCfg["corner_detection"]["sharpen_kernel"]
    canny_min=algoCfg["corner_detection"]["canny_min"]
    canny_max=algoCfg["corner_detection"]["canny_max"]
    sharpen_kernel=np.array(sharpen_kernel)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur = cv2.medianBlur(gray, 5)
    # if (debugFlag):
    #         cv2.imshow("blur",cv2.resize(blur,(1024,768)))
    #         cv2.waitKey(0)
    # sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    # if (debugFlag):
    #         cv2.imshow("sharpen",cv2.resize(sharpen,(1024,768)))
    #         cv2.waitKey(0)

    # thresh = cv2.threshold(sharpen,160,255, cv2.THRESH_BINARY_INV)[1]
    # if (debugFlag):
    #     cv2.imshow("thresh",cv2.resize(thresh,(1024,768)))
    #     cv2.waitKey(0)
    
    # canny = cv2.Canny(sharpen ,canny_min,canny_max)
        
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # close = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=2)
    # if (debugFlag):
    #         cv2.imshow("close",cv2.resize(close,(1024,768)))
    #         cv2.waitKey(0)
    
    # img = cv2.cvtColor(close, cv2.COLOR_GRAY2RGB)

    canny = cv2.Canny(img ,canny_min,canny_max)

    # canny = cv2.Canny(img ,canny_min,canny_max)
    # # #################################
    # canny = cv2.Canny(img,canny_min,canny_max)
    # #################################
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # blur = cv2.blur(img,(3,3))
    # if (debugFlag):
    #     cv2.imshow("BLUR",cv2.resize(blur,(1024,768)))
    #     cv2.waitKey(0)
    
    # croppedImage = cv2.Canny(
    #         blur,
    #         algoCfg["ceramic_crop"]["canny_min"],
    #         algoCfg["ceramic_crop"]["canny_max"]
    #         )
    # if (debugFlag):
    #     cv2.imshow("edgesImg",cv2.resize(croppedImage,(1024,768)))
    #     cv2.waitKey(0)
    # th, croppedImage = cv2.threshold(
    #     croppedImage,
    #     algoCfg["ceramic_crop"]["canny_thr_min"],
    #     algoCfg["ceramic_crop"]["canny_thr_max"],
    #     cv2.THRESH_BINARY
    #     )
    # if (debugFlag):
    #     cv2.imshow("canny thresholded",cv2.resize(croppedImage,(1024,768)))
    #     cv2.waitKey(0)
    # canny = croppedImage

    # sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
    # thresh = cv2.threshold(gray,160,255, cv2.THRESH_BINARY)[1]
   
    #??????
    
    if (debugFlag):
        cv2.imshow("cannyImage",cv2.resize(canny,(1024,768)))
        cv2.waitKey(0)
    cnts,_ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    innerCnts = []
    outerCnts = []
    if(maskPoint is None):
        marginPoint1 = (algoCfg["point_mask"]["P1"][0] + algoCfg["corner_detection"]["point_mask_margin"],
            algoCfg["point_mask"]["P1"][1] + algoCfg["corner_detection"]["point_mask_margin"])
        marginPoint4 = (algoCfg["point_mask"]["P4"][0] - algoCfg["corner_detection"]["point_mask_margin"],
            algoCfg["point_mask"]["P4"][1] - algoCfg["corner_detection"]["point_mask_margin"])
    else:
        marginPoint1 = (maskPoint[0][0] + algoCfg["corner_detection"]["point_mask_margin"],
            maskPoint[0][1] + algoCfg["corner_detection"]["point_mask_margin"])
        marginPoint4 = (maskPoint[3][0] - algoCfg["corner_detection"]["point_mask_margin"],
            maskPoint[3][1] - algoCfg["corner_detection"]["point_mask_margin"])

    # for cntWalk in cnts:
    #     if(cv2.pointPolygonTest(cntWalk,marginPoint1,False) >= 0):
    #         continue
    #     if(cv2.pointPolygonTest(cntWalk,marginPoint4,False) >= 0):
    #         continue
    #     innerCnts.append(cntWalk)
 
    # print(str(len(innerCnts))+":"+str(len(cnts)))
    # cnts = sorted(innerCnts, key = cv2.contourArea,reverse=True)
    # if(debugFlag):
    #     for i,cWalk in enumerate(cnts):
    #         image=img.copy()
    #         cv2.drawContours(image, [cWalk], -1, (0, 0, 255), 5) 
    #         # print ("hierarchy##########\n",h[0][i])
    #         cv2.imshow(str(i),cv2.resize(image,(1024,768)))
    #         cv2.waitKey(0)
    #         if( i>=10):
    #             break

        
    for cntWalk in cnts:
        if(cv2.pointPolygonTest(cntWalk,marginPoint1,False) >= 0):
            outerCnts.append(cntWalk)
            continue
        if(cv2.pointPolygonTest(cntWalk,marginPoint4,False) >= 0):
            outerCnts.append(cntWalk)

    cnts = sorted(outerCnts, key = cv2.contourArea,reverse=False)
    cnts = cnts[:-2]
    if(debugFlag):
        for i,cWalk in enumerate(cnts):
            image=img.copy()
            cv2.drawContours(image, [cWalk], -1, (0, 0, 255), 5) 
            # print ("hierarchy##########\n",h[0][i])
            # cv2.imshow(str(i),cv2.resize(image,(1024,768)))
            # cv2.waitKey(0)
            if( i>=10):
                break
    cv2.drawContours(mask, cnts, -1, color=(0, 0,0), thickness=cv2.FILLED)
    graymask =  cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
    cnts,_ = cv2.findContours(graymask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea,reverse=True)
    if(debugFlag):
        cv2.imshow("innercnts",cv2.resize(mask,(1024,768)))
        cv2.waitKey(0)
        for j,cWalk in enumerate(cnts):
            image=img.copy()
            cv2.drawContours(image, [cWalk], -1, (0, 0, 255), 5) 
            # print ("hierarchy##########\n",h[0][i])
            cv2.imshow(str(j),cv2.resize(image,(1024,768)))
            cv2.waitKey(0)
            if( j>=10):
                break
 
    peri = cv2.arcLength(cnts[0], True) 
    approx1 = cv2.approxPolyDP(cnts[0], 0.01 * peri, True)
    peri = cv2.arcLength(cnts[1], True)
    approx2 = cv2.approxPolyDP(cnts[1], 0.01 * peri, True)
    if(debugFlag):
        print(type(approx1))
    # print(approx2)
    if (debugFlag):
        image=img.copy()
        cv2.drawContours(image, approx1, -1, (0, 0, 255), 5) 
        cv2.drawContours(image, approx2, -1, (255, 0, 255), 5) 
        print("approx")
        print(approx1)
        cv2.imshow("approx",cv2.resize(image,(1024,768)))
        cv2.waitKey(0)
    print("######end CornerDetection"+str(datetime.now()))
    
    return approx1,approx2


def findOrderedTileCorners(pointsList1,pointsList2):
    allpointsList = np.concatenate((pointsList1[:,0,:], pointsList2[:,0,:]), axis=0)
    allpointsList = sorted(allpointsList,key=lambda l:l[1])
    topPoint = np.zeros((2, 2), dtype=np.int32)
    bottomPoint = np.zeros((2, 2), dtype=np.int32)
    topPoint[0]=allpointsList[0]
    topPoint[1]=allpointsList[1]
    bottomPoint[0]=allpointsList[-2]
    bottomPoint[1]=allpointsList[-1]
    topPoint = sorted(topPoint,key=lambda l:l[0])
    bottomPoint = sorted(bottomPoint,key=lambda l:l[0])
    pointsList = np.zeros((4, 2), dtype=np.int32)
    pointsList[0] = topPoint[0]
    pointsList[1] = topPoint[1]
    pointsList[2] = bottomPoint[1]
    pointsList[3] = bottomPoint[0]
    print("######end findOrderedTileCorners"+str(datetime.now()))
    return pointsList


def CalculateAngle(img,pointsList,debugFlag = False):
    angD=[]
    image=img.copy()
    def angle(a, b, c):
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.arccos(cosine_angle)

    for i in range(4):
        ang = angle(pointsList[i-1],pointsList[i],pointsList[(i+1)%4]) 
        angR = round(math.degrees(ang),4)
        if (debugFlag):
            cv2.putText(image,str(angR),(pointsList[i][0]-20,pointsList[i][1]-20),cv2.FONT_HERSHEY_COMPLEX,
                1.5,(0,0,255),2)
        cv2.imshow("angle",cv2.resize(image,(1024,768)))
        cv2.waitKey(0) 
        angD.append(angR)

    print("######end CalculateAngle"+str(datetime.now()))

    return angD,image

def CalculateAngle2(img,pointsList,debugFlag = False):
    angD=[]
    image=img.copy()
  
    # pointslist=[[313,313],[686,313],[686,685],[313,686]]
    # cv2.circle(image,pointsList[1],5,(0,0,255),cv2.FILLED)
    # cv2.circle(image,pointsList[2],5,(0,0,255),cv2.FILLED)
    # cv2.circle(image,pointsList[3],5,(0,0,255),cv2.FILLED)
    # cv2.circle(defisheyeimage,(1658,367),5,(0,255,0),cv2.FILLED)
    # cv2.circle(defisheyeimage,(891,1172),5,(255,0,0),cv2.FILLED)
    # cv2.circle(defisheyeimage,(1720,1232),5,(255,255,0),cv2.FILLED)
    # cv2.imshow("deg",cv2.resize(image,(1024,768)))
    # cv2.waitKey(0)


    def gradient(pt1,pt2):
        return (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
    for i in range(4):
        if(debugFlag):
            cv2.circle(image,pointsList[i],5,(0,0,255),cv2.FILLED)
            cv2.imshow("deg",cv2.resize(image,(1024,768)))
            cv2.waitKey(0)
        # print(i)
   
        m1=gradient(pointsList[i],pointsList[i-1])
        # print("m1")
        # print(m1)
        if i!=3:
            # print(pointsList[i+1])
            m2=gradient(pointsList[i],pointsList[i+1])
            # print("m2")
            # print(m2)
        else:
            m2=gradient(pointsList[i],pointsList[0])
            # print("m2")
            # print(m2)

        if abs(m1)==float('inf'):
            # print("m1 is inffini")
            # print(i)
            ang = abs(math.atan((m2-0)/(1+(m2*0))))
            # print(ang)
            ang=90.00000-ang
            # print(ang)

        if abs(m2)==float('inf'):
            # print("m2is infin")
            # print(i)
            ang = abs(math.atan((0-m1)/(1+(0*m1))))
            ang=90.00000-ang
            # print(ang)
            

        

        if (abs(m1)!=float('inf') and abs(m2)!=float('inf')):  
                # print("m1 and m2")
                # print(m1)
                # print(m2)      
                ang = abs(math.atan((m2-m1)/(1+(m2*m1))))
                ang=math.degrees(ang)
                # print(ang)
     
        angR = round(ang,5)
        if (debugFlag):
            cv2.putText(image,str(angR),(pointsList[i][0]-20,pointsList[i][1]-20),cv2.FONT_HERSHEY_COMPLEX,
                1.5,(0,0,255),2)
        angD.append(angR)


    print(angD)

    sum = 0
    i=0
    for e in angD:
        sum = sum + e

    r1 = (360.0 - sum) / 4.0
    r2 = 360 / sum

    # print ("Additive")
    for e2 in angD:
        additive=e2+r1
        # print (str(e2+r1))
    # print ("multiple")
    for e2 in angD:        
        # print (str(e2*r2))
        k=round((e2*r2),5)
        angD[i]=k
        cv2.putText(img,str(k),(pointsList[i][0]-20,pointsList[i][1]-20),cv2.FONT_HERSHEY_COMPLEX,
                1.5,(0,0,255),2)
        i=i+1

    if(debugFlag):
        cv2.imshow("angele",img)
        cv2.waitKey(0)
    print("######end CalculateAngle"+str(datetime.now()))



    return angD,img


def CalculateDiameter(image,pointsList,debugFlag):
    Diameter1=math.sqrt((pointsList[2][0]-pointsList[0][0])**2+(pointsList[2][1]-pointsList[0][1])**2)
    Diameter2=math.sqrt((pointsList[3][0]-pointsList[1][0])**2+(pointsList[3][1]-pointsList[1][1])**2)
    D=Diameter1/Diameter2
    D=round(D,5)
    cv2.putText(image,str(D),((int((pointsList[1][0]-pointsList[0][0])/2)+int(pointsList[0][0])),(int((pointsList[2][1]-pointsList[0][1])/2)+int(pointsList[0][1]))),cv2.FONT_HERSHEY_COMPLEX,
            1.5,(0,0,255),2)
    if (debugFlag):
        cv2.imshow("Diameter",image)
        cv2.waitKey(0)
    print("######end CalculateDiameter"+str(datetime.now()))
    return image,D


def getPerspectiveTransformMatrix (img,pointsList,algoCfg):
    tileW = algoCfg["unperspectiv_image"]["tileW"]
    tileH = algoCfg["unperspectiv_image"]["tileH"]
    widthImg= algoCfg["unperspectiv_image"]["widthImg"]
    heightImg= algoCfg["unperspectiv_image"]["heightImg"]
    x1 = int((widthImg-tileW)/2)
    y1 = int((heightImg-tileH)/2)
    pts1 = np.float32([
        [pointsList[0][0], pointsList[0][1]],
        [pointsList[1][0], pointsList[1][1]],
        [pointsList[2][0], pointsList[2][1]],
        [pointsList[3][0], pointsList[3][1]]])
    pts2 = np.float32([[x1, y1],[x1+tileW, y1], [x1+tileW, y1+tileH],[x1, y1+tileH]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    cv2.imshow("unperspective Image",cv2.resize(imgWarpColored,(1024,768)))
    cv2.waitKey(0)
    print("perspective matrix:\n")
    print(matrix)
    return imgWarpColored




    





     




    







