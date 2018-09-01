import cv2
import smtplib 
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import sys
import time
import logging as log
import datetime as dt
from time import sleep
import numpy as np

def callBack(x):
    pass

def checkCrossLine(x, line):
    AbsDistance = abs(x - line)    

    if (AbsDistance <= 4 ):
        return 1
    else:
        return 0

videoName = None
MinCountourArea = 5000  #Adjust ths value according to your usage
BinarizationThreshold = 10  #Adjust ths value according to your usage
sizeMult = 1
prevCounter = -1
curCounter = 0
start = 0
end = 0
timerBool = False
curSizeMult = sizeMult
cascPath = "haarcascade_frontalface_alt2.xml"
cascPath2 = "haarcascade_eye.xml"
cascPath3 = "Nose.xml"
cascPath4 = "Mouth.xml"
# cascPath3 = "haarcascade_profileFace.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(cascPath2)
noseCascade = cv2.CascadeClassifier(cascPath3)
mouthCascade = cv2.CascadeClassifier(cascPath4)
# profileCascade = cv2.CascadeClassifier(cascPath3)
#log.basicConfig(filename='webcam.log',level=log.INFO)

cv2.namedWindow('trackbars',cv2.WINDOW_NORMAL)
videoSwitch = "ON/OFF"
binarySlider = "Binarization Threshold"
noseSwitch = "Nose"
faceSwitch = "Face"
eyeSwitch = "Eyes"
mouthSwitch = "Mouth"
colorMode = "RGB Mode"
greyMode = "Greyscale Mode"
binaryMode = "Binary Mode"
recordSwitch = "Security Mode"
pauseSwitch = "Pause"

cv2.createTrackbar(videoSwitch, 'trackbars', 0, 1, callBack)
cv2.createTrackbar(binarySlider,'trackbars',BinarizationThreshold,50,callBack)
cv2.createTrackbar(faceSwitch,'trackbars',1,1,callBack)
cv2.createTrackbar(eyeSwitch, 'trackbars', 0, 1, callBack)
cv2.createTrackbar(colorMode, 'trackbars', 1, 1, callBack)
cv2.createTrackbar(pauseSwitch, 'trackbars', 0,1,callBack)
cv2.createTrackbar(greyMode, 'trackbars', 1, 1, callBack)
cv2.createTrackbar(binaryMode, 'trackbars', 1, 1, callBack)
cv2.createTrackbar(recordSwitch, 'trackbars', 0, 1, callBack)
cv2.createTrackbar(mouthSwitch, 'trackbars', 0, 1, callBack)
cv2.createTrackbar(noseSwitch, 'trackbars', 0, 1, callBack)

video_capture = cv2.VideoCapture(0)
capWidth = 320
capHeight = 240
video_capture.set(3, sizeMult * capWidth)
video_capture.set(4, sizeMult * capHeight)

out = None

count = 0


#fgbg = cv2.createBackgroundSubtractorMOG2()
referenceFrame = None
counter = 0

for i in range(0,20):
    video_capture.read()
#each iteration is ONE frame of the video
while True:
    if not video_capture.isOpened():
        print('No Camera detected')
        sleep(5)
        continue
        
    trackStatus = cv2.getTrackbarPos(videoSwitch, 'trackbars')
    binaryStatus = cv2.getTrackbarPos(binarySlider, 'trackbars')
    modeStatus1 = cv2.getTrackbarPos(colorMode, 'trackbars')
    faceStatus = cv2.getTrackbarPos(faceSwitch, 'trackbars')
    pauseStatus = cv2.getTrackbarPos(pauseSwitch, 'trackbars')
    recordStatus = cv2.getTrackbarPos(recordSwitch, 'trackbars')
    eyeStatus = cv2.getTrackbarPos(eyeSwitch, 'trackbars')
    modeStatus2 = cv2.getTrackbarPos(greyMode, 'trackbars')
    modeStatus3 = cv2.getTrackbarPos(binaryMode, 'trackbars')
    noseStatus = cv2.getTrackbarPos(noseSwitch, 'trackbars')
    mouthStatus = cv2.getTrackbarPos(mouthSwitch, 'trackbars')

    # Capture frame-by-frame 
    #save frame before binarization
    ret, frame = video_capture.read()
    height = np.size(frame,0)
    width = np.size(frame,1)
    
    #grayscale conversion and applying gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    #Background subtraction and image binarization
    if referenceFrame is None:
        referenceFrame = gray
        continue
    FrameDelta = cv2.absdiff(referenceFrame, gray)
    _, threshHold = cv2.threshold(FrameDelta, BinarizationThreshold, 255, cv2.THRESH_BINARY)
    #threshHold = cv2.adaptiveThreshold(FrameDelta, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    #_,threshHold = cv2.threshold(FrameDelta, 0, BinarizationThreshold, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    threshHold = cv2.dilate(threshHold, None, iterations=3)
    _, cntsList, _ = cv2.findContours(threshHold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objectsOnScreen = 0


    middleLine = int(width / 2)

    #contours each are numpy arrays of (x,y) coordinates of boundary points
    #cntsList is list of those arrays
    for contours in cntsList:
        #ignore small contours
        if cv2.contourArea(contours) < MinCountourArea:
            continue
        if recordStatus == 1:
            if timerBool == False:
                #start = time.time()
                timeStamp = str(time.asctime( time.localtime(time.time()) ))
                videoName = 'security_' + timeStamp + '.avi'
                out = cv2.VideoWriter(videoName,cv2.VideoWriter_fourcc('M','J','P','G'), 15, (sizeMult* capWidth, sizeMult* capHeight))
                timerBool = True
        

        #rectangle around object
        (x, y, w, h) = cv2.boundingRect(contours)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        #if w > 0 and h > 0: (redundant)
        #1 object per loop as size of cntsList will be number of objects in frame
        objectsOnScreen += 1
        

    if timerBool == True:
        out.write(frame)
        count += 1
        print(count)
        #end = time.time()
        #FPS/count = number of seconds in video(if vid_cap is slow, avi will look sped up)
        if count >= 100: #150
            #end = 0
            #start = 0
            count = 0
            out.release()
            timerBool = False
            msg = MIMEMultipart()

            fromaddr = "pythonWebcamCV@gmail.com"
            #enter your email address
            toaddr = "address@email.com"
 
            msg['From'] = fromaddr
            msg['To'] = toaddr
            msg['Subject'] = "Motion Detected on Camera"
             
            body = "Motion Detected at: " + timeStamp + " Footage will be attached at the bottom of this email."
             
            msg.attach(MIMEText(body, 'plain'))
             
            filename = videoName
            #file path to this file
            attachment = open("/Users/sky/desktop/Webcam-Face-Detect/" + videoName, "rb")
             
            part = MIMEBase('application', 'octet-stream')
            part.set_payload((attachment).read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
             
            msg.attach(part)
             
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            #enter your email password
            server.login(fromaddr, "email password")
            text = msg.as_string()
            server.sendmail(fromaddr, toaddr, text)
            server.quit()


    

    # Draw a rectangle around the faces
    cv2.line(frame, (middleLine,0), (middleLine,height), (150, 0, 150), 2)
    faces = faceCascade.detectMultiScale( gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
    eye = eyeCascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 10, minSize=(30,30))
    nose = noseCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 55, minSize = (30,30), maxSize = (85,80))
    mouth = mouthCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 40, minSize = (30,30), maxSize = (85,85))
    # for(wx,wy,ww,wh) in profile:
    #     cv2.rectangle(frame, (wx,wy), (wx+ww, wy+wh), (200,0, 100), 2)

    #text on frame
    #print(objectsOnScreen)
    # cv2.rectangle(frame,(0,0),(int(sizeMult * capWidth/3) + 1,int(sizeMult* capHeight/5) + 1), (255,255,255),1)
    # cv2.rectangle(frame,(1,1),(int(sizeMult * capWidth/3),int(sizeMult * capHeight/5)),(0,0,0),-1)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(frame,'Object Counter = ' + str(counter),(10,20), font, sizeMult * .2 ,(255,255,255),1,cv2.LINE_AA)
    # cv2.putText(frame,'Objects on screen = ' + str(objectsOnScreen), (10, 40), font, sizeMult * .2 ,(255,255,255),1,cv2.LINE_AA)

    # # Display the resulting frame
    # cv2.moveWindow('Video', 100,200)
    # cv2.moveWindow('Video2', 500, 200)
    # cv2.moveWindow('Video3', 900, 200)

    #regular frames
    #cv2.imshow('Video', frame)

    #threshold frames
    #cv2.imshow('Video', threshHold)

    BinarizationThreshold = binaryStatus
    if faceStatus == 1:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            xCenter = int((x+x+w)/2)
            yCenter = int((y+y+h)/2)
            objectCenter = (xCenter,yCenter)
            cv2.circle(frame, objectCenter, 1, (0, 0, 0), 3)

            if checkCrossLine(xCenter ,middleLine):
                counter += 1
    if eyeStatus == 1:
        for(ex,ey,ew,eh) in eye:
            cv2.rectangle(frame, (ex,ey), (ex+ew, ey+eh), (100,0,0), 2)
    if noseStatus == 1:
        for(nx,ny,nw,nh) in nose:
            cv2.rectangle(frame, (nx,ny), (nx+nw, ny+nh), (120,75,0), 2)
    if mouthStatus == 1:
        for(mx,my,mw,mh) in mouth:
            cv2.rectangle(frame, (mx,my), (mx+mw, my+mh), (40,0,80), 2)
    #print (BinarizationThreshold)

    cv2.rectangle(frame,(0,0),(int(sizeMult * capWidth/3) + 1,int(sizeMult* capHeight/5) + 1), (255,255,255),1)
    cv2.rectangle(frame,(1,1),(int(sizeMult * capWidth/3),int(sizeMult * capHeight/5)),(0,0,0),-1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'Face Counter = ' + str(counter),(10,20), font, sizeMult * .2 ,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(frame,'Motions on screen = ' + str(objectsOnScreen), (10, 40), font, sizeMult * .2 ,(255,255,255),1,cv2.LINE_AA)

    # Display the resulting frame
    cv2.moveWindow('Video', 100,200)
    cv2.moveWindow('Video2', 500, 200)
    cv2.moveWindow('Video3', 900, 200)

    if modeStatus1 == 1:
        #cv2.imshow('Video', FrameDelta)
        cv2.imshow('Video', frame)
        #cv2.setTrackbarPos(colorMode,trackBars,)
    else:
        cv2.destroyWindow('Video')
    if modeStatus2 == 1:
        cv2.imshow('Video2', FrameDelta)
    else:
        cv2.destroyWindow('Video2')
    if modeStatus3 == 1:
        cv2.imshow('Video3', threshHold)
    else:
        cv2.destroyWindow('Video3')
    #match binary

    cv2.waitKey(15)
    if trackStatus == 1:
        break

    referenceFrame = gray
    while(pauseStatus):
        cv2.waitKey(1)
        pauseStatus = cv2.getTrackbarPos(pauseSwitch, 'trackbars')


    #cv2.imshow('Video', frame)

    # Display the resulting frame
# When everything is done, release the capture
if out != None:
    out.release()
video_capture.release()
cv2.destroyAllWindows()
