# =============================================================================
#  |   Assignment:  StickMan_Suit
#  |       Author:  joaopaulof.soares@gmail.com, davidamurim7@gmail.com
#  |
#  |       Course:  Image Processing
#  |       Date:    2019-Nov
#  |
#  |  Description:  Tracking some LEDs in a suit to generete an StickMan using opencv.
#  |                
#  | Deficiencies:  We had problems with the RED leds. They's bright was not enought to
#  |     the camera to capture. Maybe if we had change them it would work better.
#  |     To test in a brighter room we stopped track them. 
#  |
# ===========================================================================


import numpy as np
import cv2
from operator import itemgetter


#Open webcam 
cap = cv2.VideoCapture(0)

c1=[0,0,0]
c2=[0,0,0]
c3=[0,0,0]
c4=[0,0,0]
c5=[0,0,0]
c6=[0,0,0]
c7=[0,0,0]
c8=[0,0,0]
c9=[0,0,0]
c10=[0,0,0]
c11=[0,0,0]


# ---------------------------------------------------------------------
#  |  Method ligaPontos
#  |
#  |  Purpose:  Draw a line between p1 and p2 in a Frame
#  |  None  
# -------------------------------------------------------------------
def ligaPontos(p1,p2):
    cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (0,0,0))

# /*---------------------------------------------------------------------
# |  Method desenhaBoneco
# |
# |  Purpose:  Organize a list of point to plot each part of the stickman 
# |     in the right place of a white frame.
# *-------------------------------------------------------------------*/    
def desenhaBoneco(pontos):
    pontoTroncoMeio = 0
    k = 0
    #connect body points
    backup = 0
    pontos = sorted(pontos, key=itemgetter(1))
    for i in range(len(pontos)):
        if (pontos[i][2] == 0):
            cv2.circle(img, (pontos[i][0], pontos[i][1]), 5, (0,0,255), -1)
            k = k + 1
            if(k == 1):
                cv2.circle(img, (pontos[i][0], pontos[i][1]-30), 30, (0,0,0))
            if(k == 2):
                pontoTroncoMeio = pontos[i]
            if(backup != 0):
                ligaPontos(backup, pontos[i])
            backup = pontos[i]
            
    #connect arms and legs
    k = 0
    bracoE = [img.shape[1]-1, 0 , 0]
    pernaE = [img.shape[1]-1, 0 , 0]
    bracoD = [0,0,0]
    pernaD = [0,0,0]
    for i in range(len(pontos)):
        if (pontos[i][2] == 1):
            cv2.circle(img, (pontos[i][0], pontos[i][1]), 5, (0,255,0), -1)
            k = k + 1
            if(k <= 2):
                ligaPontos(pontoTroncoMeio, pontos[i])
                if(pontos[i][0] < bracoE[0]):
                    bracoE = pontos[i]
                if(pontos[i][0] > bracoD[0]):
                    bracoD = pontos[i]
            else:
                ligaPontos(backup, pontos[i])
                if(pontos[i][0] < pernaE[0]):
                    pernaE = pontos[i]
                if(pontos[i][0] > pernaD[0]):
                    pernaD = pontos[i]
     
    #connect left hand and foot to the body
    k = 0
    for i in range(len(pontos)):
        if (pontos[i][2] == 2):
            cv2.circle(img, (pontos[i][0], pontos[i][1]), 5, (255,0,0), -1)
            k = k + 1
            if(k == 1):
                ligaPontos(bracoE, pontos[i])
            else:
                ligaPontos(pernaE, pontos[i])
                
    
    #connect right hand and foot to the body
    k = 0
    for i in range(len(pontos)):
        if (pontos[i][2] == 3):
            cv2.circle(img, (pontos[i][0], pontos[i][1]), 5, (0,255,255), -1)
            k = k + 1
            if(k == 1):
                ligaPontos(bracoD, pontos[i])
            else:
                ligaPontos(pernaD, pontos[i])
#======================================================================================================
# /*---------------------------------------------------------------------
# |  Method findMaxValueCount
# |
# |  Purpose:  To find the contour with max area, its ID and its area.
# |     
# |  Parameters:
# |	c -- list of contours
# |
# |  Returns:  the contour with max area, its id and its area
# *-------------------------------------------------------------------*/    
def findMaxValueCount(c):
    maxArea = cv2.contourArea(c[0])
    contourId = 0
    i = 0
    for cnt in c:
        if maxArea < cv2.contourArea(cnt):
            maxArea = cv2.contourArea(cnt)
            contourId = i
        i += 1
    cnt = c[contourId]
    return cnt, contourId,maxArea

# /*---------------------------------------------------------------------
# |  Method desenhaRetangulo
# |
# |  Purpose:  Draw a rectangle around the bright of the detected Leds. If 
# |     the braight is to small, the rectangle is 
# |
# *-------------------------------------------------------------------*/    
def desenhaRetangulo(cntT, maxArea):
    x2,y2,w2,h2 = cv2.boundingRect(cntT)
    cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255,0), 2)
    center2 = (x2,y2)
    if(maxArea < 100.0):
        cv2.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(0,0,255),3)
        return center2
    return (0,0)

while (True):
    #read a new frame from webcam
    _, frame = cap.read()
    frame = cv2.flip(frame,1)

    #convert img from BGR to HSV
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #RED
    lowerRed = np.array([161, 155, 84])
    upperRed = np.array([179, 255, 255])

    #GREEN
    lowerGreen = np.array([40, 100, 84])
    upperGreen = np.array([80, 255, 255])

    #BLUE
    lowerBlue = np.array([94, 100, 84])
    upperBlue = np.array([126, 255, 255])

    #YELLOW
    lowerYellow = np.array([20, 100, 84])
    upperYellow = np.array([30, 255, 255])

    #CFG TO LED RED
    mask = cv2.inRange(hsvImage, lowerRed, upperRed)
    result = cv2.bitwise_and(frame, frame, mask = mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)    
    _,gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    #CFG TO LED GREEN
    maskGreen = cv2.inRange(hsvImage, lowerGreen, upperGreen)
    resultGreen = cv2.bitwise_and(frame, frame, mask = maskGreen)
    grayGreen = cv2.cvtColor(resultGreen, cv2.COLOR_BGR2GRAY)    
    _,grayGreen = cv2.threshold(grayGreen, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contoursGreen, hierarchyGreen = cv2.findContours(grayGreen, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #CFG TO LED BLUE
    maskBlue = cv2.inRange(hsvImage, lowerBlue, upperBlue)
    resultBlue = cv2.bitwise_and(frame, frame, mask = maskBlue)
    grayBlue = cv2.cvtColor(resultBlue, cv2.COLOR_BGR2GRAY)    
    _,grayBlue = cv2.threshold(grayBlue, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contoursBlue, hierarchyBlue = cv2.findContours(grayBlue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #CFG TO LED YELLOW
    maskYellow = cv2.inRange(hsvImage, lowerYellow, upperYellow)
    resultYellow = cv2.bitwise_and(frame, frame, mask = maskYellow)
    grayYellow = cv2.cvtColor(resultYellow, cv2.COLOR_BGR2GRAY)    
    _,grayYellow = cv2.threshold(grayYellow, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contoursYellow, hierarchyYellow = cv2.findContours(grayYellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #if exist contour for the RED LED
    if contours:
        #verificadores
        v2=0
        v3=0
        
        #points
        c1=[0,0,0]
        c2=[0,0,0]
        c3=[0,0,0]
        
        cnt, idRemove,maxArea1 = findMaxValueCount(contours)
        del contours[idRemove]
        if(contours):
            v2=1
            cnt2, idRemove,maxArea2 = findMaxValueCount(contours)
            del contours[idRemove]
        if(contours):
            v3=1
            cnt3, idRemove,maxArea3 = findMaxValueCount(contours)
            del contours[idRemove]
                    
        #return a rectangle on the contour
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255,0), 2)
        c1 = (x,y)        
        #print the position of the red led
        #print ("centro_1 : ",c1)        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        
        #draw the rect to red point
        if(v2==1 ):
            c2 = desenhaRetangulo(cnt2,maxArea2)
            #print("centro_2 : ", c2)
            
        #draw the rect to red point
        if(v3==1 ):
            c3 = desenhaRetangulo(cnt3,maxArea3)
            #print("centro_3 : ", c3)

    #if exist contour for the GREEN LED
    if contoursGreen:
        
        #verificadores
        v5=0
        v6=0
        v7=0
        
        #pontos
        c4=[0,0,0]
        c5=[0,0,0]
        c6=[0,0,0]
        c7=[0,0,0]
        
        cnt4, idRemove, maxArea4 = findMaxValueCount(contoursGreen)
        del contoursGreen[idRemove]
        if(contoursGreen):
            v5=1
            cnt5, idRemove,maxArea5 = findMaxValueCount(contoursGreen)
            del contoursGreen[idRemove]
        if(contoursGreen):
            v6=1
            cnt6, idRemove, maxArea6 = findMaxValueCount(contoursGreen)
            del contoursGreen[idRemove]
        if(contoursGreen):
            v7=1
            cnt7, idRemove, maxArea7 = findMaxValueCount(contoursGreen)
            del contoursGreen[idRemove]
                    
        #return a rectangle on the contour
        x4,y4,w4,h4 = cv2.boundingRect(cnt4)
        cv2.rectangle(frame, (x4, y4), (x4 + w4, y4 + h4), (0, 255,0), 2)
        c4 = (x4,y4)        

        #print ("centro_4 : ",c4)        
        cv2.rectangle(frame,(x4,y4),(x4+w4,y4+h4),(0,0,255),3)
        
        if(v5==1 ):
            c5 = desenhaRetangulo(cnt5,maxArea5)
            #print("centro_5 : ", c5)
            

        if(v6==1 ):
            c6 = desenhaRetangulo(cnt6,maxArea6)
           # print("centro_6 : ", c6)

        if(v7==1 ):
            c7 = desenhaRetangulo(cnt7,maxArea7)
            #print("centro_7 : ", c7)

    #if exist contour for the BLUE LED
    if contoursBlue:
        #verificadores
        v9=0
        #novo
        v10=0
        v11=0
        
        #pontos
        c8=[0,0,0]
        c9=[0,0,0]
        c10=[0,0,0]
        c11=[0,0,0]

    img = np.zeros((1200,1200,3),  dtype="uint8")
    img = cv2.bitwise_not(img)

    

    lista = []
    lista = [ [c1[0], c1[1], 0],  [c2[0], c2[1], 0],  [c3[0], c3[1], 0],  [c4[0], c4[1], 1],   [c5[0], c5[1], 1], [c6[0], c6[1], 1],   [c7[0], c7[1], 1],    [c8[0], c8[1], 2],    [c9[0], c9[1], 2],  [c10[0], c10[1], 3],  [c11[0], c11[1], 3]]
    #lista = [ [0,0, 0],  [0,0, 0],  [0,0, 0],  [c4[0], c4[1], 1],   [c5[0], c5[1], 1], [0,0, 1],   [0,0, 1],    [0,0, 2],    [c9[0], c9[1], 2],  [c10[0], c10[1], 3],  [0,0, 3]]

    desenhaBoneco( lista )
    cv2.imshow("a", img)
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
