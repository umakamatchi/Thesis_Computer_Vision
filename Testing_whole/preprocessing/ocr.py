
import cv2
import imutils
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Uma\Anacondanew\envs\uma_tensor_demo\Testing_whole\Tesseract-OCR\tesseract.exe'
import streamlit as st
import string
class ocr:
    def license(self,img, licensePlateInput):
        #img = cv2.imread('C:\\Users\\Uma\\Anacondanew\\envs\\uma_tensor\\OCR_Opencv\\images\\OCR_1.jpg')
        img = cv2.resize(img,(600,400) )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        gray = cv2.bilateralFilter(gray, 13, 15, 15) 
        edged = cv2.Canny(gray, 30, 200) #Perform Edge detection
        contours=cv2.findContours(edged.copy(),cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
        screenCnt = None
        for c in contours:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                break
        # Masking the part other than the number plate
        mask = np.zeros(gray.shape,np.uint8)
        cont_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
        new_image = cv2.bitwise_and(img,img,mask=mask)    
        # Now crop
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx+1, topy:bottomy+1]
        #Read the number plate
        text = pytesseract.image_to_string(Cropped, config='--psm 13')        
        text=text.translate({ord(c): None for c in string.whitespace})
        if(text==licensePlateInput):
            st.image(img, channels="BGR")
            st.image(new_image)
            st.image(Cropped)	
            st.info(licensePlateInput + ' detected')	            
        else:
            st.info(licensePlateInput + ' not detected')

        
   

