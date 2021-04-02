import streamlit as st
import time
from PIL import Image, ImageOps
import numpy as np
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.preprocess import PreProcess
from tensorflow.keras.preprocessing.image import img_to_array			
import io
import cv2
import tempfile
from preprocessing.boundingboxes import BoundingBoxes
from preprocessing.prediction import prediction

try:
	st.title('Vehicle Color and Type Classification')
	vehicleOption = st.sidebar.selectbox(
		'Choose VehicleType',
		['Sedan','SUV'])
	# 
	colorOption = st.sidebar.selectbox(
		'Choose Color',
		['Red','White','Blue','Yellow','Black','Gray'])
	progress_bar = st.progress(0)
	#upload multiple images	
	file = st.file_uploader("Please upload an image file", type=["jpg", "png"],accept_multiple_files=True)
	Resized_data = []
	Normal_data = []
	if st.button('Submit'):		
		if len(file)>0:		
			#Process uploaded images
			#Get Raw images for bounding box
			#Get preprocessed image collection for prediction
			for uploaded_file in file:			
				tfile = tempfile.NamedTemporaryFile(delete=False)
				tfile.write(uploaded_file.read())		
				image_data = cv2.imread(tfile.name)
				size = (100,100)    
				#Resize image
				image = cv2.resize(image_data, size, interpolation=cv2.INTER_AREA)					
				imageArray = img_to_array(image, data_format=None) 		
				#Raw image for bounding box
				notResizedArray =  img_to_array(image_data, data_format=None) 					
				Resized_data.append(imageArray)		
				Normal_data.append(notResizedArray)			
			#Set numpy collection objects
			notResizedData=np.array(Normal_data)
			data=np.array(Resized_data)
			data2=np.array(Resized_data)
			progress_bar.progress(15)
			#Process Color
			p1=PreProcess()
			st.info("Processing Color : "+ colorOption)
			colorResults = p1.processColor(data,colorOption)		
			type_image_input=[]
			trimmedArray=[]
			bounded_image_index=[]
			if(len(colorResults)>0):
				progress_bar.progress(35)
				st.info("Found " + str(len(colorResults)) +" "+ colorOption + " objects.")
				p2=PreProcess()
				st.info("Processing Vehicle Type : "+ vehicleOption)
				#Process Vehicle Type
				vehicleTypeResults = p2.processType(data2,vehicleOption)			
				if(len(vehicleTypeResults)>0):				
					st.info("Found " + str(len(vehicleTypeResults)) +" " + vehicleOption + " objects.")				
					progress_bar.progress(65)
					for colorPredict in colorResults:
						for vehicleTypePredict in vehicleTypeResults:						
							if(colorPredict.label.lower()==colorOption.lower() and vehicleTypePredict.label.lower()==vehicleOption.lower()):
								if(colorPredict.index not in bounded_image_index):								
									bounded_image_index.append(colorPredict.index)
									bounding=BoundingBoxes()
									bounding.box(notResizedData[colorPredict.index], colorPredict.label + " " + vehicleTypePredict.label)
					progress_bar.progress(100)
				else:
					st.warning("No images of " + vehicleOption + " detected by model.")							
					progress_bar.progress(0)
			else:
				st.warning("No images of " + colorOption + " color detected by model.")		
				progress_bar.progress(0)
		else:
			st.warning("Please upload images")
			progress_bar.progress(0)
except:
	st.error("Unexpected error:", sys.exc_info()[0])		
	progress_bar.progress(0)


