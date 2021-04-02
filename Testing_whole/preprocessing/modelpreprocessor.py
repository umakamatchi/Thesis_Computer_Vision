from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import streamlit as st
from PIL import Image, ImageOps
from preprocessing.boundingboxes import BoundingBoxes
from preprocessing.prediction import prediction

class ModelPreProcessor:
	def process(self, dataSet, typeToDetect, model, classLabels):				
		data = dataSet	
		data = data.astype("float")/255.0				
		# load the CNN_color-trained network		
		model = load_model(model)
		# Predicting
		preds = model.predict(data, batch_size=32).argmax(axis=1)
		image_index=[]				
		i=0
		for pred in preds: 				
			if classLabels[pred].lower()==typeToDetect.lower():	
				#Append predictions to result object
				image_index.append(prediction(i,classLabels[pred]))
			i=i+1		
		return image_index

