# import the necessary packages
#from  import img_to_array
from tensorflow.keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
	
	def preprocess(self, image, dataFormat=None):
		# apply the Keras utility function that correctly rearranges
		# the dimensions of the image
		return img_to_array(image, data_format=dataFormat)