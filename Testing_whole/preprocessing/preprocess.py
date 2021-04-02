from preprocessing import ModelPreProcessor

class PreProcess:
	def processColor(self, images, type):
		print(type)
		p1 = ModelPreProcessor()
		return p1.process(images, type, "Colorfinal_VGG.h5",["black", "blue","cyan","gray","green","white","yellow","red"])
		
	def processType(self, images, type):
		print(type)
		p2= ModelPreProcessor()
		return p2.process(images, type, "type_VGG_30.h5",["bus", "truck","motorcycle","van","suv","sedan"])
		
		
	
	
