'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import cv2
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt

class BinDetector():
	def __init__(self):
		'''
			Initilize your bin detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		pass

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		mu = np.array([[0.12305833, 0.26567634, 0.64023492],
		[0.42882381, 0.57094613, 0.75512574],
		[0.28762304, 0.34534259, 0.21421107],
		[0.54224106, 0.43418504, 0.38058496]])

		var = np.array([[0.01222447, 0.02025565, 0.03034596],
		[0.06225271, 0.05990138, 0.05171289],
		[0.03851334, 0.03714906, 0.01906854],
		[0.04964767, 0.04069557, 0.03690298]])

		theta = np.array([0.2616719792141001, 0.2091179385530228, 0.23882356092465112, 0.29038652130822595]).reshape(-1,1)

		def pdfunc(x, mu, var):
			return np.exp((-1*(x-mu)**2)/(2*var))*(1/(2*3.14*var)**0.5)


		maxap = np.zeros((4,1))
		X_flat = img.reshape(img.size//3,3)/255
		y = np.zeros((X_flat.shape[0],1))
		for k in range(y.shape[0]):   
			for i in range(4): # i is for class number
				prob = 1
				for j in range(3): # j is for r/g/b
					prob *= pdfunc(X_flat[k][j], mu[i][j], var[i][j])
				maxap[i] = prob * theta[i]
			out = np.argmax(maxap) + 1
			y[k] = out

		bin = y.reshape((img.shape[0], img.shape[1]))
		bin[bin > 1] = 0
		mask_img = bin 

		# YOUR CODE BEFORE THIS LINE
		################################################################
		return mask_img

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		mask_labels = label(np.asarray(img))
		props = regionprops(mask_labels)
		boxes = []
		for prop in props:
			h=abs(prop.bbox[0]-prop.bbox[2])
			w=abs(prop.bbox[1]-prop.bbox[3])
			if 1.3*h>w and 40000>h*w>10000:
				boxes.append([prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]])
		# YOUR CODE BEFORE THIS LINE
		################################################################
		
		return boxes


