import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import os

def get_kernel(size, sigma):
	""" Get the filter with the Gaussian formula applied to its original values. """ 

	# Creating a vector of the desired size and evenly spaced
	kernel = np.linspace(-(size // 2), size // 2, size)

	# Calculate the gaussian for each vector element
	for i in range(size):
		kernel[i] = 1 / (np.sqrt(2 * np.pi) * sigma) * np.e ** (-np.power((kernel[i]) / sigma, 2) / 2)

	# Transform the vector into a matrix, to use in in the convolution process
	kernel = np.outer(kernel.T, kernel.T)

	# Normalizing the kernel
	kernel *= 1.0 / kernel.max()
	return kernel

def gaussian_blur(image, filter_size, color=True):
	""" Perform Gaussian Blur on an image. image_array = GRAY image's array""" 
	kernel = get_kernel(filter_size, math.sqrt(filter_size))
	image_array = np.array(image)

	if color:
		# For color images, perform the process on the value channel of an HSV image
		height, width, _ = image_array.shape
		X_STEP, Y_STEP = kernel.shape

		resulting_array = np.zeros(image_array.shape)
		resulting_array[:,:,0] = image_array[:,:,0]
		resulting_array[:,:,1] = image_array[:,:,1]
		pad_height = int((X_STEP - 1) / 2)
		pad_width = int((Y_STEP - 1) / 2)

		padded_image = np.zeros((height + (2 * pad_height), width + (2 * pad_width)))
		padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image_array[:,:,2]

		# Perfom the convolutions
		for Xo in range(height):
			for Yo in range(width):
				Xf = Xo + X_STEP
				Yf = Yo + Y_STEP
				resulting_array[Xo, Yo, 2] = np.sum(kernel * padded_image[Xo:Xf, Yo:Yf])
		resulting_array[:,:,2] = resulting_array[:,:,2] * 255 / np.max(resulting_array[:,:,2])

		return resulting_array 
	else:
		# For B&W images
		
		height, width = image_array.shape
		X_STEP, Y_STEP = kernel.shape

		resulting_array = np.zeros(image_array.shape)

		pad_height = int((X_STEP - 1) / 2)
		pad_width = int((Y_STEP - 1) / 2)

		padded_image = np.zeros((height + (2 * pad_height), width + (2 * pad_width)))
		padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image_array

		# Perfom the convolutions
		for Xo in range(height):
			for Yo in range(width):
				Xf = Xo + X_STEP
				Yf = Yo + Y_STEP
				resulting_array[Xo, Yo] = np.sum(kernel * padded_image[Xo:Xf, Yo:Yf])
		resulting_array = resulting_array * 255 / np.max(resulting_array)
		return resulting_array

def sobel_filters(image_array):
	height, width = image_array.shape
	X_STEP = 3
	Y_STEP = 3

	Kx = np.array([-1, 0, 1,-2, 0, 2,-1, 0, 1], np.float32).reshape(3,3)
	Ky = np.array([1, 2, 1,0, 0, 0,-1, -2, -1], np.float32).reshape(3,3)
	
	resulting_array = np.zeros((height,width,2))

	pad_height = int((X_STEP - 1) / 2)
	pad_width = int((Y_STEP - 1) / 2)

	padded_image = np.zeros((height + (2 * pad_height), width + (2 * pad_width)))
	padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image_array


	for Xo in range(height):
		for Yo in range(width):
			Xf = Xo + X_STEP
			Yf = Yo + Y_STEP
			resulting_array[Xo, Yo, 0] = np.sum(Kx * padded_image[Xo:Xf, Yo:Yf]) #Horizontal
			resulting_array[Xo, Yo, 1] = np.sum(Ky * padded_image[Xo:Xf, Yo:Yf]) #Vertical
	
	G = np.hypot(resulting_array[:,:,0], resulting_array[:,:,1])
	G = G / G.max() * 255
	theta = np.arctan2(resulting_array[:,:,1], resulting_array[:,:,0])	
	return (G, theta)

def non_maxima_supression(image_array, gradient_directions):
	height, width = image_array.shape
	resulting_array = np.zeros((height,width))

	gradient_directions = gradient_directions * 180 / np.pi
	gradient_directions[gradient_directions < 0] += 180

	for X in range(1, height-1):
		for Y in range(1, width-1):
			resulting_array[X, Y] = image_array[X, Y]
			direction = gradient_directions[X,Y]

			#angle 0
			if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
				pixel_after = image_array[X, Y+1]
				pixel_before = image_array[X, Y-1]
			#angle 45
			elif (22.5 <= direction < 67.5):
				pixel_after = image_array[X+1, Y-1]
				pixel_before = image_array[X-1, Y+1]
			#angle 90
			elif (67.5 <= direction < 112.5):
				pixel_after = image_array[X+1, Y]
				pixel_before = image_array[X-1, Y]
			#angle 135
			elif (112.5 <= direction < 157.5):
				pixel_after = image_array[X-1, Y-1]
				pixel_before = image_array[X+1, Y+1]

			if (image_array[X,Y] >= pixel_after) and (image_array[X,Y] >= pixel_before):
				resulting_array[X,Y] = image_array[X,Y]
			else:
				resulting_array[X,Y] = 0

	return resulting_array



image_original = cv2.imread(os.path.join(os.path.dirname(__file__), "pic4PR2/house.jpg")).astype('uint8')
image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
image_gaussian = gaussian_blur(image_original, 5, False)
sobel_image, G = sobel_filters(np.array(image_gaussian))
# img_edge = cv2.Canny(image_original,100,200,L2gradient = True)	#https://java2blog.com/cv2-canny-python/
non_maxima_img = non_maxima_supression(sobel_image, G)

if True:
	fig = plt.figure(figsize=(15,15))
	rows = 2
	columns = 2

	fig.add_subplot(rows, columns, 1)
	plt.imshow(image_original, cmap='gray')
	plt.axis('off')
	plt.title("Original")

	fig.add_subplot(rows, columns, 2)
	plt.imshow(image_gaussian, cmap='gray')
	plt.axis('off')
	plt.title("Gaussian")

	fig.add_subplot(rows, columns, 3)
	plt.imshow(sobel_image, cmap='gray')
	plt.axis('off')
	plt.title("sobel_image")

	fig.add_subplot(rows, columns, 4)
	plt.imshow(non_maxima_img, cmap='gray')
	plt.axis('off')
	plt.title("Non Maxima")

	plt.subplots_adjust(wspace=0, hspace=0)
	plt.show()

