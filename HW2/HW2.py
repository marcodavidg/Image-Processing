import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import os


def laplacian(image, filter_config, filter_size=3):
    """Method to apply a sharpening Laplacian filter to the image."""
    image_array = np.array(image)
    resulting_array = np.zeros(image_array.shape)
    image_array = np.pad(image_array, 1, mode='constant')
    X_STEP = filter_size
    Y_STEP = filter_size
    height, width = resulting_array.shape

    # The filter config contains the matrix definition for the filter
    laplacian_filter = -1 * np.array(filter_config).reshape((filter_size, filter_size))
    for Xo in range(height):
        for Yo in range(width):
            Xf = Xo + X_STEP
            Yf = Yo + Y_STEP
            if Yf > width:
                continue
            if Xf > height:
                continue
            region = image_array[Xo:Xf, Yo:Yf]

            # Apply the filter
            resulting_array[Xo, Yo] = np.sum(np.multiply(region, laplacian_filter))

    sharpened_image = (image_array[1:-1, 1:-1] + resulting_array)

    # Return the sharpenend image and the laplacian mask used
    return sharpened_image


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
        resulting_array[:, :, 0] = image_array[:, :, 0]
        resulting_array[:, :, 1] = image_array[:, :, 1]
        pad_height = int((X_STEP - 1) / 2)
        pad_width = int((Y_STEP - 1) / 2)

        padded_image = np.zeros((height + (2 * pad_height), width + (2 * pad_width)))
        padded_image[pad_height:padded_image.shape[0] - pad_height,
        pad_width:padded_image.shape[1] - pad_width] = image_array[:, :, 2]

        # Perfom the convolutions
        for Xo in range(height):
            for Yo in range(width):
                Xf = Xo + X_STEP
                Yf = Yo + Y_STEP
                resulting_array[Xo, Yo, 2] = np.sum(kernel * padded_image[Xo:Xf, Yo:Yf])
        resulting_array[:, :, 2] = resulting_array[:, :, 2] * 255 / np.max(resulting_array[:, :, 2])

        return resulting_array
    else:
        # For B&W images

        height, width = image_array.shape
        X_STEP, Y_STEP = kernel.shape

        resulting_array = np.zeros(image_array.shape)

        pad_height = int((X_STEP - 1) / 2)
        pad_width = int((Y_STEP - 1) / 2)

        padded_image = np.zeros((height + (2 * pad_height), width + (2 * pad_width)))
        padded_image[pad_height:padded_image.shape[0] - pad_height,
        pad_width:padded_image.shape[1] - pad_width] = image_array

        # Perfom the convolutions
        for Xo in range(height):
            for Yo in range(width):
                Xf = Xo + X_STEP
                Yf = Yo + Y_STEP
                resulting_array[Xo, Yo] = np.sum(kernel * padded_image[Xo:Xf, Yo:Yf])
        resulting_array = resulting_array * 255 / np.max(resulting_array)
        return resulting_array


def sobel_filters(image_array):
	"""Apply horizontal and vertical Sobel filters on the image array provided."""
    height, width = image_array.shape
    X_STEP = 3
    Y_STEP = 3

    # Definition of the filters
    Kx = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1], np.float32).reshape(3, 3)
    Ky = np.array([1, 2, 1, 0, 0, 0, -1, -2, -1], np.float32).reshape(3, 3)

    # The resulting image's array
    resulting_array = np.zeros((height, width, 2))

    # Pad the image with zeros
    pad_height = int((X_STEP - 1) / 2)
    pad_width = int((Y_STEP - 1) / 2)

    padded_image = np.zeros((height + (2 * pad_height), width + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height,
    pad_width:padded_image.shape[1] - pad_width] = image_array

    # Apply filters
    for Xo in range(height):
        for Yo in range(width):
            Xf = Xo + X_STEP
            Yf = Yo + Y_STEP
            resulting_array[Xo, Yo, 0] = np.sum(Kx * padded_image[Xo:Xf, Yo:Yf])  # Horizontal
            resulting_array[Xo, Yo, 1] = np.sum(Ky * padded_image[Xo:Xf, Yo:Yf])  # Vertical

    # Get the gradient directions and magnitude
    gradient_directions = np.hypot(resulting_array[:, :, 0], resulting_array[:, :, 1])
    gradient_directions = gradient_directions / gradient_directions.max() * 255
    theta = np.arctan2(resulting_array[:, :, 1], resulting_array[:, :, 0])
    return (gradient_directions, theta)

def gravitational_filters(image_array):
	"""Apply the gravitational filters of the improvede Canny algorithm"""
    height, width = image_array.shape
    X_STEP = 3
    Y_STEP = 3

    # Gravitational intensity operators
    Kx = np.array([-(2**(1/2))/4, 0, (2**(1/2))/4, -1, 0, 1, -(2**(1/2))/4, 0, (2**(1/2))/4], np.float32).reshape(3, 3)
    Ky = np.array([(2**(1/2))/4, 1, (2**(1/2))/4, 0, 0, 0, -(2**(1/2))/4, -1, -(2**(1/2))/4], np.float32).reshape(3, 3)

    resulting_array = np.zeros((height, width, 2))

    # Pad the image
    pad_height = int((X_STEP - 1) / 2)
    pad_width = int((Y_STEP - 1) / 2)

    padded_image = np.zeros((height + (2 * pad_height), width + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height,
    pad_width:padded_image.shape[1] - pad_width] = image_array

    for Xo in range(height):
        for Yo in range(width):
            Xf = Xo + X_STEP
            Yf = Yo + Y_STEP
            resulting_array[Xo, Yo, 0] = np.sum(Kx * padded_image[Xo:Xf, Yo:Yf])  # Horizontal
            resulting_array[Xo, Yo, 1] = np.sum(Ky * padded_image[Xo:Xf, Yo:Yf])  # Vertical

    # Get gradient directions and magnitude
    gradient_directions = np.hypot(resulting_array[:, :, 0], resulting_array[:, :, 1])
    gradient_directions = gradient_directions / gradient_directions.max() * 255
    theta = np.arctan2(resulting_array[:, :, 1], resulting_array[:, :, 0])
    return (gradient_directions, theta)


def non_maxima_supression(image_array, gradient_directions):
    # Apply the non-maxima suprresion on the image's array with the help from the gradient directions
    height, width = image_array.shape
    resulting_array = np.zeros((height, width))

    # Convert the directions to degrees, and flip negative values
    gradient_directions = gradient_directions * 180 / np.pi
    gradient_directions[gradient_directions < 0] += 180

    for X in range(1, height - 1):
        for Y in range(1, width - 1):
            resulting_array[X, Y] = image_array[X, Y]
            direction = gradient_directions[X, Y]
            #Compare intensities and keep only the strongest pixels

            # angle  0
            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                pixel_after = image_array[X, Y + 1]
                pixel_before = image_array[X, Y - 1]
            # angle 45
            elif (22.5 <= direction < 67.5):
                pixel_after = image_array[X + 1, Y - 1]
                pixel_before = image_array[X - 1, Y + 1]
            # angle 90
            elif (67.5 <= direction < 112.5):
                pixel_after = image_array[X + 1, Y]
                pixel_before = image_array[X - 1, Y]
            # angle 135
            elif (112.5 <= direction < 157.5):
                pixel_after = image_array[X - 1, Y - 1]
                pixel_before = image_array[X + 1, Y + 1]

            if (image_array[X, Y] >= pixel_after) and (image_array[X, Y] >= pixel_before):
                resulting_array[X, Y] = image_array[X, Y]
            else:
                resulting_array[X, Y] = 0

    return resulting_array


def threshold(img, ratio = True, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    # Perform the double threhsolding
    if ratio: # The parameters' values define a percentage of detail to keep
        highThreshold = img.max() * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio
    else: # The parameters' values give absolute intensity values
        highThreshold = highThresholdRatio
        lowThreshold = lowThresholdRatio
    result = np.zeros(img.shape)

    # Define strong and weak pixels
    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    weak_value = 25
    strong_value = 255

    # Change the intensity of the strong and weak pixels identified
    result[strong_i, strong_j] = strong_value
    result[weak_i, weak_j] = weak_value

    return (result, weak_value, strong_value)

def hysteresis(image, weak=25, strong=255):
	# Transform weak pixels into strong pixels or discard them
    img = image.copy()
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def using_improved(image_original, k = 1.3, plots = True):
	# Process the image with the improved version of the Canny edge detector

	# Gaussian smoothing filter
    image_gaussian = gaussian_blur(image_original, 5, False) 
    
    # Gravitational filters
    gravitational_image, gradient_directions = gravitational_filters(np.array(image_gaussian))
    
    # Non-maxima suppression
    non_maxima_img = non_maxima_supression(gravitational_image, gradient_directions)

    width, height = image_original.shape

    # Get Eave and calculate sigma as defined by the improved Canny paper
    Eave = gravitational_image.sum()/(width*height)
    sigma = 0
    for i in range(width):
        for j in range(height):
            sigma += (gravitational_image[i,j]-Eave)**2
    sigma = (sigma / (width*height))**(1/2)

    # Define the two thresholds for the image
    highThresholdRatio = Eave + k * sigma
    lowThresholdRatio = highThresholdRatio / 2

    # Get the result
    double_threshold_img, weak_value, strong_value = threshold(non_maxima_img, ratio=False, lowThresholdRatio=lowThresholdRatio, highThresholdRatio=highThresholdRatio)
    resulting_img = hysteresis(double_threshold_img, weak_value, strong_value)

    # Get the plots of the process
    if plots:
        fig = plt.figure(figsize=(15, 15))
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
        plt.imshow(gravitational_image, cmap='gray')
        plt.axis('off')
        plt.title("gravitational_image")

        fig.add_subplot(rows, columns, 4)
        plt.imshow(non_maxima_img, cmap='gray')
        plt.axis('off')
        plt.title("non_maxima_img")

        plt.show()

        fig = plt.figure(figsize=(15, 15))
        rows = 1
        columns = 1

        fig.add_subplot(rows, columns, 1)
        plt.imshow(resulting_img, cmap='gray')
        plt.axis('off')
        plt.title("Non Maxima")

        plt.show()

    return resulting_img

def using_own(image_original, plots=False):
	# Process the image with the original base Canny edge detector algorithm

	# Gaussian smoothing
    image_gaussian = gaussian_blur(image_original, 5, False)
    
    # Sobel filters
    sobel_image, gradient_directions = sobel_filters(np.array(image_gaussian))
    
    # Non-maxima suppression
    non_maxima_img = non_maxima_supression(sobel_image, gradient_directions)
    
    # Double thresholding
    double_threshold_img, weak_value, strong_value = threshold(non_maxima_img)
    
    # Hysteresis process
    resulting_img = hysteresis(double_threshold_img, weak_value, strong_value)

    # Get process' plots
    if plots:
        fig = plt.figure(figsize=(15, 15))
        rows = 2
        columns = 2

        fig.add_subplot(rows, columns, 1)
        plt.imshow(image_original, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.title("B&W Image")

        fig.add_subplot(rows, columns, 2)
        plt.imshow(image_gaussian, cmap='gray')
        plt.axis('off')
        plt.title("After Gaussian Smoothing filter")

        fig.add_subplot(rows, columns, 3)
        plt.imshow(sobel_image, cmap='gray')
        plt.axis('off')
        plt.title("After Sobel filters")

        fig.add_subplot(rows, columns, 4)
        plt.imshow(non_maxima_img, cmap='gray')
        plt.axis('off')
        plt.title("Non-Maxima Suppression")

        # plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

        fig = plt.figure(figsize=(15, 15))
        rows = 1
        columns = 2

        fig.add_subplot(rows, columns, 1)
        plt.imshow(double_threshold_img, cmap='gray')
        plt.axis('off')
        plt.title("Double-Thresholding")

        fig.add_subplot(rows, columns, 2)
        plt.imshow(resulting_img, cmap='gray')
        plt.axis('off')
        plt.title("Edge tracking by Hysteresis")
        
        plt.show()

    return resulting_img


def using_cv2(image_original, plots=False):
    #Proces the image with the open-source libray OpenCV2

    edges = cv2.Canny(image_original, 110, 210, False)

    if plots:
        plt.subplot(121), plt.imshow(image_original, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    return edges


def compare(figure1, figure2, text1, text2):
    # Plot two figures side by side
    fig = plt.figure(figsize=(15, 15))
    rows = 1
    columns = 2

    fig.add_subplot(rows, columns, 1)
    plt.imshow(figure1, cmap='gray')
    plt.axis('off')
    plt.title(text1)

    fig.add_subplot(rows, columns, 2)
    plt.imshow(figure2, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title(text2)

    plt.show()


# All the images to be tested, with their corresponding k value for the improved algorithm
images = np.array(["pic4PR2/brain_mri.jpg",1.6,
          "pic4PR2/basketball.jpg",0.7,
          "pic4PR2/balls.jpg",0.9,
          "pic4PR2/house.jpg",1.1,
          "pic4PR2/pets.jpg",1.3,
          "pic4PR2/toy_story.jpg",0.7]).reshape(6,2)

# Process all the images from the array
for image, k in images:
    image_originalColor = cv2.imread(os.path.join(os.path.dirname(__file__), image)).astype('uint8')
    
    # Get B&W image
    image_original = cv2.cvtColor(image_originalColor, cv2.COLOR_BGR2GRAY)
    
    # Laplacian sharpening filter
    # sharpened_image = laplacian(image_original, [0, 1, 0, 1, -4, 1, 0, 1, 0])


    # Get different versions of the Canny algorithm
    own_canny = using_own(image_original, False)

    # sharpened_canny = using_own(sharpened_image, False)
    improved_canny = using_improved(image_original, k=float(k), plots=False)
    cv2_canny = using_cv2(image_original, False)

    # Plot results
    fig = plt.figure(figsize=(15, 15))
    rows = 2
    columns = 2

    fig.add_subplot(rows, columns, 1)
    plt.imshow(cv2.cvtColor(image_originalColor, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Original Image")

    fig.add_subplot(rows, columns, 2)
    plt.imshow(own_canny, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title("Canny algorithm")

    fig.add_subplot(rows, columns, 3)
    plt.imshow(improved_canny, cmap='gray')
    plt.axis('off')
    plt.title("Improved Canny algorithm")

    fig.add_subplot(rows, columns, 4)
    plt.imshow(cv2_canny, cmap='gray')
    plt.axis('off')
    plt.title("OpenCV Implementation")

    plt.show()
