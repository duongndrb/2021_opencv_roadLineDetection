#Part 1:

#Import the required libraries
import matplotlib.pylab as plt
import cv2
import numpy as np

#function to mask every other thing other than our region of interest
def region_of_interest(img, vertices):
    #define the blank matrix
    mask = np.zeros_like(img)

    #channel_count = img.shape[2]

    #create the match color with the same color channel counts
    #match_mask_color = (255,) *channel_count
    #using the match color with the grayscale image
    match_mask_color = 255

    #Fill poly-function deals with multiple polygon
    cv2.fillPoly(mask, vertices,match_mask_color)
    #Bitwise operation between canny image and mask image
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#function draw the Line
# 1 param = the image
# 2 param = the lines vector
def draw_the_Lines(img, lines):
    img = np.copy(img)
    #blanked matrix
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    #the param line give 4 parameters: the coordinates of first and second point
    for line in lines:
        for x1, y1, x2, y2 in line:
            #draw the line
            cv2.line(blank_image, (x1,y1), (x2,y2), (0,255,0), thickness=3)
    #merge the blank image with original image


#addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) -> dst
#@brief Calculates the weighted sum of two arrays.

#The function addWeighted calculates the weighted sum of two arrays as follows
#  where I is a multi-dimensional index of array elements. 
# In case of multi-channel arrays, each channel is processed independently. 
# The function can be replaced with a matrix expression:

# @code{.cpp}
#     dst = src1*alpha + src2*beta + gamma;

# @param src1: first input array.
# @param alpha: weight of the first array elements.
# @param src2: second input array of the same size and channel number as src1.
# @param beta: weight of the second array elements.
# @param gamma: scalar added to each sum.
# @param dst:  output array that has the same size and number of channels as the input arrays.
# @param dtype optional depth of the output array; 
# when both input arrays have the same depth, dtype can be set to -1, which will be equivalent to src1.depth().
    img = cv2.addWeighted(img, 0.8, blank_image, 1,0.0)

    return img


#read the image
image = cv2.imread('road.png')
#convert image to the RGB channel
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

#find out the shape of image
print(image.shape)

height = image.shape[0]
width = image.shape[1]
#define vung quan tam
region_of_interest_vertices = [
    (0,height),
    (width/2, height/2),
    (width,height)
]


gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

canny_image = cv2.Canny(gray_image, 100,200)

cropped_image = region_of_interest(canny_image,
                 np.array([region_of_interest_vertices],np.int32))

#part2:
#draw the line using hough line Transform
#param rho Distance resolution of the accumulator in pixels.
#@param theta Angle resolution of the accumulator in radians.
#@param threshold Accumulator threshold parameter. 
# Only those lines are returned that get enough votes ( \f$>\texttt{threshold}\f$ ).
#@param minLineLength Minimum line length. Line segments shorter than that are rejected.
#@param maxLineGap Maximum allowed gap between points on the same line to link them.

lines = cv2.HoughLinesP(cropped_image, 
            rho=6, 
            theta=np.pi/60,
            threshold=160,
            lines=np.array([]),
            minLineLength=40,
            maxLineGap=25)

image_with_lines = draw_the_Lines(image, lines)

#plt.imshow(cropped_image)
plt.imshow(image_with_lines)
plt.show()