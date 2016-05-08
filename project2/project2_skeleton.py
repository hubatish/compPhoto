#!/usr/bin/env python
import sys
import numpy as np
import cv2
from scipy import signal


def remove_vertical_seam(image, seam):
    """
    Removes the given seam from the image.
    
    image : an n x m array (may have multiple channels)
    seam : an n x 1 array of X-coordinates defining the seam pixels in top-down order.
    
    Thus, seam[0] means remove pixel (0, seam[0]) from the input image.
    
    returns: an n x (m - 1) image with the seam removed.
    """
    # TODO: CODE ME!
    pass

def applyConvolution(image,kernel):
    """
    Returns image with kernel applied to it by multiplying each number in kernel to corresponding grid location, then adding them all together
    """
    gradient = np.zeros(image.shape)
    kWidth = np.floor(len(kernel) / 2).astype(int)
    kHeight = np.floor(len(kernel[0])/2).astype(int)
    #print 'kernel: w' , kWidth, 'h', kHeight
    
    #for each pixel in image
    for i in range(kWidth,image.shape[0]-kWidth):
        for j in range(kHeight,image.shape[1]-kHeight):
            #add up each kernel square
            sum = 0.0
            for kI in range(0,len(kernel)):
                for kJ in range(0,len(kernel[0])):
                    sum += image[i-kWidth+kI][j-kHeight+kJ]*kernel[kI][kJ]
            gradient[i][j] = sum
            #print "sum for ",i,",",j,":",sum

    #TODO, handle edges

    print "finished making a gradient"
    return gradient
    
def normalize2D(image):
    """
    Make all values of [X,Y] array between 0 and 1
    """
    IMax = np.amax(image)
    IMin = np.amin(image)
    Is = image - IMin
    Is = np.divide(Is,float(IMax))
    return Is
    #print Is.shape , 'max ', IMax, ' min ', IMin

def gradient_magnitude(image):
    """
    Returns the L1 gradient magnitude of the given image.
    The result is an n x m numpy array of floating point values,
    where n is the number of rows in the image and m is the number of columns.
    """
    # First, convert the input image to a 32-bit floating point grayscale.
    # Be sure to scale it such that the intensity varies between 0 and 1.
    Is = np.mean(image,axis=2,dtype=np.float64)
    Is = normalize2D(Is)
    
    # Next, compute the graient in the x direction using the sobel operator with a kernel size of 5
    sobelX = [[2.0,1.0,0.0,-1.0,-2.0],
              [3.0,2.0,0.0,-2.0,-3.0],
              [4.0,3.0,0,-3.0,-4.0],
              [3.0,2.0,0.0,-2.0,-3.0],
              [2.0,1.0,0.0,-1.0,-2.0]]
    #gradientX = applyConvolution(Is,sobelX) #apparently there's a np.convolute2d - and it is so much faster!
    gradientX = signal.convolve2d(Is,sobelX)
    # Compute the graient in the y direction using the sobel operator with a kernel size of 5
    gradientY = signal.convolve2d(Is,np.transpose(sobelX))
    #gradientY = applyConvolution(Is,np.transpose(sobelX))    

    # Finally, compute the l1 norm of the x and y gradients at each pixel value.
    # The l1 norm is the sum of their absolute values.
    # convert the final image from a double-precision floating point to single.
    gradientX = np.absolute(gradientX)
    gradientY = np.absolute(gradientY)
    energy = gradientX + gradientY
    
    # and return the result
    return energy

bogusValue = -10 # we haven't done this part of image yet

def get_min_energy_path(energy,M,i,j):
    """
    Return the minimum energy to get to this point.
    Recursively compute this number if not available,
    otherwise return cached value
    """
    if j<0 or j>=energy.shape[1]:
        return float("inf") #out of bounds, don't use this! return infinity
    if M[i][j]!=bogusValue:
        return M[i][j] # we already solved this, return
    if i==0:
        M[i][j] = energy[i][j] #base case, reached left edge
        return M[i][j]
    
    #actually recurse
    minL = np.amin([get_min_energy_path(energy,M,i-1,j-1),
                    get_min_energy_path(energy,M,i-1,j),
                    get_min_energy_path(energy,M,i-1,j+1)])
    M[i][j] = energy[i][j]+minL
    return M[i][j]

def compute_seam_costs(energy):
    """
    Computes the cumulative minimum energy of every possible seam in the provided energy image.
    You can do this using the dynamic programming rule:
         M(i, j) = e(i, j) + min( M(i-1, j-1), M(i-1, j), M(i-1, j+1) 
    
    energy : an n x m single channel image defining the energy at each pixel.
    returns : an n x m image containing the cumulative minimum energy cost of all seams through each pixel.
    """
    # TODO: Create M, an n x m matrix with the first row equal to energy.
    M = np.zeros(energy.shape) +bogusValue
     
    # TODO: Iterate over the rows, starting at row 1
    for j in range(0,len(energy[0])):
        e = get_min_energy_path(energy,M,len(energy)-1,j)
        #print 'got energy for j ',j,":",e
        # TODO: Iterate over the column 1 to m - 1
        #for j in range(1, m - 1):
            # TODO: Compute M(i, j) = e(i, j) + min( M(i-1, j-1), M(i-1, j), M(i-1, j+1)
            # Be sure to handle edge cases where j = 0 and j = m - 1
         #   pass
    # return the result!
    return M


def minimal_seam(M):
    """
    Find the seam with minimal energy cost given the provided seam cost
    matrix M. Returns the X-coordinates of the minimal-cost vertical seam in
    top-down order.
    
    M: the output from compute_seam_costs.
    
    return: a list of x-coordinates starting at row 0 of M containing the ones to remove, and a cost of that seam.
    """
    path = []

    # minimal cost.
    #select rightmost row
    #find starting position 
    lastRow = M[:,-1]
    print 'size M', M.shape, 'w of M? ',len(M),'size c' , lastRow.shape
    cost = 0.0
    mini = float('inf')
    pos = 0
    for c in range(0,len(lastRow)):
        if lastRow[c]<mini:
            mini = lastRow[c]
            pos = c
            path += [pos]
    
    for i in range(len(M)-2,-1,-1):
        lV = float('inf')
        if(pos-1>=0):
            lV = M[i][pos-1]
        rV = float('inf')
        if(pos+1<len(M)):
            rV = M[i][pos+1]
        mV = M[i][pos]
        if(lV<rV):
            if(lV<mV):
                pos = pos-1
        elif(rV<mV):
            pos = pos + 1
        print "going to pos ", pos, "v: ",M[i][pos]
        path += [pos]
    
    print "I got a path",len(path)
    
    # TODO: Compute the bottom-up path of pixel X-coordinates for the seam with

    # Return the top-down seam X-coordinates and the total energy cost of
    # removing that seam.
    return np.asarray(path)[::-1]

def compute_ordering(image, target_size):
    """
    Compute the optimal order of horizontal and vertical seam removals to
    achieve the given target image size. Order should be returned as a list of
    0 or 1 values corresponding to horizontal and vertical seams
    respectively.
    """
    r = image.shape[0] - target_size[0] + 1
    c = image.shape[1] - target_size[1] + 1
    if r < 0 or c < 0:
        raise ValueError("Target size must be smaller than the input size.")
    return [0,1] * min(r-1, c-1) + [0] * max(r-c, 0) + [1] * max(c-r, 0)

def resize(image, target_size):
    output = image.copy()
    order = compute_ordering(output, target_size)

    for i, seam_type in enumerate(order):
        #print "Removing seam {} / {} ".format(i, len(order))

        # TODO: check if order = 0, if so, transpose the image!
        
        # TODO: compute the energy using gradient_magnitude
        energy = gradient_magnitude(image)
        
        # TODO: Compute M using compute_seam_costs
        M = compute_seam_costs(energy)
        # TODO: get the minimal seam using 'minimal_seam'
        seam = minimal_seam(M)
        # TODO: remove it using 'remove_vertical_seam'
        
        # TODO: check if order = 0, if so, transpose the image back!

    # Sanity check.....
    assert(output.shape[0] == target_size[0] and
           output.shape[1] == target_size[1])
           
    # return results...
    return output


if __name__ == "__main__":
    try:
        in_fn, h, w, out_fn = sys.argv[1:]
        h, w = int(h), int(w)
    except:
        print("Usage: python p2.py FILE TARGET_HEIGHT TARGET_WIDTH OUTPUT")
        exit(1)

    image = cv2.imread(in_fn)
    #resized = normalize2D(gradient_magnitude(image))*255.0
    resized = resize(image, (h,w))
    cv2.imwrite(out_fn, resized)
