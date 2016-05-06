#!/usr/bin/env python
import sys
import numpy as np
import cv2


def remove_vertical_seam(image, seam):
    """
    Removes the given seam from the image.
    
    image : an n x m array (may have multiple channels)
    seam : an n x 1 array of X-coordinates defining the seam pixels in top-down order.
    
    Thus, seam[0] means remove pixel (0, seam[0]) from the input image.
    
    returns: an n x (m - 1) image with the seam removed.
    """
    # TODO: CODE ME!

def gradient_magnitude(image):
    """
    Returns the L1 gradient magnitude of the given image.
    The result is an n x m numpy array of floating point values,
    where n is the number of rows in the image and m is the number of columns.
    """
    # TODO: First, convert the input image to a 32-bit floating point grayscale.
    # Be sure to scale it such that the intensity varies between 0 and 1.
    
    
    # TODO: Next, compute the graient in the x direction using the sobel operator with a kernel size of 5
    
    # TODO: and compute the graient in the y direction using the sobel operator with a kernel size of 5
    
    
    # TODO: Finally, compute the l1 norm of the x and y gradients at each pixel value.
    # The l1 norm is the sum of their absolute values.
    # convert the final image from a double-precision floating point to single.
    
    
    # and return the result
    return energy


def compute_seam_costs(energy):
    """
    Computes the cumulative minimum energy of every possible seam in the provided energy image.
    You can do this using the dynamic programming rule:
         M(i, j) = e(i, j) + min( M(i-1, j-1), M(i-1, j), M(i-1, j+1) 
    
    energy : an n x m single channel image defining the energy at each pixel.
    returns : an n x m image containing the cumulative minimum energy cost of all seams through each pixel.
    """
    # TODO: Create M, an n x m matrix with the first row equal to energy.
    
    # TODO: Iterate over the rows, starting at row 1
    
        # TODO: Iterate over the column 1 to m - 1
        for j in range(1, m - 1):
            # TODO: Compute M(i, j) = e(i, j) + min( M(i-1, j-1), M(i-1, j), M(i-1, j+1)
            # Be sure to handle edge cases where j = 0 and j = m - 1

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

    # TODO: Compute the bottom-up path of pixel X-coordinates for the seam with
    # minimal cost.
    # Return the top-down seam X-coordinates and the total energy cost of
    # removing that seam.
    return np.asarray(path)[::-1], cost

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
        print "Removing seam {} / {} ".format(i, len(order))
        
        # TODO: check if order = 0, if so, transpose the image!
        
        # TODO: compute the energy using gradient_magnitude
        
        # TODO: Compute M using compute_seam_costs
        
        # TODO: get the minimal seam using 'minimal_seam'
        
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
    resized = resize(image, (h,w))
    cv2.imwrite(out_fn, resized)
