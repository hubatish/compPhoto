"""combine_exposures -- Combine exposures into a single HDR image.

Output an HDR image by computing the inverse response function for each input image,
using the inverse repsonse function to compute the irradiance of each image,
and perform a weighted average to merge them.
"""

import argparse
import os
import sys

import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg

import Imath
import OpenEXR


# constants for pixel values
ncolors = 256
zmin = 0
zmax = ncolors - 1
zmid = (zmax + zmin) / 2

# compute weight array
weights = np.array([z - zmin if z <= zmid else zmax - z for z in xrange(ncolors)])


def compute_response(imagelist, exposuretimes, channel, npixels, smoothweight):
    """Compute and return inverse response function and irradiance values
    
    Uses the method of Debevec and Malik, "Recovering High Dynamic Range Radiance Maps from Photographs"
    """

    # check parameters
    assert len(imagelist) == len(exposuretimes)
    nimages = len(imagelist)

    # initialize matrix and array
    A = scipy.sparse.lil_matrix((nimages * npixels + ncolors - 1, ncolors + npixels))
    b = np.zeros((nimages * npixels + ncolors - 1,))

    # construct A and b

    # add data term
    # ENTER CODE HERE
    # ENTER CODE HERE
    # ENTER CODE HERE

    # add smoothness constraint
    # ENTER CODE HERE
    # ENTER CODE HERE
    # ENTER CODE HERE

    # add constraint g(z_mid) = 0
    # ENTER CODE HERE
    # ENTER CODE HERE
    # ENTER CODE HERE

    # solve least square system
    result = scipy.sparse.linalg.lsqr(A.tocsr(), b)[0]

    return result[:ncolors], result[ncolors:]


def combine_exposures(imagelist, exposuretimes, npixels, smoothweight):
    """Combine a series of expsures into a single HDR photo"""

    # check parameters
    assert len(imagelist) == len(exposuretimes)
    nimages = len(imagelist)

    final = np.zeros((imagelist[0].shape), dtype=np.float32)
    weightsum = np.zeros((imagelist[0].shape), dtype=np.float32)

    # do each color channel separately
    for c in xrange(3):
        # compute inverse response function
        responsefunc, radiance = compute_response(imagelist, exposuretimes, c, npixels, smoothweight)

        # optionally plot the response function (helpful for debugging)
        #import matplotlib.pyplot as plt
        #plt.plot(np.arange(len(responsefunc)), responsefunc)
        #plt.show()
        
        # combine exposures
        for i in xrange(nimages):
            # ENTER CODE HERE
            # ENTER CODE HERE
            # ENTER CODE HERE
            pass

    final /= weightsum
    return np.exp(final)


def writeexr(path, image):
    """Write image to path as an EXR file"""
    header = OpenEXR.Header(image.shape[1], image.shape[0])
    out = OpenEXR.OutputFile(path, header)
    out.writePixels({'R': image[:, :, 0].tostring(),
                     'G': image[:, :, 1].tostring(),
                     'B': image[:, :, 2].tostring()})


if __name__ == "__main__":
    description = '\n\n'.join(__doc__.split('\n\n'))
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('output_path', type=str)
    parser.add_argument('imagelist_path', type=str,
                        help="path to a file that lists each image and corresponding shutter speed")
    parser.add_argument('--npixels', type=int, default=4096,
                        help="Number of pixels used to compute inverse response function")
    parser.add_argument('--smoothweight', type=float, default=256.,
                        help="Weighting on smoothness of response function")
    if len(sys.argv) == 1:
        parser.print_help()

    args = parser.parse_args()

    imageinfo = np.genfromtxt(args.imagelist_path, dtype=None)

    imagelist = [scipy.ndimage.imread(os.path.join(os.path.dirname(args.imagelist_path), x)) for x in imageinfo['f0']]
    exposuretimes = imageinfo['f1']

    hdr = combine_exposures(imagelist, exposuretimes, args.npixels, args.smoothweight)
    writeexr(args.output_path, hdr)
