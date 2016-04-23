"""tonemap -- Tone map an HDR image.

Tonemap an HDR input image using a global method or local method.
"""

from __future__ import print_function

import argparse
import sys

import numpy as np
import scipy.misc

import Imath
import OpenEXR


def loadexr(filename):
    """Open an exr image and return a numpy array."""
    f = OpenEXR.InputFile(filename)
    dw = f.header()['dataWindow']
    sz = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    image = np.empty((sz[0], sz[1], 3), dtype=np.float32)
    image[..., 0] = np.fromstring(f.channel("R", FLOAT), dtype=np.float32).reshape(sz)
    image[..., 1] = np.fromstring(f.channel("G", FLOAT), dtype=np.float32).reshape(sz)
    image[..., 2] = np.fromstring(f.channel("B", FLOAT), dtype=np.float32).reshape(sz)

    return image


def bilateral_filter(image, spatialsigma, rangesigma):
    """Compute bilateral filter of image"""
    # ENTER CODE HERE
    # ENTER CODE HERE
    # ENTER CODE HERE
    # see http://en.wikipedia.org/wiki/Bilateral_filter
    pass


def log_tonemap(image):
    image[image == 0] = np.min(image) / 16.
    logimage = np.log(image)
    imgmin = np.min(logimage)
    imgmax = np.max(logimage)
    return (255. * (logimage - imgmin) / (imgmax - imgmin)).astype(np.uint8)


def divide_tonemap(image):
    divimage = image / (1. + image)
    return (divimage * (255. / np.max(divimage))).astype(np.uint8)


def sqrt_tonemap(image):
    sqrtimage = np.sqrt(image)
    return (sqrtimage * (255. / np.max(sqrtimage))).astype(np.uint8)


def bilateral_tonemap(image):
    """Tonemap image (HDR) using Durand 2002"""
    tests(image)
    
    # compute intensity
    Is = np.mean(image,axis=2)

    # compute log intensity
    LIs = np.log(Is)

    # compute bilateral filter
    # ENTER CODE HERE

    # compute detail layer
    # ENTER CODE HERE

    # apply an offset and scale to the base
    # ENTER CODE HERE

    # reconstruct the log intensity
    # ENTER CODE HERE

    # put back colors
    # ENTER CODE HERE

    # gamma compress
    # ENTER CODE HERE

    # rescale to 0..255 range
    # ENTER CODE HERE

    # convert to LDR
    return image#tonemapped.astype(np.uint8)

def printArray(a):
    for p in a:
        for v in p:
            print(str(v)+",",end='')
        print()

def tests(image):
    #print("image size: "+str(image.size))
    print("image dimensions: "+str(image.shape))
    a = image[:5,:5]
    print("image dimensions: "+str(a.shape))
    #printArray(a)
    #a = [[1,50,200],[240,240,10],[60,120,120]]
    a2 = np.mean(a,axis=2)
    a3 = np.log(a2)
    printArray(a3)

if __name__ == "__main__":
    def check_method(s):
        if s == 'durand02':
            return bilateral_tonemap
        elif s == 'log':
            return log_tonemap
        elif s == 'divide':
            return divide_tonemap
        elif s == 'sqrt':
            return sqrt_tonemap
        else:
            raise argparse.ArgumentTypeError('Unknown method: '+s)

    description = '\n\n'.join(__doc__.split('\n\n'))
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('output_path', type=str)
    parser.add_argument('input_path', type=str)
    parser.add_argument('--method', type=check_method, default=bilateral_tonemap,
                        help="Must be one of, 'durand02', 'log', 'divide', or 'sqrt'")
    if len(sys.argv) == 1:
        parser.print_help()

    args = parser.parse_args()

    scipy.misc.imsave(args.output_path, args.method(loadexr(args.input_path)))
