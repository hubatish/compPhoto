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

class ZPoint(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

#Takes the whole image, position on that image, width around that pos
#Returns array to apply on those points
def bilateral_filter(image, pos, spatialSigma):
    """Compute bilateral filter of image"""
    # see http://en.wikipedia.org/wiki/Bilateral_filter
    w = np.floor(spatialSigma/2.0) # Width to left
    
    imageW = image.shape[0]
    imageH = image.shape[1]
    print("sizeL: "+str(w))
    filter = np.zeros((spatialSigma,spatialSigma))

    #Takes numerator
    #spits out gaussian value for that point
    def oneGauss(x,y):
        gaussSigma = 1.0
        gS2 = gaussSigma*gaussSigma
        n2 = x*x+y*y
        v = 1.0/(2.0*gS2*np.pi)*np.exp(-n2/(2.0*gS2))
        return v

    xs = xrange(int(-w),int(w+1))
    ys = xrange(int(-w),int(w+1))
    print(xs)
    print(ys)


    for x in xs:
        for y in ys:
            destV = 0.0
            #if(x>0 && x<imageW)
            destV = oneGauss(x,y)
            print("got v: "+str(destV) + " at "+str(x),',',y)
            filter[x+w,y+w] = destV

    #ensure filter totals to 1
    sumFr = 0.0
    for x in xs:
        for y in ys:
            sumFr += filter[x+w,y+w]

    print("the sum fraction is: "+str(sumFr))
    
    #filter = gaussian_filter(a,rangesigma)
    
    return filter


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
    a = bilateral_tonemap(a)
    printArray(a)
    filter = bilateral_filter(a,ZPoint(2,2),3)
    print("filter")
    printArray(filter)

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

    tests(loadexr(args.input_path))
    #scipy.misc.imsave(args.output_path, args.method(loadexr(args.input_path)))


