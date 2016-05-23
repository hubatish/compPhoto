#!/usr/bin/env python
import os
import shutil
import sys
import argparse

import numpy as np
from scipy.misc import imread, imsave
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

#########################################
###########    Skeleton    ##############
#########################################	
	
def barycentric(tri,points):
    """Compute the barycentric coordinates for a query point within the given
    triangle points."""
	
    #From: http://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
    
    simplex = tri.find_simplex(points)
    X = tri.transform[simplex,:2]
    Y = points - tri.transform[simplex,2]
    #could theoretically run this faster with einsum        #b = np.einsum('ijk,ik->ij',X,Y)
    b = np.array([x.dot(y) for x, y in zip(X, Y)]) #nowork#b = X.dot(Y[,:])
    #b = tri.transform[tetrahedra,:2].dot(tri.points - tri.transform[tetrahedra,2])
    bcoords = np.c_[b,1-b.sum(axis=1)]
    return bcoords

    """#Debuggin goodness
    print "simplex",simplex.shape
    #print "transform",tri.transform
    print "transfrom shape",tri.transform.shape
    print "points",points.shape
    print "X",X.shape
    print "Y",Y.shape
    print b
    print "got bcoords!", bcoords, "shape", bcoords.shape
   #"""
	
def bilinear_interp(image, destPoints,points):
    """Perform bilinearly-interpolated color lookup in a given image at a given
    point."""
    # Compute the four integer corner coordinates for interpolation.

    # Compute the fractional part of the point coordinates.

    # Interpolate between the top two pixels.

    # Interpolate between the bottom two pixels.

    # Return the result of the final interpolation between top and bottom.

    newImage = np.zeros_like(image)
    
    for p in xrange(0,len(points)):
        pX = points[p,0]
        rX = np.round(pX).astype(int)
        uX = np.ceil(pX).astype(int)
        lX = uX-1
        offX = pX - lX
        
        pY = points[p,1]
        rY = np.round(pY).astype(int)
        uY = np.ceil(pY).astype(int)
        lY = uY-1
        offY = pY - lY
        
        #print "source ps[",pX,",",pY,"] destPs [",destPoints[p,0],",",destPoints[p,1],"]"
        
        if(uX>=len(image)):
            uX = len(image)-1
            lX = uX -1
        if(uY>=len(image[0])):
            uY = len(image[0])-1
            lY = uY-1
        
        newImage[destPoints[p,0],destPoints[p,1]] = (
                          image[uX,uY]*offX*offY + 
                          image[uX,lY]*offX*(1.0-offY) + 
                          image[lX,uY]*(1.0-offX)*offY + 
                          image[lX,lY]*(1.0-offX)*(1.0-offY))
    return newImage

    #cool low for loop way
    topRPoints = np.ceil(points).astype(int)
    for i in xrange(0,len(points)):
        #check for highest value
        if(topRPoints[i][0]>=len(image)):
            topRPoints[i][0]-=1
        if(topRPoints[i][1]>=len(image)):
            topRPoints[i][1]-=1
    
    topLPoints = topRPoints - [1,0]
    bRPoints = topRPoints - [0,1]
    bLPoints = topRPoints -[1,1]
    offsets = points - bLPoints.astype(float)
    reshapeGoal = (image.shape[0],image.shape[1],1)
    xOffsets = (points[:,0]-bLPoints[:,0].astype(float)).reshape(reshapeGoal)
    yOffsets = (points[:,1]-bLPoints[:,1].astype(float)).reshape(reshapeGoal)
    
    print "topRPoints",topRPoints.shape
    print "offsets",offsets.shape
    print "image",image.shape
    print "xOffsets",xOffsets.shape
    #offsets = offsets.reshape(image.shape)
    
    imageTL = imageTR = imageBL = imageBR = image#np.zeros_like(image.shape)    
    imageTR = image#*offsets
    imageTR *= xOffsets #* yOffsets
    imageTL *= (1-xOffsets)*yOffsets#image*(([1.0,0]-offsets)*[1.0,-1.0])
    imageBL *= (1-xOffsets)*(1-yOffsets)#image*([1.0,1.0]-offsets)
    imageBR *= (1-yOffsets)*xOffsets#image*(([0.0,1.0]-offsets)*[-1.0,1.0])
    
    newImage = imageTR+imageTL+imageBL+imageBR
    return imageTL#newImage
    
    print "points", points
    print "topRPoints",topRPoints
    print "xoffsets",offsets
    
    import sys
    sys.exit()

def tocartesian(baryPoints,origPoints,sourceTri,destTri):
    #Convert from barycentric back to source
    simplexIs = destTri.find_simplex(origPoints) #indices of which triangles points lie in
    #print "simplexIs",simplexIs, "a simplex",source_tri.simplices[0]
    #This may not necessarily line up!#simplices = sourceTri.points[sourceTri.simplices[simplexIs]]
    simplices = sourceTri.points[destTri.simplices[simplexIs]]
    
    #print "simplices sh",simplices
    #go from [[1,2,3] multiply each index of tri with bary coors
    warpedPoints = simplices*baryPoints[:,:,None]
    #print "warpedPoints after multiplying shape ",warpedPoints.shape," points:\n",warpedPoints
    warpedPoints = warpedPoints.sum(axis=1)
    #print "warpedPoints after summing, shape ",warpedPoints.shape, " points:\n",warpedPoints
    return warpedPoints    

def warp(source, source_tri, dest_triangulation):
    """Warp the source image so that its correspondences match the destination
    triangulation."""
    result = np.zeros_like(source)#np.zeros((3,3,3))#

    # Fill in the pixels of the result image.

    # NOTE: This can be done much more efficiently in Python using a series of
    # numpy array operations as opposed to a for loop.
    
    #print "source shape",source.shape
    #Find Points to find coordinates for
    Xs,Ys = np.meshgrid(xrange(0,len(result)),xrange(0,len(result[0])))
    points = np.c_[Xs.flatten(),Ys.flatten()]#np.asarray(zip(Xs.flatten(),Ys.flatten()))
    #print "points",points.shape
    
    #Get barycentric coordinates
    bcoords = barycentric(dest_triangulation,points)
    #print "bcoords shape",bcoords

    warpedPoints = tocartesian(bcoords,points,source_tri,dest_triangulation)

    warpedImage = bilinear_interp(source,points,warpedPoints)

    return warpedImage

def morph(img1, img2, tri1, tri2, fraction):
    """Computes the intermediate morph of the given fraciton between img1
    and img2 using their correspondences."""

    # Compute the intermediate points between the points of the first and
    # second triangulations according to the warp fraction.
    intermediate_pts = tri1.points*(1.0-fraction) + tri2.points*fraction #tri2.points#

    # Compute the triangulation for the intermediate points.
    intermediate_tri = Delaunay(intermediate_pts)
    #plot_triangles(intermediate_triang)

    # Warp the first image to the intermediate triangulation.
    warp1 = warp(img1,tri1,intermediate_tri)

    # Warp the second image to the intermediate triangulation.
    warp2 = warp(img2,tri2,intermediate_tri)

    # Blend the two warped images according to the warp fraction.
    result = warp1*(1.0-fraction)+warp2*fraction

    return result#warp2#

#########################################
#####    Utility Code      ##############
#########################################

def plot_triangles(tri):
    """Sample code from http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.Delaunay.html"""
    plt.triplot(tri.points[:,0], tri.points[:,1], tri.simplices.copy())
    plt.plot(tri.points[:,0], tri.points[:,1], 'o')
    plt.show()    

def morph_sequence(start_img, end_img, corrs, n_frames):
    """Computes the n_frames long sequence of intermediate images for the warp
    between start_img and end_img, using the given correspondences in corrs."""

    start_pts, end_pts = corrs

    start_tri = Delaunay(start_pts)
    end_tri = Delaunay(end_pts)
    #plot_triangles(start_tri)

    morph_frames = []
    for frame in range(1, n_frames-1):#range(1, n_frames-1):
        print("Computing intermediate frame %d..." % frame)
        progress = frame/(n_frames-1.)#0.5#

        intermediate_frame = morph(
                start_img, end_img,
                start_tri, end_tri,
                progress)
        #print "image shape",np.asarray(intermediate_frame).shape
        #print "orig shape",np.asarray(start_img).shape
        imsave("frames/%d.png" % frame, intermediate_frame)#start_img)#
        morph_frames.append( intermediate_frame)

    return [start_img] + morph_frames + [end_img]

def read_corrs(fn):
    pts1 = []
    pts2 = []
    try:
        for line in open(fn).readlines():
            x1, y1, x2, y2 = [int(i) for i in line.strip().split()]
            pts1.append((x1, y1))
            pts2.append((x2, y2))
    except:
        return None

    return (np.array(pts1), np.array(pts2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = "CS435 Project 3: Morph one image into another.",
            formatter_class = argparse.ArgumentDefaultsHelpFormatter
            )

    parser.add_argument("start_fn", metavar="START_IMG", type=str, help="Filename for the start image.")
    parser.add_argument("end_fn", metavar="END_IMG", type=str, help="Filename for the end image.")
    parser.add_argument("corrs_file", metavar="CORRS_FILE", type=str, help="Text file with corresponding points. Four integers per line separated by spaces: X1 Y1 X2 Y2.")
    parser.add_argument("--n_frames", metavar="N_FRAMES", type=int, default=10, required=False, help="Number of intermediate frames to compute in the morph.")
    parser.add_argument("--outfile", metavar="OUTPUT_FILE", type=str, default="morph.mp4", required=False, help="Name of the file to save the morph movie as. Must end in .mp4")

    args = parser.parse_args()

    if not args.outfile.endswith(".mp4"):
        print("Output filename must end with mp4.")
        exit(1)

    try:
        start_img = imread(args.start_fn)
        start_img = np.dstack(3*[start_img]) if start_img.ndim == 2 else start_img[:, :, :3]
    except:
        print("Error reading start image.")
        exit(1)

    try:
        end_img = imread(args.end_fn)
        end_img = np.dstack(3*[end_img]) if end_img.ndim == 2 else end_img[:, :, :3]
    except:
        print("Error reading end image.")
        exit(1)

    if start_img.shape != end_img.shape:
        print("Start and end images must be the same shape.")
        exit(1)

    corrs = read_corrs(args.corrs_file)
    if corrs is None:
        print("Error reading correspondences file.")
        exit(1)

    # Swap X/Y for Numpy Y/X (Row/Col)
    corrs = [c[:, ::-1] for c in corrs]

    # Add in the 4 corner points of the image to each correspondence set (for
    # triangulation purposes).
    nr, nc = start_img.shape[:2]
    nr -= 1; nc -= 1
    corrs[0] = np.vstack([[[0,0],[nr,0],[0,nc],[nr,nc]], corrs[0]])
    corrs[1] = np.vstack([[[0,0],[nr,0],[0,nc],[nr,nc]], corrs[1]])

    start_img = start_img.astype(np.float32) / start_img.max()
    end_img = end_img.astype(np.float32) / end_img.max()

    if not os.path.exists("frames"):
        os.mkdir("frames")

    print "start morphing"
    morph_img_sequence = morph_sequence(start_img, end_img, corrs, args.n_frames)

    try:
        converter = "avconv" if (os.system("which avconv") == 0) else "ffmpeg"
        codec = "h264" if (os.system("%s -codecs | grep EV | grep h264" % converter) == 0) else "mpeg4 -b 1024k"
        os.system("%s -framerate 15 -i frames/%%d.png -c:v %s -r 30 -pix_fmt yuv420p %s" % (converter, codec, args.outfile))
    except:
        print("Error converting frames to movie.")
        exit(1)
