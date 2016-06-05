#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tour into Picture.

This program has two states.

In the first state, the user selects the corners of the back wall of the image
and the vanishing point.

In the second state, the user can navigate the constructed 3d volume. Drag the
left mouse button to look around, drag the right mouse button to zoom, and use
WASD keys to move.

Use the spacebar to switch between states.
"""
import argparse
import sys

import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image


def getY(x, p1, p2):
    return (p2[1] - p1[1]) * (x - p1[0]) / (p2[0] - p1[0]) + p1[1]


def getX(y, p1, p2):
    return (p2[0] - p1[0]) * (y - p1[1]) / (p2[1] - p1[1]) + p1[0]


###############################################################################
#                        Fill in the code below
###############################################################################

def computefaces(corner1, corner2, vanishingpt, imwidth, imheight, focallen):
    """compute the 2d and 3d faces of the box given the back wall corners and
    vanishing point.
    corner1 --- ndarray(shape=(2,), dtype=float), image location of corner 1
    corner2 --- ndarray(shape=(2,), dtype=float), image location of corner 2
    vanishingpt --- ndarray(shape=(2,), dtype=float), image location of vanishing point
    imwidth --- image width
    imheight --- image height
    focallen --- focal length of camera
    
    returns:
    back3d ---   ndarray(shape=(4,3), dtype=float) plane defining the back of the box in 3d.
    right3d ---  ndarray(shape=(4,3), dtype=float) plane defining the right side of the box in 3d.
    top3d ---    ndarray(shape=(4,3), dtype=float) plane defining the top of the box in 3d.
    left3d ---   ndarray(shape=(4,3), dtype=float) plane defining the left side of the box in 3d.
    bottom3d --- ndarray(shape=(4,3), dtype=float) plane defining the bottom of the box in 3d.
    back2d ---   ndarray(shape=(4,2), dtype=float) plane defining the back of the box in 2d.
    right2d ---  ndarray(shape=(4,2), dtype=float) plane defining the right side of the box in 2d.
    top2d ---    ndarray(shape=(4,2), dtype=float) plane defining the top of the box in 2d.
    left2d ---   ndarray(shape=(4,2), dtype=float) plane defining the left side of the box in 2d.
    bottom2d --- ndarray(shape=(4,2), dtype=float) plane defining the bottom of the box in 2d.
    """

    bl = backleft = min(corner1[0], corner2[0])
    br = backright = max(corner1[0], corner2[0])
    bb = backbottom = max(corner1[1], corner2[1])
    bt = backtop = min(corner1[1], corner2[1])
    
    vpx = vanishingpt[0]
    vpy = vanishingpt[1]
    cam_height = imheight - vpy #camera height, from bottom of image #0 is at top though
    a = bb - vpy #height of camera from focal/image plane
    d = focallen*cam_height/a - focallen #distance to actual image from sensor

    """Hint: A straightfoward implementation to get the 2D corners of each face would be
    like the following. Does this properly take care of the corners lying outside the 
    image domain? If not, how do you take care of them. The same applies to 
    the 3D coordinates, too.
    """
    back2d = np.array([[backright, backtop],
                       [backleft, backtop],
                       [backleft, backbottom],
                       [backright, backbottom]])

    getYRT = getY(imwidth, [backright, backtop], vanishingpt)
    getYRB = getY(imwidth, [backright, backbottom], vanishingpt)
    right2d = np.array([[imwidth, getYRT],
                        [backright, backtop],
                        [backright, backbottom],
                        [imwidth, getYRB]])

    top2d = np.array([[getX(0, [backright, backtop], vanishingpt), 0.],
                      [getX(0, [backleft, backtop], vanishingpt), 0.],
                      [backleft, backtop],
                      [backright, backtop]])


    left2d = np.array([[backleft, backtop],
                       [0., getY(0, [backleft, backtop], vanishingpt)],
                       [0., getY(0, [backleft, backbottom], vanishingpt)],
                       [backleft, backbottom]])

    bottom2d = np.array([[backright, backbottom],
                         [backleft, backbottom],
                         [getX(imheight, [backleft, backbottom], vanishingpt), imheight],
                         [getX(imheight, [backright, backbottom], vanishingpt), imheight]])

    #using straight corner for now but...
    #TODO: Actually plot out that line & see if it doesn't quite match up... maybe????    
    back3d = np.array([[imwidth,imheight,d],
                      [0,imheight,d],
                      [0,0,d],
                      [imwidth,0,d]])

    right3d = np.array([[imwidth,getYRB,d],#maybe getY stuff
                       [imwidth,imheight,0],
                       [imwidth,0,0],
                       [imwidth,getYRT,d]])

    top3d = np.array([[getX(0, [backright, backtop], vanishingpt), imheight,d],
                      [getX(0, [backleft, backtop], vanishingpt), imheight,d],
                      [0, imheight,0],
                      [imwidth,imheight,0]])
                       
    left3d = np.array([[0, imheight,0],
                       [0., getY(0, [backleft, backbottom], vanishingpt),d],
                       [0., getY(0, [backleft, backtop], vanishingpt),d],
                       [0, 0,0]])    

    bottom3d = np.array([[imwidth, 0,0],
                         [0, 0,0],
                         [getX(imheight, [backleft, backbottom], vanishingpt), 0,d],
                         [getX(imheight, [backright, backbottom], vanishingpt), 0,d]])    

    return back3d, right3d, top3d, left3d, bottom3d, back2d, right2d, top2d, left2d, bottom2d


def transformimage(image, face2d, face3d):
    """transform the image region face2d to face3d using a homography.
    image --- ndarray(shape=(imheight, imwidth, 3), dtype=uint8)
    face2d --- ndarray(shape=(4, 2), dtype=float)
    face3d --- ndarray(shape=(4, 3), dtype=float)"""
    imwidth = int(np.linalg.norm(face3d[1] - face3d[0]))
    imheight = int(np.linalg.norm(face3d[2] - face3d[1]))

    target = np.array([[imwidth, 0.],
                       [0., 0.],
                       [0., imheight],
                       [imwidth, imheight]])
    homography = computehomography(face2d, target)

    return homographywarp(image, homography, imwidth, imheight)


def computehomography(facefrom, faceto):
    """compute homography between two quadrilaterals.
    facefrom --- ndarray(shape=(4, 2), dtype=float)
    faceto --- ndarray(shape=(4, 2), dtype=float)"""
    A = np.zeros((8, 9))
    #A = np.ndarray([[]])
    
    # TO solve the homography, you'll need to use least squares.
    # Set the values in A such that Ax = b will solve the homography...    

    #see page 81 of Week8
    #[x1,y1,1,0 ,0 ,0,-x'*x,-x'*y,-x']
    #[0 ,0 ,0,x1,y1,1,-y'*x,-y'*y,-y']
    for row in xrange(0,len(A),2):
        x = facefrom[row/2][0]
        y = facefrom[row/2][1]
        xp = faceto[row/2][0]
        yp = faceto[row/2][1]
        A[row,0] = x
        A[row,1] = y
        A[row,2] = 1
        A[row,6] = -xp*x
        A[row,7] = -xp*y
        A[row,8] = -xp
        A[row+1,3] = x
        A[row+1,4] = y
        A[row+1,5] = 1
        A[row+1,6] = -yp*x
        A[row+1,7] = -yp*y
        A[row+1,8] = -yp

    return np.linalg.svd(np.dot(A.T, A))[2][-1, :].reshape((3, 3))


def homographywarp(source, homography, imwidth, imheight):
    """warp image using homography into destination image of size imwidth x
    imheight
    source --- ndarray(shape=(height, width, 3), dtype=np.uint8), source image
    homography --- ndarray(shape=(3, 3), dtype=float), homography transformation
    imwidth --- new image width
    imheight --- new image height"""
    newimage = np.zeros((imheight, imwidth, 3), dtype=np.uint8)

    # hint: you can loop over each pixel in the image, but that's slow.
    #       instead, you can use fancy numpy indexing to do the homography
    #       warp very quickly. check out the methods numpy.meshgrid and
    #       numpy.take.

    #Copied from Kratz computer ~    
    for y in xrange(imheight):
        for x in xrange(imwidth):
            p = np.linalg.solve(homography,[x,y,1])
            p /= p[2]
            p = p.astype(int)
            if(p[0]>=0 and p[0]<imwidth and
               p[1]>=0 and p[1]<imheight):
                #maybe make sure y,x in bounds
                newimage[y,x] = source[p[1],p[0]]

    return newimage


###############################################################################
#                           Begin interface code
###############################################################################

class SelectPoints():
    """interface for selecting corner and vanishing points"""
    def __init__(self, windowwidth, windowheight, imgpath):
        self.corner1 = np.array([windowwidth * 1 / 4, windowheight * 1 / 4])
        self.corner2 = np.array([windowwidth * 3 / 4, windowheight * 3 / 4])
        self.vanishingpt = np.array([windowwidth * 3 / 5, windowheight * 5 / 8])

        self.windowwidth = windowwidth
        self.windowheight = windowheight

        self.selectedpoint = None

        # load image into texture
        self.texture = glGenTextures(1)
        self.image = np.asarray(Image.open(imgpath))
        self.imheight, self.imwidth = self.image.shape[0:2]
        settexture(self.texture, self.image)

    def getCorner1(self):
        return self.corner1 * [self.imwidth, self.imheight] / [self.windowwidth, self.windowheight]

    def getCorner2(self):
        return self.corner2 * [self.imwidth, self.imheight] / [self.windowwidth, self.windowheight]

    def getVanishingPt(self):
        return self.vanishingpt * [self.imwidth, self.imheight] / [self.windowwidth, self.windowheight]

    def activate(self):
        self.resize(self.windowwidth, self.windowheight)

    def resize(self, windowwidth, windowheight):
        # init orthographic projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, windowwidth, windowheight, 0)
        glMatrixMode(GL_MODELVIEW)

        self.windowwidth = windowwidth
        self.windowheight = windowheight

    def keyPressed(self, key, x, y):
        pass

    def mousePressed(self, button, state, x, y):
        if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
            for point in [self.corner1, self.corner2, self.vanishingpt]:
                if np.sum((point - np.array([x, y])) ** 2) < 100.:
                    self.selectedpoint = point

        if state == GLUT_UP:
            self.selectedpoint = None

    def mouseMotion(self, x, y):
        if self.selectedpoint is not None:
            self.selectedpoint[0] = x
            self.selectedpoint[1] = y

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

        # draw background image
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glBegin(GL_QUADS)
        
        glTexCoord2f(0.0, 0.0)
        glVertex2f(0., 0.)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(self.windowwidth, 0.)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(self.windowwidth, self.windowheight)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(0., self.windowheight)

        glEnd()

        # draw corner points and vanishing point
        glDisable(GL_TEXTURE_2D)
        glColor3f(1., 0., 0.)
        for point in [self.corner1, self.corner2, self.vanishingpt]:
            glBegin(GL_QUADS)
            
            glVertex2f(point[0] - 3.0, point[1] - 3.0)
            glVertex2f(point[0] + 3.0, point[1] - 3.0)
            glVertex2f(point[0] + 3.0, point[1] + 3.0)
            glVertex2f(point[0] - 3.0, point[1] + 3.0)

            glEnd()

        # create box for back face
        glBegin(GL_LINE_STRIP)
        
        glVertex2f(self.corner1[0], self.corner1[1])
        glVertex2f(self.corner2[0], self.corner1[1])
        glVertex2f(self.corner2[0], self.corner2[1])
        glVertex2f(self.corner1[0], self.corner2[1])
        glVertex2f(self.corner1[0], self.corner1[1])

        glEnd()

        # draw fake perspective lines from vanishing point
        for point in [np.array([self.corner1[0], self.corner1[1]]),
                      np.array([self.corner2[0], self.corner1[1]]),
                      np.array([self.corner2[0], self.corner2[1]]),
                      np.array([self.corner1[0], self.corner2[1]])]:
            glBegin(GL_LINES)

            glVertex2f(self.vanishingpt[0], self.vanishingpt[1])
            direction = point - self.vanishingpt
            direction *= 1000. / np.linalg.norm(direction)
            glVertex2f(self.vanishingpt[0] + direction[0], self.vanishingpt[1] + direction[1])

            glEnd()


class Navigate():
    """interface for navigating the constructed 3d environment"""
    def __init__(self, windowwidth, windowheight, focallen, imgpath):
        self.windowwidth = windowwidth
        self.windowheight = windowheight
        self.focallen = focallen
        self.image = np.asarray(Image.open(imgpath))

        # viewing information
        self.camera = np.array([200., 200., 300.])
        self.forward = np.array([0., -1.0, 1.]) #was facing away from painting, set to 1 to face it

        self.mousedown = [False, False]
        self.mousedownpt = [0, 0]

        # generate face OpenGL textures
        self.backtex, self.righttex, self.toptex, self.lefttex, self.bottomtex = glGenTextures(5)

    def activate(self, corner1, corner2, vanishingpt):
        self.resize(self.windowwidth, self.windowheight)

        # compute 3d face vertices
        self.back3d, self.right3d, self.top3d, self.left3d, self.bottom3d, \
                back, right, top, left, bottom = \
                computefaces(corner1, corner2, vanishingpt, self.image.shape[1], self.image.shape[0], self.focallen)

        # transform images
        backimg = transformimage(self.image, back, self.back3d)
        rightimg = transformimage(self.image, right, self.right3d)
        topimg = transformimage(self.image, top, self.top3d)
        leftimg = transformimage(self.image, left, self.left3d)
        bottomimg = transformimage(self.image, bottom, self.bottom3d)

        # debugging
        Image.fromarray(backimg).save("debug_back.png")
        Image.fromarray(rightimg).save("debug_right.png")
        Image.fromarray(topimg).save("debug_top.png")
        Image.fromarray(leftimg).save("debug_left.png")
        Image.fromarray(bottomimg).save("debug_bottom.png")

        # pass to OpenGL
        settexture(self.backtex, backimg)
        settexture(self.righttex, rightimg)
        settexture(self.toptex, topimg)
        settexture(self.lefttex, leftimg)
        settexture(self.bottomtex, bottomimg)

    def resize(self, windowwidth, windowheight):
        # init perspective projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(windowwidth) / float(windowheight), 10.0, 10000.0)
        glMatrixMode(GL_MODELVIEW)

        self.windowwidth = windowwidth
        self.windowheight = windowheight

    def keyPressed(self, key, x, y):
        # movement keys
        right = np.cross(self.forward, [0, 1, 0])
        right /= np.linalg.norm(right)

        if key == 'w':
            self.camera += self.forward * 10.
        elif key == 's':
            self.camera -= self.forward * 10.
        elif key == 'a':
            self.camera -= right * 10.
        elif key == 'd':
            self.camera += right * 10.
        elif key == 'q':
            self.camera[1] += 10.
        elif key == 'z':
            self.camera[1] -= 10.

    def mousePressed(self, button, state, x, y):
        # camera look
        if state == GLUT_DOWN:
            self.mousedownpt = np.array([x, y])
            if button == GLUT_LEFT_BUTTON:
                self.mousedown[0] = True
            if button == GLUT_RIGHT_BUTTON:
                self.mousedown[1] = True
        elif state == GLUT_UP:
            if button == GLUT_LEFT_BUTTON:
                self.mousedown[0] = False
            if button == GLUT_RIGHT_BUTTON:
                self.mousedown[1] = False

    def mouseMotion(self, x, y):
        # camera look code
        if self.mousedown[0]:
            vec1 = np.r_[self.mousedownpt - [self.windowwidth / 2, self.windowheight / 2], self.focallen]
            vec2 = np.r_[np.array([x, y]) - [self.windowwidth / 2, self.windowheight / 2], self.focallen]
            rotvec = np.cross(vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2))
            rotvec[1] *= -1.
            if np.linalg.norm(rotvec) > 1e-5:
                diff = self.mousedownpt - [x, y]
                self.forward = np.dot(rotation_matrix(rotvec, np.linalg.norm(diff) / 500.), self.forward)
                #self.forward[1] = np.clip(self.forward[1], -0.7, 0.7)
                self.forward /= np.linalg.norm(self.forward)
            self.mousedownpt = np.array([x, y])
        elif self.mousedown[1]:
            diff = self.mousedownpt[1] - y
            self.camera += self.forward * diff

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glEnable(GL_DEPTH_TEST)

        # set up camera
        gluLookAt(self.camera[0], self.camera[1], self.camera[2],
                self.camera[0] + self.forward[0], self.camera[1] + self.forward[1], self.camera[2] + self.forward[2],
                0., 1., 0.)

        # draw all faces
        glEnable(GL_TEXTURE_2D)

        for tex, face in zip([self.backtex, self.righttex, self.toptex, self.lefttex, self.bottomtex],
                             [self.back3d, self.right3d, self.top3d, self.left3d, self.bottom3d]):
            glBindTexture(GL_TEXTURE_2D, tex)
            glBegin(GL_QUADS)
            glTexCoord2f(1.0, 0.0)
            glVertex3f(face[0, 0], face[0, 1], face[0, 2])
            glTexCoord2f(0.0, 0.0)
            glVertex3f(face[1, 0], face[1, 1], face[1, 2])
            glTexCoord2f(0.0, 1.0)
            glVertex3f(face[2, 0], face[2, 1], face[2, 2])
            glTexCoord2f(1.0, 1.0)
            glVertex3f(face[3, 0], face[3, 1], face[3, 2])
            glEnd()


# state management
state = "selectpoints"
selectpoints = None
navigate = None


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis /= np.linalg.norm(axis)
    a = np.cos(theta/2)
    b, c, d = -axis * np.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def settexture(texture, image):
    imheight = image.shape[0]
    imwidth = image.shape[1]
    image = image.tostring()

    glBindTexture(GL_TEXTURE_2D, texture)

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, 3, imwidth, imheight, 0, GL_RGB, GL_UNSIGNED_BYTE, image)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)


def init(windowwidth, windowheight, imgpath, focallen):
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
        
    global selectpoints
    global navigate
    selectpoints = SelectPoints(windowwidth, windowheight, imgpath)
    navigate = Navigate(windowwidth, windowheight, focallen, imgpath)


def resize(windowwidth, windowheight):
    if windowheight == 0:
        windowheight = 1

    glViewport(0, 0, windowwidth, windowheight)

    if state == "selectpoints":
        selectpoints.resize(windowwidth, windowheight)
    elif state == "navigate":
        navigate.resize(windowwidth, windowheight)


def idle(dummy):
    glutPostRedisplay()

    glutTimerFunc(30, idle, 0)


def draw():
    if state == "selectpoints":
        selectpoints.draw()
    elif state == "navigate":
        navigate.draw()

    glutSwapBuffers()


def keyPressed(*args):
    global state

    # switch state if space bar is pressed
    if args[0] == '\040':
        state = "selectpoints" if state == "navigate" else "navigate"

        if state == "selectpoints":
            selectpoints.activate()
        elif state == "navigate":
            navigate.activate(selectpoints.getCorner1(), selectpoints.getCorner2(), selectpoints.getVanishingPt())
    elif args[0] == 'q':
        sys.exit(0)

    if state == "selectpoints":
        selectpoints.keyPressed(*args)
    elif state == "navigate":
        navigate.keyPressed(*args)


def mousePressed(*args):
    if state == "selectpoints":
        selectpoints.mousePressed(*args)
    elif state == "navigate":
        navigate.mousePressed(*args)


def mouseMotion(*args):
    if state == "selectpoints":
        selectpoints.mouseMotion(*args)
    elif state == "navigate":
        navigate.mouseMotion(*args)


def main(imgpath, focallen):
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(640, 480)
    glutInitWindowPosition(0, 0)
    
    window = glutCreateWindow("Tour into Picture")

    glutDisplayFunc(draw)
    glutReshapeFunc(resize)
    glutKeyboardFunc(keyPressed)
    glutMouseFunc(mousePressed)
    glutMotionFunc(mouseMotion)
    glutTimerFunc(0, idle, 0)

    init(640, 480, imgpath, focallen)

    glutMainLoop()


if __name__ == "__main__":
    description = '\n\n'.join(__doc__.split('\n\n'))
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('image_path', type=str, default="sjerome.jpg",
                        help="path to input file")
    parser.add_argument('focallen', type=float, default=500.,
                        help="focal length of camera")
    if len(sys.argv) == 1:
        parser.print_help()

    args = parser.parse_args()

    main(args.image_path, args.focallen)
