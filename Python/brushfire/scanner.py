import cv2
from matplotlib.pyplot import gray
import numpy as np
import math
from dataclasses import dataclass

UPDATESPEED = 1

@dataclass
class Point:
    x: int
    y: int

@dataclass
class SensorObject:
    point: Point = (0,0)
    distance: float = 0
    angle: float = 0
    #iteration: int = 0

# Method for checking if in range of map
def isValid(img, x, y): 
    p = Point(x,y)
    rows, cols = img.shape[:2]
    # range checking
    if(p.x < cols and p.x >= 0 and p.y < rows and p.y >= 0):
        return True
    else:
        return False

# Checking if point is edge
def isNotEdge(img, x, y):
    color = img[y, x] # img has to be opposite
    # Check if it is a object/edge
    if len(img.shape) == 2:
        if color > 50:
            return True
        else:
            return False
    else:
        if color[0] < 50 and color[1] < 50 and color[2] < 50:
            return True
        else:
            return False

def checkIfLineIsConnected(img, points):
    # TODO: Append to array and do an 'any()?' check to make sure white pixels are present.
    # Or just break if a white pixel is detected?
    
    # For each point check all other points. If any of these has a white pixel between them, the point is valid.
    for i in points:
        for j in points:
            if i == j:
                continue
            else:
                p1 = np.array([i.point.x,i.point.y])
                p2 = np.array([j.point.x,j.point.y]) # Two example points.

                p = p1#; print('p: ', p[0])
                # Subtract points
                d = p2-p1#; print('d: ', d)
                # Find largest value of distance between x and y
                N = np.max(np.abs(d))#; print('N: ', N)
                # Get a new addition value for each point
                s = d/N#; print('s: ', s)
                #print(np.rint(p).astype('int'))
                whites = 0
                blacks = 0
                for ii in range(0,N):
                    p = p+s
                    #print(np.rint(p).astype('int'))
                    #print(img[int(p[0]), int(p[1])])
                    if img[int(p[0]), int(p[1])] == 255:
                        whites += 1
                    else:
                        blacks += 2
                if whites > blacks:
                    return True
    return False

# Method to be run for each move
# Parameters:
# Img the image to plot route
# Current the current point
# Range how long range the scanner has
# Resolution how many directions the scan is
def getSensorData(color_img, gray_img, current, minRange, resolution):
    # Cloned to remove scan drawings from main picture
    if gray_img[current.y, current.x] == 0:
        return 0
    angles = np.zeros(resolution) 
    # Get angles for scan
    for i in range(resolution): 
        # Get angles in radians
        angles[i] = (0 + i * 360 / len(angles)) * math.pi / 180.0
    

    #std::vector<sensorObject> scans;
    scans = []
    # scan in every angle
    for i in range(resolution):
        # code for range scanner in one direction, maximize range while possible
        currentRange = 1
        oi = SensorObject()
        while True:
            x = int(current.x + currentRange * math.cos(angles[i]))
            y = int(current.y + currentRange * math.sin(angles[i]))
            if (isValid(gray_img,x,y) and isNotEdge(gray_img,x,y)):
                oi.point = Point(x, y)
                # setting distance to infinity if not hitting edge
                oi.distance = math.inf
            
            currentRange += 1
            if not isNotEdge(gray_img,x,y):
                oi.point = Point(x, y)
                # if edge calculate the distance
                #print('current.x: ', current.x)
                #print('current.y: ', current.y)
                #print('oi.point.x: ', oi.point.x)
                #print('oi.point.y: ', oi.point.y)
                x_dist = np.sqrt(abs(current.x - oi.point.x)) ** 2
                y_dist = np.sqrt(abs(current.y - oi.point.y)) ** 2
                oi.distance = x_dist + y_dist
                break

            #if isCandy(img,x,y):
            #    # if candy
            #    oi.candyFound = True
            #    print('candy found at: ', oi.point)


            #if norm(oi.point-goal) <= thresh
            #    oi.goalFound = True
            #    print('goal found at: ', oi.point)
            if not (isValid(gray_img,x,y) and isNotEdge(gray_img,x,y) and currentRange <= minRange):
                break

        oi.angle = angles[i]
        # push back all the points from scans in vector
        scans.append(oi)

    
    # Create Voronoi
    closest = []
    closest_val = minRange
    for scan in scans:
        if scan.distance < closest_val:
            closest_val = scan.distance
            closest.clear()
            closest.append(scan)
        elif scan.distance == closest_val:
            closest.append(scan)
    
    #print(len(closest))
    #print(current.x, current.y)
    if len(closest) > 1 and checkIfLineIsConnected(gray_img, closest):
        cv2.circle(color_img, (current.x, current.y), 1, (0,255,0), cv2.FILLED, cv2.LINE_8)


    """
    # Display Scanning    
    for scan in scans: 
        cv2.line(imgToDraw, (current.x, current.y), (scan.point.x, scan.point.y), (200,0,200) ,1,8)
        cv2.circle(imgToDraw, (scan.point.x, scan.point.y), 1, (0,255,0), cv2.FILLED, cv2.LINE_8)
    """
    cv2.namedWindow("tangentBug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("tangentBug", int(len(color_img[0])*1.5), int(len(color_img[1])*1.5))
    cv2.moveWindow("tangentBug", 100, 100)
    cv2.imshow("tangentBug", color_img)
    #cv2.waitKey(0)
    cv2.waitKey(UPDATESPEED) 
    
    


"""
# method for checking if point is candy
def isCandy(img, x, y):
    color = img[y,x]
    if len(img.shape) == 2:
        if color < 50:
            return True
        else:
            return False
    else:
        if color[0] < 50 and color[1] < 50 and color[2] > 150:
            return True
        else:
            return False
    
"""
""" We dont pick up candy
def PickUpCandy(img, scans, current, stepsize, thresh):
    #Point2d candyLoc, returnPoint{current};
    candyLoc = Point()
    returnPoint = current
    stepCount = 0
    print('picking up candy')

    angle = 0
    for i in range(len(scans)):
        # iterate through the scan and
        if scans[i].candyFound:
            candyLoc = scans[i].point
            angle=scans[i].angle

    #double x,y;
    while (distance.euclidean(candyLoc,current) >= thresh):
        current.x = current.x + stepsize * math.cos(angle)
        current.y = current.y + stepsize * math.sin(angle)
        cv2.circle(img,current, 1, (0,255,0), cv2.FILLED, cv2.LINE_8)
        cv2.imshow("tangentBug", img)
        stepCount += 1
        cv2.waitKey(UPDATESPEED)
    
    print("candy picked up")
    
    cv2.circle(img,candyLoc, 6, (255,255,255), cv2.FILLED, cv2.LINE_8)
    angle = math.atan2(returnPoint.y - current.y, returnPoint.x - current.x)
    while(distance.euclidean(returnPoint, current) >= thresh):
        current.x = current.x + stepsize * math.cos(angle)
        current.y = current.y + stepsize * math.sin(angle)
        cv2.circle(img, current, 1, (155,51,51), cv2.FILLED, cv2.LINE_8)
        cv2.imshow("tangentBug", img)
        stepCount += 1
        cv2.waitKey(UPDATESPEED)
    
    print('back on track with candy')
    cv2.destroyWindow("tangentBug")
    return stepCount
"""
""" Its not needed - we dont look for candy
#int scanCandyNPickUp(Mat &img, Point2d &current, Point2d goal, int range, int resolution, int thresh, int stepsize){
def scanCandyNPickUp(img, current, goal, range, resolution, thresh, stepsize):
    scans = getSensorData(img, current, goal, range, resolution, thresh)
    #std::vector<sensorObject> scans = getSensorData(img,current,goal,range,resolution,thresh);
    for scan in scans:
        if scan.candyFound:
            return PickUpCandy(img, scans, current, stepsize, thresh)    
    return 0
"""
"""
# Inspiration: https://stackoverflow.com/questions/3330181/algorithm-for-finding-nearest-object-on-2d-grid
def scanner():
    # Start coordinates
    xs = 0 
    ys = 0

    # Check point (xs, ys)
    maxDistance = 5
    for d in range(1, maxDistance):
        for i in range(d + 1): #for (int i = 0; i < d + 1; i++)
            x1 = xs - d + i
            y1 = ys - i

            # Check point (x1, y1)

            x2 = xs + d - i
            y2 = ys + i

            # Check point (x2, y2)

        for i in range(1, d):
        #for (int i = 1; i < d; i++)
    
            x1 = xs - i
            y1 = ys + d - i

            # Check point (x1, y1)

            x2 = xs + i
            y2 = ys - d + i

            # Check point (x2, y2)
"""

