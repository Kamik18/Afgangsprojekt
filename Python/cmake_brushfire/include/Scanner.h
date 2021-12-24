#ifndef SCANNER_H
#define SCANNER_H
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <limits>

using namespace cv;

Scalar red(0,0,255);
Scalar green(0,255,0);
Scalar blue(255,0,0);
Scalar white(255,255,255);
Scalar pink(255,51,51);
Scalar orange(200,200,200);
const int UPDATESPEED = 1;
//defining sensorObject
struct sensorObject{
    Point2d point;
    double distance;
    double angle;
    bool goalFound{0};
    bool candyFound{0};
};
//method for picking up candy when found and returning to the first position
int pickUpCandy(Mat &img, std::vector<sensorObject> scans, Point2d &current, int stepsize, int thresh){
    Point2d candyLoc, returnPoint{current};
    int stepCount = 0;
    std::cout << "picking up candy" << std::endl;

    double angle{0};
    for (int i = 0;i<scans.size();i++) {
        //iterate through the scan and
        if(scans[i].candyFound){
            candyLoc = scans[i].point;
            angle=scans[i].angle;
        }
    }
    double x,y;
    while(norm(candyLoc-current) >=thresh){
        current.x = current.x+stepsize*cos(angle);
        current.y = current.y+stepsize*sin(angle);
        circle(img,current,1,green,FILLED,LINE_8);
        imshow("tangentBug", img);
        stepCount++;
        waitKey(UPDATESPEED);
    }
    std::cout << "candy picked up" << std::endl;
    circle(img,candyLoc,6,white,FILLED,LINE_8);
    angle=atan2(returnPoint.y-current.y,returnPoint.x-current.x);
    while(norm(returnPoint-current) >=thresh){
        current.x = current.x+stepsize*cos(angle);
        current.y = current.y+stepsize*sin(angle);
        circle(img,current,1,pink,FILLED,LINE_8);
        imshow("tangentBug", img);
        stepCount++;
        waitKey(UPDATESPEED);
    }
    std::cout << "back on track with candy" << std::endl;
    destroyWindow("tangentBug");
    return stepCount;
}

//method for checking if point is candy
bool isCandy(Mat &img, double x, double y){
    Point2d p(x,y);
    Vec3b colour = img.at<Vec3b>(p);
    if(colour[0] < 50 && colour[1] < 50 && colour[2]>150){
        return true;
    }
    else{
        return false;
    }
}
//checking if point is edge
bool isNotEdge(Mat &img, double x, double y){
    Point2d p(x,y);
    Vec3b colour = img.at<Vec3b>(p);
    //check if it is a object/edge
    if(colour[0] > 50 || colour[1] > 50 || colour[2] > 50)
        return true;
    else{
        return false;}
}

//method for checking if in range of map
bool isValid(Mat& img, double x, double y){
    Point2d p(x,y);
    //range checking
    if(p.x<img.cols && p.x>= 0 && p.y <img.rows && p.y >=0){
        return true;
    }
    else{
        return false;
    }
}

//method for scanning the area and getting the sensorObject points, which holds if goal is found and candy is found
//scan is done with given range and resolution
std::vector<sensorObject> getSensorData(Mat img, Point2d current, Point2d goal, int range, int resolution, int thresh){
    //cloned to remove scan drawings from main picture
    Mat imgToDraw = img.clone();
    std::vector<double> angles(resolution);
    //get angles for scan
    for (int i =0; i<angles.size();i++) {
        //get angles in radians
        angles[i]=(0+i*360/angles.size())*CV_PI/180.0;
    }

    std::vector<sensorObject> scans;
    //scan in every angle
    for (int i = 0;i<angles.size();i++) {
        //code for range scanner in one direction, maximize range while possible
        int currentRange{0};
        double x,y;
        sensorObject oi;
        oi.candyFound=false;
        do {
            x = current.x+currentRange*cos(angles[i]);
            y = current.y+currentRange*sin(angles[i]);
            if(isValid(img,x,y) && isNotEdge(img,x,y)){
                oi.point=Point2d(x,y);
                //setting distance to infinity if not hitting edge
                oi.distance=std::numeric_limits<double>::max();
            }
            currentRange++;
            if(!isNotEdge(img,x,y)){
                //if edge calculate the distance
                oi.distance = norm(current-oi.point);
            }

            if(isCandy(img,x,y)){
                //if candy
                oi.candyFound=true;
                std::cout << "candy found at:" << oi.point << std::endl;
            }

            //if(norm(oi.point-goal) <= thresh){
            //    oi.goalFound=true;
            //    std::cout << "goal found at:" << oi.point << std::endl;
            //}

        } while (isValid(img,x,y) && isNotEdge(img,x,y) && currentRange<=range && !oi.goalFound && !oi.candyFound);
        oi.angle=angles[i];
        //push back all the points from scans in vector
        scans.push_back(oi);
    }
    for (int i =0; i<scans.size();i++) {
        line(imgToDraw,current,scans[i].point,orange,1,8);
        circle(imgToDraw,scans[i].point,1,green,FILLED,LINE_8);
    }
    namedWindow("tangentBug", WINDOW_NORMAL);
    resizeWindow("tangentBug", img.cols, img.rows);
    moveWindow("tangentBug", 1450, 550);
    imshow("tangentBug", imgToDraw);
    waitKey(UPDATESPEED);
    return scans;
}

//method to be run for each move
//method will scan for both goal and candies
//parameters:
//img the image to plot route
//current the current point
//goal the point for the goal(or sub goal)
//range how long range the scanner has
//resolution how many directions the scan is
//thresh how close the scan should be to count as goal and how close the robot needs to be to candy to have picked it up
//stepsize how long each step toward candy is
int scanCandyNPickUp(Mat &img, Point2d &current, Point2d goal, int range, int resolution, int thresh, int stepsize){
    std::vector<sensorObject> scans = getSensorData(img,current,goal,range,resolution,thresh);
    for(sensorObject scan : scans){
        if(scan.candyFound){
            return pickUpCandy(img, scans, current,stepsize,thresh);
        }
    }
    return 0;

}

#endif // SCANNER_H
