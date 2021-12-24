#include <iostream>
#include <opencv2/opencv.hpp>
#include "Brushfire.h"

using namespace cv;

int main() 
{

    Brushfire someImage;

    // Iterate through the picture in divisions of it and set points at local maxima to see if it is possible to make lines.
    // make cost'

    // Check surronding 4 pixels. If any of these are larger, dont draw.

    Mat brushfireImg = someImage.brushfireAlgorithmGrayScale(1);
    vector<Point> map = someImage.createPath(brushfireImg);
    someImage.goToPoint(map);
    //vector<Point> map = someImage.createPath(someImage.brushfireAlgorithmGrayScale(1));


    return 0;
}
