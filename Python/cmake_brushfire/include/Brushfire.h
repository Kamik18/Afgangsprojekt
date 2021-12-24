#ifndef BRUSHFIRE_H
#define BRUSHFIRE_H
#include <opencv2/opencv.hpp>
#include <map>
#include "Scanner.h"

using namespace cv;
using namespace std;

class Brushfire
{
public:
    // Standard constructor
    Brushfire()
    {

    }
    
    // Iterate through the image in forward direction and give all pixels values.
    // iterate through the image again in the backwards direction and give the rest of the pixel values.
    Mat brushfireAlgorithmGrayScale(int intensityChange)
    {
        // Setup images
        string imageName = "../MapNoPoints.png";
        
        Mat oriImage=imread(imageName.c_str(), CV_LOAD_IMAGE_GRAYSCALE); // Read the file
        if( oriImage.empty() ) // Check for invalid input
        {
            cout << "Could not open or find the image" << endl;
            return Mat{};
        }
        
        Mat image = oriImage;
        
        // Create multiple images to draw both forward and backwards iteration aswell as the final brushfire image.
        Mat brushfireImage, brushfireImageForward, brushfireImageBackwards;
        image.copyTo(brushfireImage);
        image.copyTo(brushfireImageForward);
        image.copyTo(brushfireImageBackwards);

        // Add variables for reduced namoing in the nested forloops.
        int rows = brushfireImage.rows - 1;
        int cols = brushfireImage.cols - 1;
        
        // Brushfire image
        for (int i = 1; i < cols; ++i)
        {
            for (int j = 1; j < rows; ++j)
            {
                // Check pixel to the left.
                if((int)brushfireImage.at<uchar>(Point(i - 1,j)) < (int)brushfireImage.at<uchar>(Point(i,j)))
                {
                    brushfireImage.at<uchar>(Point(i,j)) = (((int)brushfireImage.at<uchar>(Point(i - 1,j)) + intensityChange) <= 255) ? (int)brushfireImage.at<uchar>(Point(i - 1,j)) + intensityChange : 255 ;
                }
                // Check the pixel above
                if((int)brushfireImage.at<uchar>(Point(i,j - 1)) < (int)brushfireImage.at<uchar>(Point(i,j)))
                {
                    brushfireImage.at<uchar>(Point(i,j)) = ((int)brushfireImage.at<uchar>(Point(i,j - 1)) + intensityChange) <= 255 ? (int)brushfireImage.at<uchar>(Point(i,j - 1)) + intensityChange : 255;
                }
                // Check the pixel to the right
                if((int)brushfireImage.at<uchar>(Point(cols - i + 1, rows - j)) < (int)brushfireImage.at<uchar>(Point(cols - i, rows - j)))
                {
                    brushfireImage.at<uchar>(Point(cols - i, rows - j)) = ((int)brushfireImage.at<uchar>(Point(cols - i + 1, rows -j)) + intensityChange) <= 255 ? (int)brushfireImage.at<uchar>(Point(cols - i + 1, rows -j)) + intensityChange: 255;
                }
                // Check the pixel below
                if((int)brushfireImage.at<uchar>(Point(cols - i, rows - j + 1)) < (int)brushfireImage.at<uchar>(Point(cols - i, rows - j)))
                {
                    brushfireImage.at<uchar>(Point(cols - i, rows - j)) = ((int)brushfireImage.at<uchar>(Point(cols - i, rows - j + 1)) + intensityChange) <= 255 ? (int)brushfireImage.at<uchar>(Point(cols - i, rows - j + 1)) + intensityChange : 255;
                }
                
            }
        }
        
        // Forward and backwards brushfire image.
        for (int i = 1; i < cols; ++i)
        {
            for (int j = 1; j < rows; ++j)
            {
                // Check pixel to the left.
                if((int)brushfireImageForward.at<uchar>(Point(i - 1,j)) <= (int)brushfireImageForward.at<uchar>(Point(i,j)))
                {
                    brushfireImageForward.at<uchar>(Point(i,j)) = ((int)brushfireImageForward.at<uchar>(Point(i - 1,j)) + intensityChange) <= 255 ? (int)brushfireImageForward.at<uchar>(Point(i - 1,j)) + intensityChange : 255;
                }
                //// Check the pixel above
                if((int)brushfireImageForward.at<uchar>(Point(i,j - 1)) <= (int)brushfireImageForward.at<uchar>(Point(i,j)))
                {
                    brushfireImageForward.at<uchar>(Point(i,j)) = ((int)brushfireImageForward.at<uchar>(Point(i,j - 1)) + intensityChange) <= 255 ? (int)brushfireImageForward.at<uchar>(Point(i,j - 1)) + intensityChange : 255;
                }
                
                // Do the same backwards.s
                // Check pixel to the right.
                if((int)brushfireImageBackwards.at<uchar>(Point(cols - i + 1, rows - j)) < (int)brushfireImageBackwards.at<uchar>(Point(cols - i, rows - j)))
                {
                    brushfireImageBackwards.at<uchar>(Point(cols - i, rows - j)) = ((int)brushfireImageBackwards.at<uchar>(Point(cols - i + 1, rows -j)) + intensityChange) <= 255 ? (int)brushfireImageBackwards.at<uchar>(Point(cols - i + 1, rows -j)) + intensityChange : 255;
                }
                // Check the pixel below
                if((int)brushfireImageBackwards.at<uchar>(Point(cols - i, rows - j + 1)) < (int)brushfireImageBackwards.at<uchar>(Point(cols - i, rows - j)))
                {
                    brushfireImageBackwards.at<uchar>(Point(cols - i, rows - j)) = ((int)brushfireImageBackwards.at<uchar>(Point(cols - i, rows - j + 1)) + intensityChange) <= 255 ? (int)brushfireImageBackwards.at<uchar>(Point(cols - i, rows - j + 1)) + intensityChange : 255;
                }
            }
        }
        
        // Printing images.
        namedWindow("Original", WINDOW_NORMAL);
        resizeWindow("Original", image.rows, image.cols);
        moveWindow("Original", 0, 0);
        imshow("Original", oriImage);

        namedWindow("BrushfireForward", WINDOW_NORMAL);
        resizeWindow("BrushfireForward", brushfireImageForward.rows, brushfireImageForward.cols);
        moveWindow("BrushfireForward", 525, 0);
        imshow("BrushfireForward", brushfireImageForward);

        namedWindow("BrushfireBackwards", WINDOW_NORMAL);
        resizeWindow("BrushfireBackwards", brushfireImageBackwards.rows, brushfireImageBackwards.cols);
        moveWindow("BrushfireBackwards", 980, 0);
        imshow("BrushfireBackwards", brushfireImageBackwards);
        
        namedWindow("Brushfire", WINDOW_NORMAL);
        resizeWindow("Brushfire", brushfireImage.rows, brushfireImage.cols);
        moveWindow("Brushfire", 1450, 0);
        imshow("Brushfire", brushfireImage);

        return brushfireImage;
    }

    vector<Point> createPath(Mat image)
    {
        // Create an image to display dots and an image to display the final voronoi diagram
        Mat imageColor, imageColor2;
        cvtColor(image, imageColor, CV_GRAY2BGR);
        imageColor.copyTo(imageColor2);

        /************************************************************************
         *                                                                      *
         *                         Setup variables                              *
         *                                                                      *
         * *********************************************************************/

        int divisionSize = 16; // For the split of the image into smaller boxes
        int horizontal = 0, horizontalPrev = 0; // To know where to iterate from in a new box.
        Point localMaxPoint; // Local maxima in boxes
        int numberOfPoints = 0; // Total number of points added to picture
        vector<Point> pointVector; // A vector holding all the points added.
        

        /************************************************************************************
         *                                                                                  *
         * Divide the picture into smaller boxes and find the local maxima for each box.    *
         *                                                                                  *
         * **********************************************************************************/

        for (int boundingBoxHori = 1; boundingBoxHori <= divisionSize/2; ++boundingBoxHori)
        {
            // previous horizontal is set to the current one.
            horizontalPrev = horizontal;
            // Move box for each iteration horizontally
            horizontal = (image.cols / (divisionSize/2)) * boundingBoxHori;
            int vertical = 0, verticalPrev = 0;
            for (int boundingBoxVert = 1; boundingBoxVert <= divisionSize/2; ++boundingBoxVert)
            {
                int localMax = 0;
                verticalPrev = vertical;
                // Move box for each iteration vertically
                vertical = (image.rows / (divisionSize/2)) * boundingBoxVert;
                // For each pixel in the box, determine the local maxima.
                for (int i = horizontalPrev; i < horizontal; ++i)
                {
                    for (int j = verticalPrev; j < vertical; ++j)
                    {
                        if( localMax < (int)image.at<uchar>(Point(i, j)))
                        {
                            localMax = (int)image.at<uchar>(Point(i, j));
                            localMaxPoint = Point(i, j);
                        }
                    }
                }
                // Push back local maxixa to vector
                pointVector.push_back(localMaxPoint);
                // Add dot to image
                circle(imageColor, localMaxPoint, 2, CV_RGB(255,0,0), 3);
                numberOfPoints++;
            }
        }

        // Remove needless points
        cout << "number of Points: " << numberOfPoints << endl;
        cout << "Vector size before: " << pointVector.size() << endl;

        // For each point check surrounding points.
        for (int vectorIndex = 0; vectorIndex < pointVector.size(); ++vectorIndex)
        {
            // If the point is less that 30 in intensity. Remove it.
            if((int)image.at<uchar>(Point(pointVector[vectorIndex].x, pointVector[vectorIndex].y)) < 30)
            {
                pointVector.erase(pointVector.begin() + vectorIndex);
                // Since the vector then decreases in the size. A new point is moved to the placement of the previous deleted one.
                vectorIndex--;
            }
            // If the point is above 30 intensity check if other points are too close.
            else
            {
                for (int i = 0; i < pointVector.size(); ++i)
                {
                    // If distance is less than 60, the point with the lowest value is removed.
                    if(cv::norm(pointVector[vectorIndex] - pointVector[i]) < 60 && vectorIndex != i)
                    {
                        if((int)image.at<uchar>(Point(pointVector[vectorIndex].x, pointVector[vectorIndex].y)) > (int)image.at<uchar>(Point(pointVector[i].x, pointVector[i].y)))
                        {
                            pointVector.erase(pointVector.begin() + i);

                        }
                        // If both are the same. The first one is deleted and we run again with the same index number.
                        else if((int)image.at<uchar>(Point(pointVector[vectorIndex].x, pointVector[vectorIndex].y)) <= (int)image.at<uchar>(Point(pointVector[i].x, pointVector[i].y)))
                        {
                            pointVector.erase(pointVector.begin() + vectorIndex);
                            vectorIndex--;
                            break;
                        }
                    }
                }
            }
        }
        
        // Add dots.
        for (int i = 0; i < pointVector.size(); ++i)
        {
            circle(imageColor2, pointVector[i], 1, CV_RGB(0,255,0), 3);
        }


        /*********************************************************************************************
         *                                                                                           *
         *                                          Connect points                                   *
         *                                                                                           *
         * ******************************************************************************************/

        vector<Point> map;
        vector<vector<int>> connectedPoints;

        for (int var = 0; var < pointVector.size(); ++var)
        {
            cout << var << ": " << pointVector[var] << endl;
        }

        for (int i = 0; i < pointVector.size(); ++i)
        {
            // Create a copy of the vector that holds all remaining points
            vector<Point> pointVectorCopy = pointVector;
            // For each point in the vector. Try and connect to the closest point
            for (int j = 0; j < pointVector.size(); ++j)
            {
                cout << "Checking point nr: " << j;
                // Search through the vector and find the closest point.
                int minDistance = 600;
                int indexToRemove;
                Point closestPoint;
                
                for (int k = 0; k < pointVectorCopy.size(); ++k)
                {
                    vector<int>::iterator it;
                    int indexOfConnectedPoints = 0;
                    bool restart = false;

                    // Here search through every point and choose the closest point
                    int dist = cv::norm(pointVector[i] - pointVectorCopy[k]);
                    if(minDistance > dist && pointVector[i] != pointVectorCopy[k] )
                    {
                        indexToRemove = k;
                        closestPoint = pointVectorCopy[k];
                        minDistance = dist;
                    }
                }

                // If no point is found within distance. Skip point.
                if(minDistance == 600)
                {
                    cout << endl;
                    break;
                }

                // After finds the closest vector, we delete it from the vector, so it doesnt get found again next time.
                pointVectorCopy.erase(pointVectorCopy.begin() + indexToRemove);
                
                // Determine direction
                bool down = false;
                bool right = false;
                int xDistance = pointVector[i].x  - closestPoint.x;
                int yDistance = pointVector[i].y  - closestPoint.y;
                
                if(xDistance < 0)
                    right = true;
                if(yDistance < 0)
                    down = true;
                
                // create a vector to save all steps between points
                vector<Point> steps;
                steps.push_back(pointVector[i]);
                // While loop to find all points that makes a road from one point to another.
                // The loop runs untill it reaches the end point or it hits an obstacle.
                while(steps[steps.size() -1] != closestPoint && (int)image.at<uchar>(Point(steps[steps.size()-1].x, steps[steps.size()-1].y)) != 0)
                {
                    if(down && right)
                    {
                        // Check if pixel to the right is brighter than pixels down right
                        if((int)image.at<uchar>(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y)) > (int)image.at<uchar>(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y + 1)))
                        {
                            steps.push_back(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y));
                        }
                        // Go down right
                        else
                        {
                            steps.push_back(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y + 1));
                        }
                    }
                    if(down && !right)
                    {
                        // Go left
                        if((int)image.at<uchar>(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y)) > (int)image.at<uchar>(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y + 1)))
                        {
                            steps.push_back(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y));
                        }
                        // Go down and left
                        else
                        {
                            steps.push_back(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y + 1));
                        }
                    }
                    if(!down && right)
                    {
                        // Go right
                        if((int)image.at<uchar>(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y)) > (int)image.at<uchar>(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y - 1)))
                        {
                            steps.push_back(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y));
                        }
                        // Go up and right
                        else
                        {
                            steps.push_back(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y - 1));
                        }
                    }
                    if(!down && !right)
                    {
                        // Go left
                        if((int)image.at<uchar>(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y)) > (int)image.at<uchar>(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y - 1)))
                        {
                            steps.push_back(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y));
                        }
                        // Go left and up
                        else
                        {
                            steps.push_back(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y - 1));
                        }
                    }
                    
                    // X is reached.
                    if(steps[steps.size()-1].x == closestPoint.x)
                    {
                        // Go up until the point is reached
                        while(steps[steps.size()-1].y < closestPoint.y && (int)image.at<uchar>(Point(steps[steps.size()-1].x, steps[steps.size()-1].y)) != 0)
                        {
                            steps.push_back(Point(steps[steps.size()-1].x, steps[steps.size()-1].y + 1));
                        }
                        // Go down until the point is reached.
                        while(steps[steps.size()-1].y > closestPoint.y && (int)image.at<uchar>(Point(steps[steps.size()-1].x, steps[steps.size()-1].y)) != 0)
                        {
                            steps.push_back(Point(steps[steps.size()-1].x, steps[steps.size()-1].y - 1));
                        }
                        break;
                    }
                    // Y is reached.
                    if(steps[steps.size()-1].y == closestPoint.y)
                    {
                        // Go right until the point is reached
                        while(steps[steps.size()-1].x < closestPoint.x && (int)image.at<uchar>(Point(steps[steps.size()-1].x, steps[steps.size()-1].y)) != 0)
                        {
                            steps.push_back(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y));
                        }
                        // Go left until the point is reached
                        while(steps[steps.size()-1].x > closestPoint.x && (int)image.at<uchar>(Point(steps[steps.size()-1].x, steps[steps.size()-1].y)) != 0)
                        {
                            steps.push_back(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y));
                        }
                        break;
                    }
                    
                }


                // If the path didn't hit an obstacle. Draw the line and add it to a vector

                if((int)image.at<uchar>(Point(steps[steps.size()-1].x, steps[steps.size()-1].y)) != 0)
                {
                    bool pathAlreadyMade = false;
                    // If its not at an obstacle. We check the whole vector of pointVector to find the one that matches the point we have.
                    for (int l = 0; l < pointVector.size(); ++l)
                    {
                        // When we find the point in the vector we procced.
                        if(closestPoint == pointVector[l])
                        {
                            bool numberAppended = false;
                            // Search through the vector of vectors with int
                            for (int g = 0; g < connectedPoints.size(); ++g)
                            {
                                // Create two iterators that searches for both numbers in every vector.
                                vector<int>::iterator iti = find(connectedPoints[g].begin(), connectedPoints[g].end(), i);
                                vector<int>::iterator itl = find(connectedPoints[g].begin(), connectedPoints[g].end(), l);
                                // If it is found. The new point is added to that vector to tell that they're connected.
                                // If both are found in the same array nothing is done and the vector is neglected as a false path.
                                if(iti != connectedPoints[g].end() && itl != connectedPoints[g].end())
                                {
                                    pathAlreadyMade = true;
                                    break;
                                }
                                if(iti != connectedPoints[g].end() && itl == connectedPoints[g].end())
                                {
                                    connectedPoints[g].push_back(l);
                                    numberAppended = true;
                                }
                                if(iti == connectedPoints[g].end() && itl != connectedPoints[g].end())
                                {
                                    connectedPoints[g].push_back(i);
                                    numberAppended = true;
                                }
                            }
                            // If the path is not made we join the points.
                            if(!pathAlreadyMade)
                            {
                                cout << "Joining " << i << " and " << l << endl;
                            }
                            // If path is made or the number is added to another path. we continue.
                            if(pathAlreadyMade || numberAppended)
                            {
                                continue;
                            }
                            
                            // Otherwise a new vector is created and appeneded to connectedPoints.
                            vector<int> temp {i,l};
                            connectedPoints.push_back(temp);

                        }
                        // To step further out.
                        if(pathAlreadyMade)
                        {
                            cout << "path aldready made" << endl;
                            break;
                        }
                    }
                    // To step further out.
                    if(pathAlreadyMade)
                    {
                        continue;
                    }
                    // Print elements in vectors
                    for (int var = 0; var < connectedPoints.size(); ++var)
                    {
                        cout << "vector nr: " << var << " of size " << connectedPoints[var].size() << " with indexes: [";
                        for (int r = 0; r < connectedPoints[var].size(); ++r)
                        {
                            cout << connectedPoints[var][r] << ", ";
                        }
                        cout << "]" << endl;
                    }
                    // Draw line
                    for (int k = 0; k < steps.size(); ++k)
                    {
                        circle(imageColor2, steps[k], 1, CV_RGB(0,0,255), 1);
                        map.push_back(steps[k]);
                    }
                    // Add vector to map to create a traversing map in the "goToPoint()" method.
                    paths[i] = steps;
                    break;
                }
            }

            // Compare struct to check if elements in vectors are equal
            struct compare
            {
                int key;
                compare(int const &i): key(i){}

                bool operator()(int const &i)
                {
                    return (i == key);
                }
            };

            // We start at the first vector
            for (int a = 0; a < connectedPoints.size(); ++a)
            {
                // We start at the first element in the first vector and check all elements in this vector.
                for (int b = 0; b < connectedPoints[a].size(); ++b)
                {
                    // Then we want to compare that element to all other elements in the rest of the vectors before checing th next one.
                    // Therefore we have to start at the second vector (index 1).
                    for (int c = 1; c < connectedPoints.size(); ++c)
                    {
                        // And in this vector we have to check all points. This is done with an iterator and a compare struct.
                        if(any_of(connectedPoints[c].begin(), connectedPoints[c].end(), compare(connectedPoints[a][b])) && a != c)
                        {
                            cout << "Found point " << connectedPoints[a][b] << " from vector " <<  a << " in vector " << c << endl;
                            // Concatenate vector a and c.
                            connectedPoints[a].insert(connectedPoints[a].end(), connectedPoints[c].begin(), connectedPoints[c].end());

                            // Sort and remove duplicates
                            sort( connectedPoints[a].begin(), connectedPoints[a].end() );
                            connectedPoints[a].erase( unique( connectedPoints[a].begin(), connectedPoints[a].end() ), connectedPoints[a].end() );
                            // Delete old vector.
                            connectedPoints.erase(connectedPoints.begin() + c);
                        }
                    }
                }
            }

            for (int var = 0; var < connectedPoints.size(); ++var)
            {
                cout << "vector nr: " << var << " of size " << connectedPoints[var].size() << " with indexes: [";
                for (int r = 0; r < connectedPoints[var].size(); ++r)
                {
                    cout << connectedPoints[var][r] << ", ";
                }
                cout << "]" << endl;
            }



        }
        

        /*********************************************************************************************
         *                                                                                           *
         *                        Connect different maps together to one                             *
         *                                                                                           *
         * ******************************************************************************************/

        cout << "\n\nConnect roadmaps" << endl;

        // We're only looking at the first vector, since we need to connect all of them it must be possible that this one is the closest to the rest.
        // We start at the first element in the first vector and check all elements in this vector.

        // since we want to add the last lines we connect to my map. We create a variable at the size of the pointvector -1 to get the last element.
        // We go from 10-11 as the point is connected so the map stops at 10.
        int pathSize = pointVector.size() - 1;
        for (int a = 0; a < connectedPoints.size() - 1; ++a)
        {
            cout << "Path size: " << pointVector.size() << endl;
            int distRoadmap = 0, smallestDist = 600, pointVec1 = 0, pointVec2 = 0;
            for (int b = 0; b < connectedPoints[0].size(); ++b)
            {
                // Check for the closest point in the next vector.
                // Therefore we have to start at the second vector (index 1).
                for (vector<int>::iterator it = connectedPoints[a+1].begin(); it != connectedPoints[a+1].end(); it++)
                {
                    distRoadmap = cv::norm(pointVector[connectedPoints[0][b]] - pointVector[*it]);
                    if(distRoadmap < smallestDist)
                    {
                        pointVec1 = connectedPoints[0][b];
                        pointVec2 = *it;
                        smallestDist = distRoadmap;
                    }
                }
            }

            // See if it is possible to make a path between the points
            vector<Point> steps = createLineIfNoWall(image, pointVector[pointVec1], pointVector[pointVec2]);


            if((int)image.at<uchar>(Point(steps[steps.size()-1].x, steps[steps.size()-1].y)) != 0)
            {
                // As a new path is added. The size increments.
                paths[pathSize] = steps;
                pathSize++;
                cout << "point " << pointVec1 << " " << pointVector[pointVec1] << " and point " << pointVec2 << " " << pointVector[pointVec2] << " has been connected" << endl;
                for (int k = 0; k < steps.size(); ++k)
                {

                    circle(imageColor2, steps[k], 1, CV_RGB(0,0,255), 1);
                    map.push_back(steps[k]);
                }
            }
            // If the point cant be connected, we try again and blacklist the point that is not possible to connect
            else
            {
                a--;
                cout << "deleting point " << pointVec1 << endl;
                connectedPoints[0].erase(connectedPoints[0].begin() + pointVec1);
                for (int var = 0; var < connectedPoints[0].size(); ++var)
                {
                    cout << connectedPoints[0][var] << " ";
                }
                cout << endl;

            }
        }

        namedWindow("Dots", WINDOW_NORMAL);
        resizeWindow("Dots", imageColor.cols, imageColor.rows);
        moveWindow("Dots", 0, 550);
        imshow("Dots", imageColor);
        
        namedWindow("Map", WINDOW_NORMAL);
        resizeWindow("Map", imageColor2.cols, imageColor2.rows);
        moveWindow("Map", 525, 550);
        imshow("Map", imageColor2);
        imwrite("../BrushfireMap.jpg", imageColor2);
        
        img = image;
        imgColor = imageColor2;
        copyPointVector = pointVector;

        return map;
    }

    void goToPoint(vector<Point> map)
    {
        Mat mapImage=imread("../Map.png", IMREAD_COLOR); // Read the file
        if( mapImage.empty() ) // Check for invalid input
        {
            cout << "Could not open or find the image" << endl;
            return;
        }

        vector<Point> finalPathStart, finalPathEnd;
        Mat imgColor2;
        imgColor.copyTo(imgColor2);

        // Create start and end point and insert them into the image.
        Point start = Point(15,15);
        //Point end = Point(360,320);
        //circle(imgColor, start, 1, CV_RGB(255,255,0), 3);
        //circle(mapImage, end, 1, CV_RGB(255,0,0), 3);

        // check for the closest point from start to map.
        int distStart = 300;
        int distEnd = distStart;
        Point closestStart, closestEnd;
        int mapStart, mapEnd;
        for (int i = 0; i < map.size(); ++i)
        {
            // Find closest point to start
            if(distStart > cv::norm(start - map[i]))
            {
                mapStart = i;
                closestStart = map[i];
                distStart = cv::norm(start - map[i]);
            }

            // Find closest point to end
            //if(distEnd > cv::norm(end - map[i]))
            //{
            //    mapEnd = i;
            //    closestEnd = map[i];
            //    distEnd = cv::norm(end - map[i]);
            //}
        }

        /*********************************************************************************************
         *                                                                                           *
         *                        Make path from start to end point                                  *
         *                                                                                           *
         * ******************************************************************************************/

        // Add start point
        vector<Point> pathStart = createLineIfNoWall(img, start, closestStart);

        if((int)img.at<uchar>(Point(pathStart[pathStart.size()-1].x, pathStart[pathStart.size()-1].y)) != 0)
        {
            for (int k = 0; k < pathStart.size(); ++k)
            {
                circle(mapImage, pathStart[k], 1, green, 1);
                finalPathStart.push_back(pathStart[k]);
            }
        }

        /*********************************************************************************************
         *                                                                                           *
         *                        Traverse through entire map                                        *
         *                                                                                           *
         * ******************************************************************************************/

        // Add vector 0 and 3 together.
        vector<Point> line0And3;
        line0And3.insert(line0And3.end(), paths[0].begin(), paths[0].end());
        line0And3.insert(line0And3.end(), paths[3].begin(), paths[3].end());

        // Add vector 8 and 11 together
        vector<Point> line8And11;
        line8And11.insert(line8And11.end(), paths[11].begin(), paths[11].end());
        line8And11.insert(line8And11.end(), paths[8].begin(), paths[8].end());

        // Add vector 2 and 6 together
        vector<Point> line2And5;
        line2And5.insert(line2And5.end(), paths[5].begin(), paths[5].end());
        line2And5.insert(line2And5.end(), paths[2].rbegin(), paths[2].rend());

        // Add vector 9, 10 and 12 together
        vector<Point> line9And10And12;
        line9And10And12.insert(line9And10And12.end(), paths[12].begin(), paths[12].end());
        line9And10And12.insert(line9And10And12.end(), paths[10].begin(), paths[10].end());
        line9And10And12.insert(line9And10And12.end(), paths[9].rbegin(), paths[9].rend());


        // Append all vectors into a big vector.
        vector<Point> structuredMap;
        structuredMap.insert(structuredMap.end(), line0And3.begin(), line0And3.end());
        structuredMap.insert(structuredMap.end(), line8And11.begin(), line8And11.end());
        structuredMap.insert(structuredMap.end(), paths[4].begin(), paths[4].end());
        structuredMap.insert(structuredMap.end(), paths[1].rbegin(), paths[1].rend());
        structuredMap.insert(structuredMap.end(), line2And5.begin(), line2And5.end());
        structuredMap.insert(structuredMap.end(), line9And10And12.begin(), line9And10And12.end());

        namedWindow("Path", WINDOW_NORMAL);
        resizeWindow("Path", imgColor.cols, imgColor.rows);
        moveWindow("Path", 980, 550);
        imshow("Path", imgColor);

        namedWindow("ReturnPath", WINDOW_NORMAL);
        resizeWindow("ReturnPath", imgColor.cols, imgColor.rows);
        moveWindow("ReturnPath", 1450, 550);


        int stepCount = 0;
        bool entireMapTraversed = false;
        do
        {
            // Start and end points are created in line 690-694.
            for (int i = 0; i < structuredMap.size(); ++i)
            {
                stepCount++;
                //circle(mapImage, structuredMap[i], 1, CV_RGB(0,255,255), 1);
                Point2d current (structuredMap[i]);
                circle(mapImage, structuredMap[i], 1, green, 1);
                int pointFound = scanCandyNPickUp(mapImage, current, Point2d(-10,-10), 100, 72, 2, 2);
                imshow("ReturnPath", mapImage);
                //waitKey(1);
                if(pointFound != 0)
                {
                    cout << "Length of pointFound: " << pointFound << endl;
                    cout << "stepCountBefore" << stepCount << endl;
                    stepCount += pointFound;
                    cout << "stepCountAfter" << stepCount << endl;
                    closestEnd = structuredMap[i];
                    break;
                }
                if(structuredMap[i] == structuredMap[structuredMap.size()-1])
                {
                    cout << "Done" << endl;
                    entireMapTraversed = true;
                    break;
                }
            }


            /*********************************************************************************************
         *                                                                                           *
         *                                 Return to dropoff                                         *
         *                                                                                           *
         * ******************************************************************************************/
            if(!entireMapTraversed)
            {
                // Find closest point


                int minDistEnd = 300;
                int closestPoint;
                Point endVectorPoint;
                for (int i = 0; i < copyPointVector.size(); ++i)
                {
                    // Find closest big point from pointVector to End.
                    if(minDistEnd > cv::norm(closestEnd - copyPointVector[i]))
                    {
                        closestPoint = i;
                        endVectorPoint = copyPointVector[i];
                        minDistEnd = cv::norm(closestEnd - copyPointVector[i]);
                    }
                }

                // A switch goes creates a shorter vector to return to dropoff in regards to endpoint.
                vector<Point> returnPath{};
                switch(closestPoint)
                {
                case 0:
                {
                    returnPath.insert(returnPath.end(), finalPathEnd.rbegin(), finalPathEnd.rend());
                    returnPath.insert(returnPath.end(), finalPathStart.rbegin(), finalPathStart.rend());
                    break;
                }
                case 1:
                {
                    returnPath.insert(returnPath.end(), finalPathEnd.rbegin(), finalPathEnd.rend());
                    returnPath.insert(returnPath.end(), paths[1].begin(), paths[1].end());
                    returnPath.insert(returnPath.end(), paths[4].rbegin(), paths[4].rend());
                    returnPath.insert(returnPath.end(), line0And3.rbegin(), line0And3.rend());
                    returnPath.insert(returnPath.end(), finalPathStart.rbegin(), finalPathStart.rend());
                    break;
                }
                case 2:
                {
                    returnPath.insert(returnPath.end(), finalPathEnd.rbegin(), finalPathEnd.rend());
                    returnPath.insert(returnPath.end(), line2And5.rbegin(), line2And5.rend());
                    returnPath.insert(returnPath.end(), paths[4].rbegin(), paths[4].rend());
                    returnPath.insert(returnPath.end(), line0And3.rbegin(), line0And3.rend());
                    returnPath.insert(returnPath.end(), finalPathStart.rbegin(), finalPathStart.rend());
                    break;
                }
                case 3:
                {
                    returnPath.insert(returnPath.end(), finalPathEnd.rbegin(), finalPathEnd.rend());
                    returnPath.insert(returnPath.end(), paths[0].rbegin(), paths[0].rend());
                    returnPath.insert(returnPath.end(), finalPathStart.rbegin(), finalPathStart.rend());
                    break;
                }
                case 4:
                {
                    returnPath.insert(returnPath.end(), finalPathEnd.rbegin(), finalPathEnd.rend());
                    returnPath.insert(returnPath.end(), line0And3.rbegin(), line0And3.rend());
                    returnPath.insert(returnPath.end(), finalPathStart.rbegin(), finalPathStart.rend());
                    break;
                }
                case 5:
                {
                    returnPath.insert(returnPath.end(), finalPathEnd.rbegin(), finalPathEnd.rend());
                    returnPath.insert(returnPath.end(), paths[4].rbegin(), paths[4].rend());
                    returnPath.insert(returnPath.end(), line0And3.rbegin(), line0And3.rend());
                    returnPath.insert(returnPath.end(), finalPathStart.rbegin(), finalPathStart.rend());
                    break;
                }
                case 6:
                {
                    returnPath.insert(returnPath.end(), finalPathEnd.rbegin(), finalPathEnd.rend());
                    returnPath.insert(returnPath.end(), paths[5].rbegin(), paths[5].rend());
                    returnPath.insert(returnPath.end(), paths[4].rbegin(), paths[4].rend());
                    returnPath.insert(returnPath.end(), line0And3.rbegin(), line0And3.rend());
                    returnPath.insert(returnPath.end(), finalPathStart.rbegin(), finalPathStart.rend());
                    break;
                }
                case 7:
                {
                    returnPath.insert(returnPath.end(), finalPathEnd.rbegin(), finalPathEnd.rend());
                    returnPath.insert(returnPath.end(), line8And11.rbegin(), line8And11.rend());
                    returnPath.insert(returnPath.end(), line0And3.rbegin(), line0And3.rend());
                    returnPath.insert(returnPath.end(), finalPathStart.rbegin(), finalPathStart.rend());
                    break;
                }
                case 8:
                {
                    returnPath.insert(returnPath.end(), finalPathEnd.rbegin(), finalPathEnd.rend());
                    returnPath.insert(returnPath.end(), paths[11].rbegin(), paths[11].rend());
                    returnPath.insert(returnPath.end(), line0And3.rbegin(), line0And3.rend());
                    returnPath.insert(returnPath.end(), finalPathStart.rbegin(), finalPathStart.rend());
                    break;
                }
                case 9:
                {
                    returnPath.insert(returnPath.end(), finalPathEnd.rbegin(), finalPathEnd.rend());
                    returnPath.insert(returnPath.end(), line9And10And12.rbegin(), line9And10And12.rend());
                    returnPath.insert(returnPath.end(), paths[4].rbegin(), paths[4].rend());
                    returnPath.insert(returnPath.end(), line0And3.rbegin(), line0And3.rend());
                    returnPath.insert(returnPath.end(), finalPathStart.rbegin(), finalPathStart.rend());
                    break;
                }
                case 10:
                {
                    returnPath.insert(returnPath.end(), finalPathEnd.rbegin(), finalPathEnd.rend());
                    returnPath.insert(returnPath.end(), paths[12].rbegin(), paths[12].rend());
                    returnPath.insert(returnPath.end(), paths[4].rbegin(), paths[4].rend());
                    returnPath.insert(returnPath.end(), line0And3.rbegin(), line0And3.rend());
                    returnPath.insert(returnPath.end(), finalPathStart.rbegin(), finalPathStart.rend());
                    break;
                }
                case 11:
                {
                    returnPath.insert(returnPath.end(), finalPathEnd.rbegin(), finalPathEnd.rend());
                    returnPath.insert(returnPath.end(), paths[10].rbegin(), paths[10].rend());
                    returnPath.insert(returnPath.end(), paths[12].rbegin(), paths[12].rend());
                    returnPath.insert(returnPath.end(), paths[4].rbegin(), paths[4].rend());
                    returnPath.insert(returnPath.end(), line0And3.rbegin(), line0And3.rend());
                    returnPath.insert(returnPath.end(), finalPathStart.rbegin(), finalPathStart.rend());
                    break;
                }

                }

                // Move window to specified place
                namedWindow("ReturnPath", WINDOW_NORMAL);
                resizeWindow("ReturnPath", imgColor2.cols, imgColor2.rows);
                moveWindow("ReturnPath", 1450, 550);

                bool startPath = false;

                for (int k = 0; k < returnPath.size(); ++k)
                {
                    if(returnPath[k] == closestEnd)
                    {
                        startPath = true;
                    }
                    if(startPath)
                    {
                        circle(mapImage, returnPath[k], 1, blue, 1);
                        imshow("ReturnPath", mapImage);
                        //waitKey(5);
                    }

                }
            }
            cout << "stepCount: " << stepCount << endl;
            waitKey(0);
        } while(!entireMapTraversed);
        destroyAllWindows();
        cout << "stepCount finished: " << stepCount << endl;
        cout << "size of map: " << structuredMap.size() << endl;
    }


private:
    Mat img;
    Mat imgColor;
    vector<Point> copyPointVector;
    map<int, vector<Point>> paths;
    Scalar green = CV_RGB(0,153,0);
    Scalar blue = CV_RGB(51,51,255);


    void downAndRight(Mat image, vector<Point>& steps)
    {
        // Check if pixel to the right is brighter than pixels down right
        if((int)image.at<uchar>(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y)) > (int)image.at<uchar>(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y + 1)))
        {
            steps.push_back(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y));
        }
        // Check steps down
        else if((int)image.at<uchar>(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y + 1)) > (int)image.at<uchar>(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y + 1)))
        {
            steps.push_back(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y + 1));
        }
        else if((int)image.at<uchar>(Point(steps[steps.size()-1].x, steps[steps.size()-1].y + 1)) > (int)image.at<uchar>(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y + 1)))
        {
            steps.push_back(Point(steps[steps.size()-1].x, steps[steps.size()-1].y + 1));
        }
        else
        {
            steps.push_back(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y + 1));
        }
    }
    void downAndLeft(Mat image, vector<Point>& steps)
    {
        if((int)image.at<uchar>(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y)) >= (int)image.at<uchar>(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y + 1)))
        {
            steps.push_back(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y));
        }
        else
        {
            steps.push_back(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y + 1));
        }
    }
    void upAndRight(Mat image, vector<Point>& steps)
    {
        if((int)image.at<uchar>(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y)) >= (int)image.at<uchar>(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y - 1)))
        {
            steps.push_back(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y));
        }
        else
        {
            steps.push_back(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y - 1));
        }
    }
    void upAndLeft(Mat image, vector<Point>& steps)
    {
        if((int)image.at<uchar>(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y)) >= (int)image.at<uchar>(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y - 1)))
        {
            steps.push_back(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y));
        }
        else
        {
            steps.push_back(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y - 1));
        }
    }
    vector<Point> createLineIfNoWall(Mat image, Point start, Point closest)
    {
        bool down = false;
        bool right = false;
        int xDistance = start.x  - closest.x;
        int yDistance = start.y  - closest.y;

        if(xDistance < 0)
            right = true;
        if(yDistance < 0)
            down = true;

        vector<Point> steps {start};
        //steps.push_back(start);

        while(steps[steps.size() -1] != closest && (int)image.at<uchar>(Point(steps[steps.size()-1].x, steps[steps.size()-1].y)) != 0)
        {
            if(down && right)
            {
                downAndRight(image, steps);
            }
            if(down && !right)
            {
                downAndLeft(image, steps);
            }
            if(!down && right)
            {
                upAndRight(image, steps);
            }
            if(!down && !right)
            {
                upAndLeft(image, steps);
            }

            // X is reached.
            if(steps[steps.size()-1].x == closest.x)
            {
                // Go up until the point is reached
                while(steps[steps.size()-1].y < closest.y && (int)image.at<uchar>(Point(steps[steps.size()-1].x, steps[steps.size()-1].y)) != 0)
                {
                    steps.push_back(Point(steps[steps.size()-1].x, steps[steps.size()-1].y + 1));
                }
                // Go down until the point is reached.
                while(steps[steps.size()-1].y > closest.y && (int)image.at<uchar>(Point(steps[steps.size()-1].x, steps[steps.size()-1].y)) != 0)
                {
                    steps.push_back(Point(steps[steps.size()-1].x, steps[steps.size()-1].y - 1));
                }
                break;
            }

            if(steps[steps.size()-1].y == closest.y)
            {
                while(steps[steps.size()-1].x < closest.x && (int)image.at<uchar>(Point(steps[steps.size()-1].x, steps[steps.size()-1].y)) != 0)
                {
                    steps.push_back(Point(steps[steps.size()-1].x + 1, steps[steps.size()-1].y));
                }
                while(steps[steps.size()-1].x > closest.x && (int)image.at<uchar>(Point(steps[steps.size()-1].x, steps[steps.size()-1].y)) != 0)
                {
                    steps.push_back(Point(steps[steps.size()-1].x - 1, steps[steps.size()-1].y));
                }
                break;
            }
        }
        return steps;
    }

};

#endif // BRUSHFIRE_H

/*
 *
 * CODE THAT WAS USED AND MIGHT BE USED AGAIN AND THEREFORE SAVED.
 *
 cout << endPoint << endl;
 circle(imageColor, startPoint, 2, CV_RGB(255,0,0), 3);
 circle(imageColor, Point(100,80), 2, CV_RGB(255,0,0), 3);
 circle(imageColor, endPoint, 2, CV_RGB(255,0,0), 3);

        /*********************************************
         *          PRINT VECTORS IN MAP
         *
         * ********************************************
        for (int i = 0; i < paths.size(); ++i)
        {
            if(!paths[i].empty())
            {
                cout << "point nr: " << i << " with vector: [";
                for (int j = 0; j < paths[i].size(); ++j)
                {
                    cout << paths[i].at(j) << ", ";
                }
                cout << endl;
                cout << "vector " << i << " has size: " << paths[i].size() << endl;
            }
        }

        for (int i = 0; i < paths.size(); ++i)
        {
            cout << "vector nr: " << i << " with size: " << paths[i].size() << endl;
        }

        cout << "path end: " << paths.size() << endl;

            /*********************************************************************************************
         *                                                                                           *
         *                              END OF SCANNER IMPLEMENTATION                                *
         *                                                                                           *
         * ******************************************************************************************/

            //// Add end point
            //vector<Point> pathEnd = createLineIfNoWall(img, end, closestEnd);
            //
            //if((int)img.at<uchar>(Point(pathEnd[pathEnd.size()-1].x, pathEnd[pathEnd.size()-1].y)) != 0)
            //{
            //    for (int k = pathEnd.size(); k > 0; --k)
            //    {
            //        circle(imgColor, pathEnd[k], 1, CV_RGB(0,255,255), 1);
            //        finalPathEnd.push_back(pathEnd[k]);
            //        imshow("Path", imgColor);
            //        waitKey(5);
            //    }
            //}

/*
*/
