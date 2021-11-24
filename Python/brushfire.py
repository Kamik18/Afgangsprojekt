import cv2

oriImage = cv2.imread("./ICP/Kort_binaer.png")


brushfireImage = oriImage.copy()
brushfireImageForward = oriImage.copy()
brushfireImageReverse = oriImage.copy()

rows, cols, _ = brushfireImage.shape

print(rows)
print(cols)
intensityChange = 0
#Brushfire image
#for (int i = 1; i < cols; ++i)
for i in range(cols):
    for j in range(rows):
        # Check pixel to the left.
        if brushfireImage[i - 1][j] < brushfireImage[i][j]:
            brushfireImage[i][j] = ((brushfireImage[i - 1][j] + intensityChange) <= 255) ? brushfireImage[i - 1][j] + intensityChange : 255
        
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