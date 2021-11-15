Howto:
https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

Labels er opdelt således:
class xcen ycen width height
0 0.xx 0.xx 0.xx 0.xx

One row per object

Box coordinates must be in normalized xywh format (from 0 - 1). 
If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.

Class numbers are zero-indexed (start from 0).