#include "l2cHello.h"
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

const char* Hello::World()
{
    return "Hello World!\n";
}

int Hello::ShowImage(const char* fpath)
{
    cv::Mat image;
    printf("reading image file... \n");

    image = cv::imread( fpath, CV_LOAD_IMAGE_COLOR);
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    cv::namedWindow("Display Image", CV_WINDOW_AUTOSIZE );
    cv::imshow("Display Image", image);
    
    printf("showing window... \n");
    cv::waitKey(0); 
    return 0;
}

