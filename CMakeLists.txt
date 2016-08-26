cmake_minimum_required(VERSION 2.8)
project( l2cHello )
find_package( OpenCV REQUIRED )

add_library( l2cHello SHARED libl2cHello.cpp l2cHello.cpp l2cHello.h )
target_link_libraries( l2cHello ${OpenCV_LIBS} )

