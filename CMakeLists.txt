cmake_minimum_required(VERSION 3.0)
project(SOFT_FEATURES)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG} -Wall -std=c++11")



list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
   

#SET(G2O_ROOT /usr/local/include/g2o)

find_package( OpenCV 3.4 REQUIRED )
find_package(PCL 1.5 REQUIRED)
find_package(G2O REQUIRED)
find_package(CSparse REQUIRED)
# add_subdirectory(evaluate)

include_directories(${OpenCV_INCLUDE_DIRS} 
                    ${G2O_INCLUDE_DIRS}
                    ${CSPARSE_INCLUDE_DIR}
                    "/usr/include/eigen3/"    
                    ${PROJECT_SOURCE_DIR}/inc 
                    ${PCL_INCLUDE_DIRS}           
                    )

# include_directories(evaluate)


link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(feature SHARED "src/feature.cpp")
add_library(bucket SHARED "src/bucket.cpp")
add_library(utils SHARED "src/utils.cpp")
add_library(visualOdometry SHARED "src/visualOdometry.cpp")
add_library(MapPoint SHARED "src/MapPoint.cpp")

#add_library(Frame SHARED "Frame.cpp")


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
message(${PROJECT_BINARY_DIR})
message(STATUS "opencv: ${OpenCV_VERSION}")
message(STATUS "g2o:  ${G2O_INCLUDE_DIRS}")
message(STATUS "csparse:  ${CSPARSE_INCLUDE_DIR}")

add_executable( run src/main.cpp )
target_link_libraries( bucket ${OpenCV_LIBS} )
target_link_libraries( feature ${OpenCV_LIBS} bucket)
target_link_libraries( utils feature MapPoint ${OpenCV_LIBS} ${PCL_LIBRARIES})

target_link_libraries( visualOdometry MapPoint utils bucket feature ${OpenCV_LIBS} ${CSPARSE_LIBRARY} g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)


#target_link_libraries( Frame ${OpenCV_LIBS} )

target_link_libraries( MapPoint ${OpenCV_LIBS} )

target_link_libraries( run ${OpenCV_LIBS} bucket feature utils visualOdometry MapPoint ${PCL_LIBRARIES})
