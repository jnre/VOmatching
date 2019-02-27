#ifndef UTILS_H
#define UTILS_H

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <ctype.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common_headers.h>
#include <pcl/registration/icp.h>

#include "feature.hpp"
//#include "matrix.hpp"
#include "MapPoint.hpp"

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

void loadImageLeft(cv::Mat & image,cv::Mat & no_color,int frame_id, std::string filepath);

void loadImageRight(cv::Mat & image,cv::Mat & no_color,int frame_id, std::string filepath);

void loadImageLeft(cv::VideoCapture & cap0,cv::Mat & image,cv::Mat & no_color,int frame_id);

void loadImageRight(cv::VideoCapture & cap1,cv::Mat & image,cv::Mat & no_color,int frame_id);

void mapPointsToPointCloudsAppend(std::vector<MapPoint>& mapPoints, PointCloud::Ptr cloud);

//void simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer);

cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R);

bool isRotationMatrix(cv::Mat &R);

void integrateOdometryStereo(int frame_id, cv::Mat& rigid_body_transformation, cv::Mat& frame_pose, const cv::Mat& rotation, 
    const cv::Mat& translation_stereo);

//void display(int frame_id, cv::Mat& trajectory, cv::Mat& pose, std::vector<Matrix>& pose_matrix_gt, float fps, bool showgt);    
void display(int frame_id, cv::Mat& trajectory, cv::Mat& pose, float fps, bool showgt,const cv::Vec3d & rotation, const cv::Mat& translation_stereo,std::vector<cv::Point3d> pose_estimator_data);    
void pclDisplay(const cv::Mat &triangulated1, const cv::Mat &triangulated0, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1,pcl::PointCloud<pcl::PointXYZ>::Ptr cloud0,boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer);

void getOdometryMatch(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud0,const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1);

template <class SrcType, class DstType>
void convert2(std::vector<SrcType>& src, std::vector<DstType>& dst) {
  dst.resize(src.size());
  std::copy(src.begin(), src.end(), dst.begin());
}
#endif