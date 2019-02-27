#ifndef VISUAL_ODOM_H
#define VISUAL_ODOM_H

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
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <opencv2/core/eigen.hpp>

#include "feature.hpp"
#include "MapPoint.hpp"

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

void matchingFeatures(cv::Mat& imageLeft_t0, cv::Mat& imageRight_t0,
                      cv::Mat& imageLeft_t1, cv::Mat& imageRight_t1, 
                      FeatureSet& currentVOFeatures,
                      std::vector<MapPoint> MapPoints,
                      std::vector<cv::KeyPoint>&  pointsLeft_t0, 
                      std::vector<cv::KeyPoint>&  pointsRight_t0, 
                      std::vector<cv::KeyPoint>&  pointsLeft_t1, 
                        std::vector<cv::KeyPoint>& pointsRight_t1,
                        cv::Mat &F,cv::Rect &validRoi1,cv::Rect &validRoi2,
                        cv::Ptr<cv::ORB> &orb_detector);

void checkValidMatch(std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& points_return, std::vector<bool>& status, int threshold);

void checkValidMatchEpipolar(std::vector<cv::Point2f> &tempPoint2f_pointsLeft_t0,std::vector<cv::Point2f> &tempPoint2f_pointsLeft_t1,
std::vector<cv::Point2f> &tempPoint2f_pointsRight_t0,std::vector<cv::Point2f> &tempPoint2f_pointsRight_t1,std::vector<bool> & status_epipolarmatch);

void trackingFrame2Frame(cv::Mat& projMatrl, cv::Mat& projMatrr,
                         std::vector<cv::KeyPoint>&  keypointsLeft_t0,
                         std::vector<cv::KeyPoint>&  keypointsLeft_t1, 
                         cv::Mat& points3D_t0,
                         cv::Mat& rotation,
                         cv::Mat& translation,cv::Vec3d & rotation_euler, std::string & mode);

void removeInvalidPoints(std::vector<cv::Point2f>& points, const std::vector<bool>& status);

void removeInvalidAge(std::vector<int> &currentVOfeaturesage, const std::vector<bool>& status);

void getEulerAngles(cv::Mat &rvec_solvepnp,cv::Vec3d &rotation_euler);

void displayTracking(cv::Mat& imageLeft_t1, 
                     std::vector<cv::KeyPoint>&  keypointsLeft_t0,
                     std::vector<cv::KeyPoint>&  keypointsLeft_t1);

void distinguishNewPoints(std::vector<cv::Point2f>&  newPoints, 
                          std::vector<bool>& valid,
                          std::vector<MapPoint>& mapPoints,
                          int frameId_t0,
                          cv::Mat& points3DFrame_t0,
                          cv::Mat& points3DFrame_t1,
                          cv::Mat& points3DWorld,
                          std::vector<cv::Point2f>&  currentPointsLeft_t0, 
                          std::vector<cv::Point2f>&  currentPointsLeft_t1, 
                          std::vector<FeaturePoint>&  currentFeaturePointsLeft,
std::vector<FeaturePoint>& oldFeaturePointsLeft);

void bundleAdjustment (
    const std::vector<cv::Point3f> points_3d,
    const std::vector<cv::Point2f> points_2d,
    const cv::Mat& K,
    cv::Mat& R, cv::Mat& t);


#endif