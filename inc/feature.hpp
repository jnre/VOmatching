#ifndef FEATURE_H
#define FEATURE_H

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

struct FeaturePoint{

    cv::KeyPoint keypoints;
    int id;
    int age;

};

struct FeatureSet{

    std::vector<cv::KeyPoint> keypoints;
    std::vector<int> ages;
    cv::Mat descriptors;
    int size(){
        return descriptors.rows;
    }
    void clear(){
        keypoints.clear();
        ages.clear();
        descriptors.release();
    }
    
};

void appendNewFeatures(cv::Mat& image, FeatureSet& current_features,cv::Rect &validRoi,cv::Ptr<cv::ORB> &orb_detector);

void appendNewFeatures(std::vector<cv::Point2f> points_new, FeatureSet& current_features,cv::Rect &validRoi);

void featureDetectionFast(cv::Mat image, std::vector<cv::Point2f>& points,cv::Rect &validRoi);

void featureDetectionORB(cv::Mat image, std::vector<cv::KeyPoint>& keypoints,cv::Rect &validRoi,cv::Ptr<cv::ORB> &orb_detector,cv::Mat & new_descriptor);

void bucketingFeatures(cv::Mat& image, FeatureSet& current_features, int bucket_size, int features_per_bucket);

void LRMatching(cv::Mat &img_l_0, cv::Mat &img_r_0, std::vector<cv::KeyPoint>& keypoints_l_0, std::vector<cv::KeyPoint>& keypoints_r_0,
                    FeatureSet& current_features,cv::Mat &F,cv::Rect & validRoi,cv::Ptr<cv::ORB> &orb_detector);

void LeftMatching(cv::Mat &imageLeft_t0,cv::Mat &imageLeft_t1,std::vector<cv::KeyPoint> & keypoints_l_0 ,
std::vector<cv::KeyPoint> & keypoints_l_1,std::vector<cv::KeyPoint> & keypoints_r_0,cv::Rect &validRoi,FeatureSet& current_features,cv::Ptr<cv::ORB> &orb_detector,cv::Mat & checker);

void LRChecker(std::vector<cv::KeyPoint>& keypoints_l_0, std::vector<cv::KeyPoint>& keypoints_r_0,cv::Mat & checker);

void deleteUnmatchFeaturesCircle(std::vector<cv::Point2f>& points0, std::vector<cv::Point2f>& points1,
                          std::vector<cv::Point2f>& points2, std::vector<cv::Point2f>& points3,
                          std::vector<cv::Point2f>& points0_return,
                          std::vector<uchar>& status0, std::vector<uchar>& status1,
                          std::vector<uchar>& status2, std::vector<uchar>& status3,
                          std::vector<int>& ages);

#endif