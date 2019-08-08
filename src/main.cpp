#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/fast_math.hpp"

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
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <opencv2/core/eigen.hpp>

using namespace std;

#include "feature.hpp"
#include "utils.hpp"
//#include "evaluate_odometry.h"
#include "visualOdometry.hpp"
//#include "Frame.h"
#include "MapPoint.hpp"


int main(int argc, char** argv)
{
    if(argc <2)
    {
        // /home/joseph/SOFT_FEATURES/myparam/   path: /home/joseph/SOFT_FEATURES/mypoints/ /LIVE
        cerr<<" use ./run  path_to_calibration:/home/joseph/SOFT_FEATURES/kitti00.yaml [optional]path_to sequence: /home/joseph/dataset/sequences/00/ KITTI/LIVE /home/joseph/dataset/poses/00.txt"<<endl;
        return 1;
    }
    bool display_ground_truth = false; // from actual poses
    string strSettingPath;
    string filepath;
    string mode;
    string posepath;
    std::vector<cv::Point3d> pose_estimator_data;
    

    //just open the camera regardless
    cv::VideoCapture cap0(0);
    cv::VideoCapture cap1(1);
    if(argc ==5){
        display_ground_truth = true;
        filepath = string(argv[2]);
        cout <<"Filepath:"<<filepath <<endl;
        strSettingPath = string(argv[1]);
        cout << "Calibration Filepath: " <<strSettingPath <<endl;
        posepath = string(argv[4]);
        std::ifstream openseseme;
        openseseme.open(posepath);
        std::string line;
        mode = string(argv[3]);
        double var_file;
        cv::Point3d var_xyz;
        while(getline(openseseme,line)){
            istringstream dat(line);
            int counterish =0; 
            while(dat>>var_file){
                if(counterish ==3){
                    var_xyz.x = var_file;
                }
                if(counterish ==7){
                    var_xyz.y = var_file;
                }
                if(counterish ==11){
                    var_xyz.z = var_file;
                }
                counterish++;
            }
            pose_estimator_data.push_back(var_xyz);
        }
        //std::cout<< pose_estimator_data<< std::endl;
    }
    if(argc==4){
        filepath = string(argv[2]);
        cout <<"Filepath:"<<filepath <<endl;
        strSettingPath = string(argv[1]);
        cout << "Calibration Filepath: " <<strSettingPath <<endl;
        mode = string(argv[3]);
    }
    if(argc ==3){
    strSettingPath = string(argv[1]);
    cout << "Calibration Filepath: " <<strSettingPath <<endl;
    mode = string(argv[2]);
    }
    


    cv::Mat proj_left_matrix;
    cv::Mat proj_right_matrix;
    cv::Mat M1,M2,D1,D2,F,R,R1,R2,P1,P2,Q;
    cv::Rect validRoi1,validRoi2;

    //----------------------------------camera calib--------------------
    
    if(mode == "KITTI"){
        cv::FileStorage fSettings(strSettingPath,cv::FileStorage::READ);
        float fx,fy,cx,cy,bf;
        fSettings["Camera.fx"]>>fx;
        fSettings["Camera.fy"]>>fy;
        fSettings["Camera.cx"]>>cx;
        fSettings["Camera.cy"]>>cy;
        fSettings["Camera.bf"]>>bf;
        proj_left_matrix = (cv::Mat_<float> (3,4)<< fx,0.,cx,0.,0.,fy,cy,0.,0.,0.,1.,0.);
        proj_right_matrix = (cv::Mat_<float> (3,4)<< fx,0.,cx,bf,0.,fy,cy,0.,0.,0.,1.,0.);
        
    }

    if(mode == "LIVE"){
        cv::FileStorage fSettings(strSettingPath +"intrinsics.yml", cv::FileStorage::READ);

        fSettings["M1"]>> M1;
        fSettings["M2"]>> M2;
        fSettings["D1"]>> D1;
        fSettings["D2"]>> D2;
        fSettings["F"]>>F;
        
        fSettings.open(strSettingPath+"extrinsics.yml", cv::FileStorage::READ);  
        fSettings["R"]>>R;
        //fs["T"]>>T;
        fSettings["R1"]>>R1;
        fSettings["R2"]>>R2;
        fSettings["P1"]>>proj_left_matrix;
        fSettings["P2"]>>proj_right_matrix;
        fSettings["Q"]>>Q;
        fSettings["validRoi1"]>>validRoi1;
        fSettings["validRoi2"]>>validRoi2;
        fSettings.release();
    
    }

    std::cout<<"project left matrix: "<<std::endl<<proj_left_matrix<<std::endl;
    std::cout<<"project right matrix: "<<std::endl<<proj_right_matrix<<std::endl;
    //--------------------------------variables---------------------
    cv::Mat rotation = cv::Mat::eye(3,3,CV_64F);
    cv::Mat translation_stereo = cv::Mat::zeros(3,1,CV_64F);
    cv::Vec3d rotation_euler;
    

    cv::Ptr<cv::ORB> orb_detector = cv::ORB::create();
    orb_detector->setMaxFeatures(2000);
    //orb_detector->setNLevels(1);

    //need pose?
    cv::Mat pose = cv::Mat::zeros(3,1,CV_64F);
    cv::Mat Rpose = cv::Mat::eye(3,3,CV_64F);

    cv::Mat frame_pose = cv::Mat::eye(4,4,CV_64F);
    cv::Mat frame_pose32 = cv::Mat::eye(4,4,CV_32F);

    cv::Mat trajectory = cv::Mat::zeros(600,1200,CV_8UC3);
    //pcl::visualization::PCLVisualizer *visualizer;

    FeatureSet currentVOFeatures;
    

    PointCloud::Ptr features_cloud_ptr (new PointCloud);

    cv::Mat points4D, points3D;

    int init_frame_id = 0;

    cv::Mat imageLeft_t0_Color, imageLeft_t0;
    cv::Mat imageRight_t0_Color, imageRight_t0;
    //----first images

    if(argc >3){
        loadImageLeft(imageLeft_t0_Color,imageLeft_t0,init_frame_id,filepath);
        loadImageRight(imageRight_t0_Color,imageRight_t0,init_frame_id,filepath);
    }
    else{
        loadImageLeft(cap0,imageLeft_t0_Color,imageLeft_t0,init_frame_id);
        loadImageRight(cap1,imageRight_t0_Color,imageRight_t0,init_frame_id);
    
    }
    float fps;
    
    //rectification of image + undistort
    cv::Mat rmap[2][2];
    if(mode =="LIVE"){
    
        cv::initUndistortRectifyMap(M1,D1,R1,proj_left_matrix,imageLeft_t0.size(),CV_32FC1,rmap[0][0],rmap[0][1]);
        cv::initUndistortRectifyMap(M2,D2,R2,proj_right_matrix,imageLeft_t0.size(),CV_32FC1,rmap[1][0],rmap[1][1]);
        cv::remap(imageLeft_t0,imageLeft_t0,rmap[0][0],rmap[0][1],cv::INTER_LINEAR);
        cv::remap(imageRight_t0,imageRight_t0,rmap[1][0],rmap[1][1],cv::INTER_LINEAR);
    }

    if(mode =="KITTI"){
        validRoi1 = cv::Rect(0,0,imageLeft_t0.cols,imageLeft_t0.rows);
        validRoi2 = cv::Rect(0,0,imageLeft_t0.cols,imageLeft_t0.rows);
    }
    //----------VO

    clock_t tic = clock();

    std::vector<MapPoint> mapPoints;

    std::vector<FeaturePoint> oldFeaturePointsLeft;
    std::vector<FeaturePoint> currentFeaturePointsLeft;
    
    //boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    //boost::shared_ptr<> synonymous to <>::Ptr, a smart pointer

    //-----looppy

    for(int frame_id = init_frame_id+1;frame_id <9000;frame_id++){

        std::cout<< std::endl <<"frame_id"<<frame_id <<std::endl;
        //load images from frame 1 not 0
        cv::Mat imageLeft_t1_Color, imageLeft_t1;        
        cv::Mat imageRight_t1_Color, imageRight_t1;
        if(argc >3){
            loadImageLeft(imageLeft_t1_Color,imageLeft_t1,frame_id,filepath);
            loadImageRight(imageRight_t1_Color,imageRight_t1,frame_id,filepath);
        }
        if(argc ==3){
            loadImageLeft(cap0,imageLeft_t1_Color,imageLeft_t1,frame_id);
            loadImageRight(cap1,imageRight_t1_Color,imageRight_t1,frame_id);
        }

        if(mode =="LIVE"){
            cv::remap(imageLeft_t1,imageLeft_t1,rmap[0][0],rmap[0][1],cv::INTER_LINEAR);
            cv::remap(imageRight_t1,imageRight_t1,rmap[1][0],rmap[1][1],cv::INTER_LINEAR);
        }

        //-- create sgbm matcher for disparty( use to get world points by x Q) ------------------------------------------------

        // cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0,16,3);
        // int sgbmWinSize = 3;
        // sgbm->setPreFilterCap(63);
        // int cn = imageLeft_t0.channels();
        // sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
        // sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
        // sgbm->setUniquenessRatio(10);
        // sgbm->setSpeckleRange(32);
        // sgbm->setSpeckleWindowSize(100);
        // sgbm->setDisp12MaxDiff(1);
        // sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
        // cv::Mat disp0, disp1;
        // int64 t = cvGetTickCount();
        // sgbm->compute(imageLeft_t0,imageRight_t0,disp0);
        // sgbm->compute(imageLeft_t1,imageRight_t1,disp1);
        // t = cv::getTickCount() -t;
        // std::cout<< "time elapsed: "<<t*1000/cv::getTickFrequency() << "ms" <<std::endl;
        // ---------------------------------------------------------------------------------------------------------------------


        //currentVOFeatures  here was from previous loop pointsLeft_t1, which in next iteration is pointsLeft_t0
        //newer frames in each loop are given index 1, whereas the older frame get transfered from 1 to 0.
        std::vector<cv::KeyPoint> oldPointsLeft_t0= currentVOFeatures.keypoints;
        
        std::vector<cv::KeyPoint> keypointsLeft_t0, keypointsRight_t0, keypointsLeft_t1, keypointsRight_t1;
        
        matchingFeatures(imageLeft_t0, imageRight_t0,imageLeft_t1,imageRight_t1,currentVOFeatures,mapPoints,keypointsLeft_t0,
        keypointsRight_t0,keypointsLeft_t1,keypointsRight_t1,F,validRoi1,validRoi2,orb_detector);

        //currentVOFeatures now store data for t1.

        //currentVOFeatures.points = pointsLeft_t1
        imageLeft_t0 = imageLeft_t1;
        imageRight_t0 = imageRight_t1;

        std::vector<cv::KeyPoint> & currentPointsLeft_t0 = keypointsLeft_t0;
        std::vector<cv::KeyPoint> &currentPointsLeft_t1 = keypointsLeft_t1;         //pointsLeft_t1 after optical matching of pointsLeft_t0 circular matching

        std::cout<< "oldPointsLeft_t0 size: "<<oldPointsLeft_t0.size()<<std::endl;  //taken from previous loop of currentVOFeatures, without circular matching
        std::cout<<"currentFramePointsLeft size: "<<currentPointsLeft_t0.size() <<std::endl; //after matching with bucket and circular

        std::vector<cv::Point2f> newPoints;
        std::vector<bool> valid;

        // **********************************************************************************************************************************
        // ONLY FOR TRIANGULATION PURPOSES ARE THE VARIABLES CHANGED TO INT, DONT CHANGE THE ACTUAL POINTS BEACAUSE THEY ARE STILL CORRECT
        // **********************************************************************************************************************************

        //undistort points for triangualtePoints---------------------------------------------- might be better to change to another variable
        // std::vector<cv::Point2f> un_pointsLeft_t0, un_pointsLeft_t1,un_pointsRight_t0, un_pointsRight_t1;
        // cv::undistortPoints(pointsLeft_t0, un_pointsLeft_t0,M1,D1,R1,P1);
        // cv::undistortPoints(pointsLeft_t1, un_pointsLeft_t1,M1,D1,R1,P1);
        // cv::undistortPoints(pointsRight_t0, un_pointsRight_t0,M2,D2,R2,P2);
        // cv::undistortPoints(pointsRight_t1, un_pointsRight_t1,M2,D2,R2,P2);
        //cvRoundPoints--------------------------------------------------------------------------
        // std::vector<cv::Point2i> tempvec_pointsLeft_t0;
        // std::vector<cv::Point2i> tempvec_pointsLeft_t1;
        // std::vector<cv::Point2i> tempvec_pointsRight_t0;
        // std::vector<cv::Point2i> tempvec_pointsRight_t1;
        // // temp_pointsLeft_t0 = pointsLeft_t0;
        // // std::vector<cv::Point>(cvRound(pointsLeft_t0.x),cvRound(pointsLeft_t0.y));
        // for(int i=0;i <pointsLeft_t0.size();i++){                                                       //cvRound to pixel value integer
        //     cv::Point2i temp_pointsLeft_t0(cvRound(pointsLeft_t0[i].x), cvRound(pointsLeft_t0[i].y));
        //     cv::Point2i temp_pointsLeft_t1(cvRound(pointsLeft_t1[i].x), cvRound(pointsLeft_t1[i].y));
        //     cv::Point2i temp_pointsRight_t0(cvRound(pointsRight_t0[i].x), cvRound(pointsRight_t0[i].y));
        //     cv::Point2i temp_pointsRight_t1(cvRound(pointsRight_t1[i].x), cvRound(pointsRight_t1[i].y));
        //     tempvec_pointsLeft_t0.push_back(temp_pointsLeft_t0);
        //     tempvec_pointsLeft_t1.push_back(temp_pointsLeft_t1);
        //     tempvec_pointsRight_t0.push_back(temp_pointsRight_t0);
        //     tempvec_pointsRight_t1.push_back(temp_pointsRight_t1);
        // }
        // std::vector<cv::Point2f> tempPoint2f_pointsLeft_t0;                                             //triangulatePoints need Point2f
        // convert2<cv::Point2i,cv::Point2f>(tempvec_pointsLeft_t0,tempPoint2f_pointsLeft_t0);
        // std::vector<cv::Point2f> tempPoint2f_pointsLeft_t1;
        // convert2<cv::Point2i,cv::Point2f>(tempvec_pointsLeft_t1,tempPoint2f_pointsLeft_t1);
        // std::vector<cv::Point2f> tempPoint2f_pointsRight_t0;
        // convert2<cv::Point2i,cv::Point2f>(tempvec_pointsRight_t0,tempPoint2f_pointsRight_t0);
        // std::vector<cv::Point2f> tempPoint2f_pointsRight_t1;
        // convert2<cv::Point2i,cv::Point2f>(tempvec_pointsRight_t1,tempPoint2f_pointsRight_t1);   
        // //check epipolar line by constant y value------------------------------------------------
        // std::vector<bool> status_epipolarmatch;
        // checkValidMatchEpipolar(tempPoint2f_pointsLeft_t0,tempPoint2f_pointsLeft_t1,tempPoint2f_pointsRight_t0,tempPoint2f_pointsRight_t1,status_epipolarmatch);
        // removeInvalidPoints(tempPoint2f_pointsLeft_t0,status_epipolarmatch);
        // removeInvalidPoints(tempPoint2f_pointsLeft_t1,status_epipolarmatch);
        // removeInvalidPoints(tempPoint2f_pointsRight_t0,status_epipolarmatch);
        // removeInvalidPoints(tempPoint2f_pointsRight_t1,status_epipolarmatch);


        //sgbm -------------------------------------------
        // cv::Mat image3d0, image3d1;
        // //cv::reprojectImageTo3D(disp0,image3d0,Q);
        // //cv::reprojectImageTo3D(disp1,image3d1,Q);
        // //image3d0 = image3d0.reshape(1);
        // //image3d1 = image3d1.reshape(1);
        // cv::Mat new_image3d0;
        // new_image3d0 = cv::Mat(3,tempPoint2f_pointsLeft_t0.size(),CV_32F);
        // for(int i =0; i < tempPoint2f_pointsLeft_t0.size();i++){

        //     short d = disp0.at<short>(tempPoint2f_pointsLeft_t0[i].y,tempPoint2f_pointsLeft_t0[i].x);
        //     float disparity_d = d/16.0f;
        //     double from[4] = {
        //         static_cast<double>(tempPoint2f_pointsLeft_t1[i].x),
        //         static_cast<double>(tempPoint2f_pointsLeft_t1[i].y),
        //         static_cast<double>(d),
        //         1.0,

        //     };
        //     cv::Mat_<double> res = Q * cv::Mat_<double>(4,1,from);
        //     res /=res(3,0);
        //     new_image3d0.at<double>(0,i)=res.at<double>(0,0);
        //     new_image3d0.at<double>(1,i)=res.at<double>(1,0);
        //     new_image3d0.at<double>(2,i)=res.at<double>(2,0); 

          
        // }
        // std::cout<<"disp at"<< disp0.at<int>(tempvec_pointsLeft_t0[0].y,tempvec_pointsLeft_t0[0].y)<<std::endl;
        // std::cout<<new_image3d0 <<std::endl;
        // new_image3d0= new_image3d0.reshape(1);
        

        //triangulate 3d points-------------------------------------------------------------------

        // cv::Mat points3D_t0;
        // cv::Mat points4D_t0(1,pointsLeft_t0.size(),CV_64FC4);
        // cv::triangulatePoints(proj_left_matrix,proj_right_matrix,tempPoint2f_pointsLeft_t0,tempPoint2f_pointsRight_t0,points4D_t0);
        // cv::convertPointsFromHomogeneous(points4D_t0.t(),points3D_t0); // divides by last term to get 3D
        // points3D_t0 = points3D_t0.reshape(1);// adjust for channels

        // cv::Mat points3D_t1;
        // cv::Mat points4D_t1(1,pointsLeft_t1.size(),CV_64FC4);
        // cv::triangulatePoints(proj_left_matrix,proj_right_matrix,tempPoint2f_pointsLeft_t1,tempPoint2f_pointsRight_t1,points4D_t1);
        // cv::convertPointsFromHomogeneous(points4D_t1.t(),points3D_t1); // divides by last term to get 3D
        // points3D_t1 = points3D_t1.reshape(1); //adjust for channels


        // PointCloud::Ptr cloud0 (new PointCloud);
        // PointCloud::Ptr cloud1 (new PointCloud);
        // pclDisplay(points3D_t1,points3D_t0,cloud1,cloud0,viewer);
        // getOdometryMatch(cloud0,cloud1);
        // *****************************************************************************************************************************
        cv::Mat points3D_t0_L;
        cv::Mat points4D_t0_L(1,keypointsLeft_t0.size(),CV_64FC4);
        std::vector<cv::Point2f> pointsLeft_t0,pointsRight_t0, pointsLeft_t1,pointsRight_t1;
        cv::KeyPoint::convert(keypointsLeft_t0,pointsLeft_t0);
        cv::KeyPoint::convert(keypointsLeft_t1,pointsLeft_t1);
        cv::KeyPoint::convert(keypointsRight_t0,pointsRight_t0);

        cv::Mat points3D_t0_R = points3D_t0_L.clone();

        //cv::KeyPoint::convert(keypointsRight_t1,pointsRight_t1);

        cv::triangulatePoints(proj_left_matrix,proj_right_matrix,pointsLeft_t0,pointsRight_t0,points4D_t0_L);
        cv::convertPointsFromHomogeneous(points4D_t0_L.t(),points3D_t0_L); // divides by last term to get 3D
        points3D_t0_L = points3D_t0_L.reshape(1);// adjust for channel
        //std::cout<<" points4D_t0"<< points4D_t0 <<std::endl;    //there are weird points bettwen pointLeft_t0 and pointsright_to due to lack  of check between epipolar

        // cv::Mat points3D_t1;
        // cv::Mat points4D_t1(1,keypointsLeft_t1.size(),CV_64FC4);
        // cv::triangulatePoints(proj_left_matrix,proj_right_matrix,pointsLeft_t1,pointsRight_t1,points4D_t1);
        // cv::convertPointsFromHomogeneous(points4D_t1.t(),points3D_t1); // divides by last term to get 3D
        // points3D_t1 = points3D_t1.reshape(1); //adjust for channels

        //checking wrt to right camera frame -------------------------------------------------------------------------------------------
        cv::Mat P2toP1;
        // if(mode =="KITTI"){
        //     for(int i =0; i <points3D_t0_L.rows;i++){
        //         points3D_t0_R.at<float>(i,0) = points3D_t0_L.at<float>(i,0) +proj_right_matrix.at<float>(0,3); 

        //     }
        //     // P2toP1 = (cv::Mat_<float>(3, 4) << 1, 0, 0, -proj_right_matrix.at<float>(0, 3),
        //     //                                     0,1,0,0,
        //     //                                     0,0,1,0);
        // }
        // if(mode =="LIVE"){

        //     for(int i =0; i <points3D_t0_L.rows;i++){
        //         points3D_t0_R.at<float>(i,0) = points3D_t0_L.at<float>(i,0) +proj_right_matrix.at<double>(0,3); 

        //     }
            
        // }

        

        // 3D to 2D correspondence method----------------------------------------------------------------------------------------------
        trackingFrame2Frame(proj_left_matrix,proj_right_matrix, keypointsLeft_t0, keypointsLeft_t1,points3D_t0_L, rotation,translation_stereo,rotation_euler, mode);
        displayTracking(imageLeft_t1,keypointsLeft_t0,keypointsLeft_t1);

        //update world mat of keypoints wrt global coordinate frame------------------------------------------------------------------
        points4D = points4D_t0_L;
        frame_pose.convertTo(frame_pose32,CV_32F);
        points4D = frame_pose32 * points4D;
        cv::convertPointsFromHomogeneous(points4D.t(),points3D);

        //append new points to map point-------------------------------------------------------------
        // distinguishNewPoints(newPoints, valid, mapPoints, frame_id-1, //handles stationary img
        //                      points3D_t0, points3D_t1, points3D, 
        //             currentPointsLeft_t0, currentPointsLeft_t1, currentFeaturePointsLeft, oldFeaturePointsLeft);
        // oldFeaturePointsLeft= currentFeaturePointsLeft;
        // std::cout<< "mapPoints size: " <<mapPoints.size() <<std::endl;

        // //append feature points to point clouds-------------------------------------------------------
        // mapPointsToPointCloudsAppend(mapPoints, features_cloud_ptr);
        // std::cout << std::endl << "featureSetToPointClouds size: " << features_cloud_ptr->size() << std::endl;
        // simpleVis(features_cloud_ptr, viewer);

        //integrating and display-------------------------------------------------------------------
        //cv::Vec3f rotation_euler123 = rotationMatrixToEulerAngles(rotation);
        std::cout<<"rotation:" <<rotation_euler <<std::endl;
        std::cout<<"translation: "<<translation_stereo.t()<<std::endl;

        cv::Mat rigid_body_transformation;

        if(abs(rotation_euler[1])<10 && abs(rotation_euler[0])<10 && abs(rotation_euler[2])<10)
         {
            integrateOdometryStereo(frame_id, rigid_body_transformation, frame_pose, rotation, translation_stereo);

        }
         else{
             std::cout<< "Too large rotation"<<std::endl;
        }

        //std::cout <<"rigid_body_transformation" <<rigid_body_transformation<<std::endl;

        std::cout <<"frame_pose"<<frame_pose<<std::endl;

        Rpose = frame_pose(cv::Range(0,3), cv::Range(0,3)); // column 0 to 3 exclude 3, row 0 to 3, exlude 3, your 3 by 3 matrix 
        //cv::Vec3f Rpose_euler = rotationMatrixToEulerAngles(Rpose);
        //std:: cout<<"Rpose_euler"<<Rpose_euler <<std::endl;

        cv::Mat pose = frame_pose.col(3).clone(); // last column translation matrix for pose

        clock_t toc = clock();
        fps = float(frame_id-init_frame_id)/(toc-tic)*CLOCKS_PER_SEC;

        std::cout<<"Pose"<<pose.t() <<std::endl;    //position wrt to global frame
        std::cout<<"FPS: "<<fps<<std::endl;

        // display(frame_id, trajectory, pose, pose_matrix_gt, fps, display_ground_truth);
        display(frame_id, trajectory, pose, fps, display_ground_truth,rotation_euler,translation_stereo,pose_estimator_data);
        //std::cout<< "VO descriptors" << currentVOFeatures.descriptors <<std::endl;
    }

    return 0;
};





