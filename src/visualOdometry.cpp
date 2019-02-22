#include "visualOdometry.hpp"



void matchingFeatures(cv::Mat& imageLeft_t0, cv::Mat& imageRight_t0,
                      cv::Mat& imageLeft_t1, cv::Mat& imageRight_t1, 
                      FeatureSet& currentVOFeatures,
                      std::vector<MapPoint> MapPoints,
                      std::vector<cv::KeyPoint>&  keypointsLeft_t0, 
                      std::vector<cv::KeyPoint>&  keypointsRight_t0, 
                      std::vector<cv::KeyPoint>&  keypointsLeft_t1, 
                      std::vector<cv::KeyPoint>&  keypointsRight_t1,
                      cv::Mat &F,cv::Rect &validRoi1,cv::Rect &validRoi2,
                      cv::Ptr<cv::ORB> &orb_detector)
{
    // ----------------------------
    // Feature detection using FAST
    // ---------------------------- 
    std::vector<cv::KeyPoint>  pointsLeftReturn_t0;   // feature points to check cicular mathcing validation


    if (currentVOFeatures.size() < 2000)
    {

        // append new features with old features
        appendNewFeatures(imageLeft_t0, currentVOFeatures, validRoi1,orb_detector);   
        std::cout << "Current feature set size: " << currentVOFeatures.keypoints.size() << std::endl;
    }

    // --------------------------------------------------------
    // Feature tracking using KLT tracker, bucketing and circular matching
    // --------------------------------------------------------
    int bucket_size = 50;
    int features_per_bucket = 4;
    cv::Mat checker;
    bucketingFeatures(imageLeft_t0, currentVOFeatures, bucket_size, features_per_bucket);   
    
    LRMatching(imageLeft_t0, imageRight_t0,keypointsLeft_t0, keypointsRight_t0,currentVOFeatures, F,validRoi2,orb_detector);

    LeftMatching(imageLeft_t0,imageLeft_t1,keypointsLeft_t0,keypointsLeft_t1,keypointsRight_t0, validRoi1,currentVOFeatures,orb_detector,checker);

    //LRMatching(imageLeft_t0, imageRight_t0,keypointsLeft_t0, keypointsRight_t0,currentVOFeatures, F,validRoi1,orb_detector);
    LRChecker(keypointsLeft_t0,keypointsRight_t0,checker);
    

    // std::vector<bool> status;
    // checkValidMatch(pointsLeft_t0, pointsLeftReturn_t0, status, 1.0);

    // removeInvalidPoints(pointsLeft_t0, status);
    // removeInvalidPoints(pointsLeft_t1, status);
    // removeInvalidPoints(pointsRight_t0, status);
    // removeInvalidPoints(pointsRight_t1, status);
    // removeInvalidAge(currentVOFeatures.ages,status);

    //std::vector<cv::KeyPoint> keypoint_l_0, keypoint_r_0;
    // cv::KeyPoint::convert(pointsLeft_t0,keypoint_l_0);
    // cv::KeyPoint::convert(pointsRight_t0,keypoint_r_0);
    // // cv::drawKeypoints(imageLeft_t0,keypoint_l_0,imageLeft_t0);
    // cv::drawKeypoints(imageRight_t0,keypoint_r_0,imageRight_t0);
    // cv::imshow("image",imageLeft_t0);
    // cv::imshow("image2",imageRight_t0);
  
    // cv::waitKey(1);

    // currentVOFeatures.keypoints = pointsLeft_t1;

} 

void checkValidMatch(std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& points_return, std::vector<bool>& status, int threshold)
{
    int offset;
    for (int i = 0; i < points.size(); i++)
    {
        offset = std::max(std::abs(points[i].x - points_return[i].x), std::abs(points[i].y - points_return[i].y));
        // std::cout << offset << ", ";

        if(offset > threshold)
        {
            status.push_back(false);
        }
        else
        {
            status.push_back(true);
        }
    }
}

void checkValidMatchEpipolar(std::vector<cv::Point2f> &tempPoint2f_pointsLeft_t0,std::vector<cv::Point2f> &tempPoint2f_pointsLeft_t1,
std::vector<cv::Point2f> &tempPoint2f_pointsRight_t0,std::vector<cv::Point2f> &tempPoint2f_pointsRight_t1, std::vector<bool> &status_epipolarmatch)
{
    
    int offset0,offset1;
    for(int i =0;i <tempPoint2f_pointsLeft_t0.size();i++)
    {
        offset0 = (std::abs(tempPoint2f_pointsLeft_t0[i].y-tempPoint2f_pointsRight_t0[i].y));
        offset1 = (std::abs(tempPoint2f_pointsLeft_t1[i].y-tempPoint2f_pointsRight_t1[i].y));

        if(0 ==offset0 && 0 ==offset1)
        {
            status_epipolarmatch.push_back(true);            
        }
        else
        {
            status_epipolarmatch.push_back(false);
        }
        
    }

}

void removeInvalidPoints(std::vector<cv::Point2f>& points, const std::vector<bool>& status)
{
    int index = 0;
    for (int i = 0; i < status.size(); i++)
    {
        if (status[i] == false)
        {
            points.erase(points.begin() + index);
        }
        else
        {
            index ++;
        }
    }
}
void removeInvalidAge(std::vector<int> &currentVOfeaturesage, const std::vector<bool>& status)
{
    int index = 0;
    for (int i = 0; i < status.size(); i++)
    {
        if (status[i] == false)
        {
            currentVOfeaturesage.erase(currentVOfeaturesage.begin() + index);
        }
        else
        {
            index ++;
        }
    }
}

void trackingFrame2Frame(cv::Mat& projMatrl, cv::Mat& projMatrr,
                         std::vector<cv::KeyPoint>&  keypointsLeft_t0,
                         std::vector<cv::KeyPoint>&  keypointsLeft_t1, 
                         cv::Mat& points3D_t0,
                         cv::Mat& rotation,
                         cv::Mat& translation)
{
    std::vector<cv::Point2f> pointsLeft_t0, pointsLeft_t1;
    
    cv::KeyPoint::convert(keypointsLeft_t0, pointsLeft_t0);
    cv::KeyPoint::convert(keypointsLeft_t1, pointsLeft_t1);

      // Calculate frame to frame transformation

      // -----------------------------------------------------------
      // Rotation(R) estimation using Nister's Five Points Algorithm
      // -----------------------------------------------------------
    //   std::cout<<"project left matrix: "<<std::endl<< projMatrl<<std::endl;
    //   std::cout<<"project left matrix: "<<std::endl<< std::fixed << std::setprecision(10) << projMatrl.at<double>(0,0)<<std::endl;
    float focal = projMatrl.at<double>(0, 0);
    cv::Point2d principle_point(projMatrl.at<double>(0, 2), projMatrl.at<double>(1, 2));

      //recovering the pose and the essential cv::matrix
    //   cv::Mat E, mask;
    //   cv::Mat translation_mono = cv::Mat::zeros(3, 1, CV_64F);
    //   E = cv::findEssentialMat(pointsLeft_t1, pointsLeft_t0, focal, principle_point, cv::RANSAC, 0.999, 1.0, mask);
    //   //R and t is from second to 1st camera, x1 = R*x2 +t,hence the orientation ,NOTE: TRANSLATION IS UNIT VECTOR 
    //   cv::recoverPose(E, pointsLeft_t1, pointsLeft_t0, rotation, translation_mono, focal, principle_point, mask);
    //   std::cout << "recoverPose rotation: " << rotation << std::endl;

    // ------------------------------------------------
    // Translation (t) estimation by use solvePnPRansac
    // ------------------------------------------------
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);  
    cv::Mat inliers;  
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat intrinsic_matrix = (cv::Mat_<float>(3, 3) << projMatrl.at<double>(0, 0), projMatrl.at<double>(0, 1), projMatrl.at<double>(0, 2),
                                                projMatrl.at<double>(1, 0), projMatrl.at<double>(1, 1), projMatrl.at<double>(1, 2),
                                                projMatrl.at<double>(2, 0), projMatrl.at<double>(2, 1), projMatrl.at<double>(2, 2)); // change from 1 to 2 jo
    
    std::cout<<"project left matrix: "<<std::endl<< intrinsic_matrix<<std::endl;
    int iterationsCount = 500;        // number of Ransac iterations.
    float reprojectionError = 2.0;    // maximum allowed distance to consider it an inlier.
    float confidence = 0.95;          // RANSAC successful confidence.
    bool useExtrinsicGuess = true;
    int flags =cv::SOLVEPNP_ITERATIVE;
    

    //points3D_t0 are world coordinate frame based on stereo triangulation, hence world and camera frame are the same  
    cv::solvePnPRansac( points3D_t0, pointsLeft_t1, intrinsic_matrix, distCoeffs, rvec, translation,      
                        useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                        inliers, flags );
    
    
    translation = -translation;
    std::cout << "inliers size: " << inliers.size() << std::endl; //number of matches between points3D_t0 to pointsLeft_t1- for translation
    rvec = -rvec;
    cv::Mat rvec_solvepnp;
    cv::Rodrigues(rvec, rvec_solvepnp);
    std::cout<<"translation: "<<translation <<std::endl;
    std::cout <<"rvec for pnpransac in euler angles"<<rvec <<std::endl;
    std::cout<< "rvec_solvepnp"<< rvec_solvepnp<<std::endl; 

    rotation = rvec_solvepnp;
    

    std::vector<cv::Point3f> vec_points3D_t0(points3D_t0.rows);
    for(int i=0;i<points3D_t0.rows;i++){
        
        vec_points3D_t0[i].x = points3D_t0.at<float>(i,0);
        vec_points3D_t0[i].y = points3D_t0.at<float>(i,1);
        vec_points3D_t0[i].z = points3D_t0.at<float>(i,2);
    }

    // bundle adjustment attempt
    //bundleAdjustment ( vec_points3D_t0, pointsLeft_t1, intrinsic_matrix, rotation, translation );
    

}

void bundleAdjustment (const std::vector<cv::Point3f> points_3d,const std::vector<cv::Point2f> points_2d,const cv::Mat& K,cv::Mat& R,cv::Mat& t){
    typedef g2o::BlockSolver_6_3 Block;
    std::unique_ptr<Block::LinearSolverType> linearSolver (new g2o::LinearSolverCSparse<Block::PoseMatrixType>());
    std::unique_ptr<Block> solver_ptr (new Block(std::move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    Eigen::Matrix3d R_mat;
    R_mat << R.at<double>(0,0), R.at<double>(0,1),R.at<double>(0,2),
             R.at<double>(1,0), R.at<double>(1,1),R.at<double>(1,2),
             R.at<double>(2,0), R.at<double>(2,1),R.at<double>(2,2);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(g2o::SE3Quat(R_mat,Eigen::Vector3d(t.at<double>(0,0),t.at<double>(1,0),t.at<double>(2,0))));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    //landmarks, should be more 3d points ,vertex->nodes
    int index = 1;
    for(const cv::Point3f p:points_3d){
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId(index++);
        point->setEstimate(Eigen::Vector3d(p.x,p.y,p.z));
        point->setMarginalized(true);
        optimizer.addVertex(point);

    }

    //camera
    g2o::CameraParameters* camera = new g2o::CameraParameters (
        K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0
    );
    camera->setId ( 0 );
    optimizer.addParameter ( camera );

    //edges
    index = 1;
    for(const cv::Point2f p:points_2d)
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(index);
        edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index)));
        edge->setVertex(1,vSE3);    //frame vertex
        edge->setMeasurement(Eigen::Vector2d(p.x,p.y));
        edge->setParameterId(0,0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }

    clock_t tic = clock();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    clock_t toc = clock();
    double time = (toc-tic)*CLOCKS_PER_SEC;
    std::cout<< "timing: "<< time<<std::endl;
    std::cout<<"bundled T ="<<std::endl<<Eigen::Isometry3d(vSE3->estimate()).matrix()<<std::endl;



}


void displayTracking(cv::Mat& imageLeft_t1, 
                     std::vector<cv::KeyPoint>&  keypointsLeft_t0,
                     std::vector<cv::KeyPoint>&  keypointsLeft_t1)
{
      // -----------------------------------------
      // Display feature racking
      // -----------------------------------------
      std::vector<cv::Point2f> pointsLeft_t0, pointsLeft_t1;
      cv::KeyPoint::convert(keypointsLeft_t0,pointsLeft_t0);
      cv::KeyPoint::convert(keypointsLeft_t1,pointsLeft_t1);
      int radius = 2;
      cv::Mat vis;

      cv::cvtColor(imageLeft_t1, vis, CV_GRAY2BGR,3);
      

      for (int i = 0; i < pointsLeft_t0.size(); i++)
      {
          cv::circle(vis, cvPoint(pointsLeft_t0[i].x, pointsLeft_t0[i].y), radius, CV_RGB(0,255,0));
      }

      for (int i = 0; i < pointsLeft_t1.size(); i++)
      {
          cv::circle(vis, cvPoint(pointsLeft_t1[i].x, pointsLeft_t1[i].y), radius, CV_RGB(255,0,0));
      }

      for (int i = 0; i < pointsLeft_t1.size(); i++)
      {
          cv::line(vis, pointsLeft_t0[i], pointsLeft_t1[i], CV_RGB(0,255,0));
      }

      cv::imshow("vis ", vis );  
      cv::waitKey(100);

}

void distinguishNewPoints(std::vector<cv::KeyPoint>&  newPoints, 
                          std::vector<bool>& valid,
                          std::vector<MapPoint>& mapPoints,
                          int frameId_t0,
                          cv::Mat& points3DFrame_t0,
                          cv::Mat& points3DFrame_t1,
                          cv::Mat& points3DWorld,
                          std::vector<cv::KeyPoint>&  currentPointsLeft_t0, 
                          std::vector<cv::KeyPoint>&  currentPointsLeft_t1, 
                          std::vector<FeaturePoint>&  currentFeaturePointsLeft,
                          std::vector<FeaturePoint>&  oldFeaturePointsLeft)
{
    // remove exist points, find new points
    // int idx = mapPoints.size();
    currentFeaturePointsLeft.clear();


    for (int i = 0; i < currentPointsLeft_t0.size() ; ++i)
    {
        bool exist = false;
        for (std::vector<FeaturePoint>::iterator oldPointIter = oldFeaturePointsLeft.begin() ; oldPointIter != oldFeaturePointsLeft.end(); ++oldPointIter)
        {
            if ((oldPointIter->keypoints.pt.x == currentPointsLeft_t0[i].pt.x) && (oldPointIter->keypoints.pt.y == currentPointsLeft_t0[i].pt.y)) //existing point check
            {
                exist = true;

                FeaturePoint featurePoint{.keypoints=currentPointsLeft_t1[i], .id=oldPointIter->id, .age=oldPointIter->age+1};
                currentFeaturePointsLeft.push_back(featurePoint);


                cv::Mat pointPoseIn_t1 = (cv::Mat_<float>(3, 1) << points3DFrame_t1.at<float>(i, 0), points3DFrame_t1.at<float>(i, 1), points3DFrame_t1.at<float>(i, 2));
                Observation obs;
                obs.frame_id = frameId_t0 + 1;
                obs.pointPoseInFrame = pointPoseIn_t1;

                mapPoints[oldPointIter->id].addObservation(obs); 
                // std::cout << "!!!!!!!!!!!!!!MapPoint  " << oldPointIter->id << " obs : " << mapPoints[oldPointIter->id].mObservations.size() << std::endl;

                break;
            }
        }
        if (!exist)
        {
            newPoints.push_back(currentPointsLeft_t1[i]);
            
            // add new points to currentFeaturePointsLeft
            int pointId = mapPoints.size();
            FeaturePoint featurePoint{.keypoints=currentPointsLeft_t1[i], .id=pointId, .age=1};
            currentFeaturePointsLeft.push_back(featurePoint);
            // idx ++;

            // add new points to map points
            cv::Mat worldPose = (cv::Mat_<float>(3, 1) << points3DWorld.at<float>(i, 0), points3DWorld.at<float>(i, 1), points3DWorld.at<float>(i, 2));

            MapPoint mapPoint(pointId, worldPose);

            // add observation from frame t0
            cv::Mat pointPoseIn_t0 = (cv::Mat_<float>(3, 1) << points3DFrame_t0.at<float>(i, 0), points3DFrame_t0.at<float>(i, 1), points3DFrame_t0.at<float>(i, 2));
            Observation obs;
            obs.frame_id = frameId_t0;
            obs.pointPoseInFrame = pointPoseIn_t0;
            mapPoint.addObservation(obs);

            // add observation from frame t1
            cv::Mat pointPoseIn_t1 = (cv::Mat_<float>(3, 1) << points3DFrame_t1.at<float>(i, 0), points3DFrame_t1.at<float>(i, 1), points3DFrame_t1.at<float>(i, 2));
            obs.frame_id = frameId_t0 +1 ;
            obs.pointPoseInFrame = pointPoseIn_t1;
            mapPoint.addObservation(obs);


            mapPoints.push_back(mapPoint);

        }
        valid.push_back(!exist);
    }

    // std::cout << "---------------------------------- "  << std::endl;
    // std::cout << "currentPointsLeft size : " << currentPointsLeft.size() << std::endl;
    // std::cout << "points3DFrame_t0 size : " << points3DFrame_t0.size() << std::endl;
    // std::cout << "points3DFrame_t1 size : " << points3DFrame_t1.size() << std::endl;
    // std::cout << "points3DWorld size : " << points3DWorld.size() << std::endl;



    // for (std::vector<cv::Point2f>::iterator currentPointIter = currentPointsLeft.begin() ; currentPointIter != currentPointsLeft.end(); ++currentPointIter)
    // {
    //     bool exist = false;
    //     for (std::vector<FeaturePoint>::iterator oldPointIter = oldFeaturePointsLeft.begin() ; oldPointIter != oldFeaturePointsLeft.end(); ++oldPointIter)
    //     {
    //         if ((oldPointIter->point.x == currentPointIter->x) && (oldPointIter->point.y == currentPointIter->y))
    //         {
    //            exist = true;

    //            FeaturePoint featurePoint{.point=*currentPointIter, .id=oldPointIter->id};
    //            currentFeaturePointsLeft.push_back(featurePoint);
    //            break;
    //         }
    //     }
    //     if (!exist)
    //     {
    //         newPoints.push_back(*currentPointIter);
            
    //         FeaturePoint featurePoint{.point=*currentPointIter, .id=idx};
    //         currentFeaturePointsLeft.push_back(featurePoint);
    //         idx ++;

    //     }
    //     valid.push_back(!exist);

    // }
    std::cout << "newPoints size : " << newPoints.size() << std::endl;
}
