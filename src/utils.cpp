#include "utils.hpp"

void loadImageLeft(cv::Mat & image,cv::Mat & no_color,int frame_id, std::string filepath){
    char file[200];
    sprintf(file,"image_0/%06d.png", frame_id);
    std::string filename = filepath + std::string(file);
    // std::cout<<"filename: "<< filename <<std::endl;
    image = cv::imread(filename);
    // cv::imshow("left0",no_color);
    // cv::waitKey();
    cv::cvtColor(image,no_color,cv::COLOR_BGR2GRAY);
}

void loadImageRight(cv::Mat & image,cv::Mat & no_color,int frame_id, std::string filepath){
    char file[200];
    sprintf(file,"image_1/%06d.png", frame_id);
    std::string filename = filepath + std::string(file);
    image = cv::imread(filename);
    cv::cvtColor(image,no_color,cv::COLOR_BGR2GRAY);
}

void loadImageLeft(cv::VideoCapture & cap0,cv::Mat & image,cv::Mat & no_color,int frame_id){
    
    cap0 >> image;
    cv::cvtColor(image,no_color,cv::COLOR_BGR2GRAY);
    
    // int64 t = cvGetTickCount();
    // cv::waitKey(1);
    // t = cv::getTickCount() -t;
    // std::cout<< "time elapsed: "<<t*1000/cv::getTickFrequency() << "ms" <<std::endl;
}

void loadImageRight(cv::VideoCapture & cap1,cv::Mat & image,cv::Mat & no_color,int frame_id){

    cap1 >> image; 
    cv::cvtColor(image,no_color,cv::COLOR_BGR2GRAY);
    
}

void mapPointsToPointCloudsAppend(std::vector<MapPoint>& mapPoints,  PointCloud::Ptr cloud)
{
    // append only valid points
    size_t mapSize = mapPoints.size();
    size_t start = cloud->size();
    for (size_t i = start; i < mapSize; ++i)
    {
            PointT point;
            point.x = mapPoints[i].mWorldPos.at<float>(0);
            point.y = mapPoints[i].mWorldPos.at<float>(1);
            point.z = mapPoints[i].mWorldPos.at<float>(2);
            cloud->points.push_back(point);
    }
}

void simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  viewer->setBackgroundColor (0, 0, 0);
  viewer->removePointCloud ("sample cloud");

  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->spinOnce();
  // viewer->spin();
}

//x,y,z rotation
cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R)
{
 
    assert(isRotationMatrix(R));
     
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
 
    bool singular = sy < 1e-6; // If
 
    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);
     
}

bool isRotationMatrix(cv::Mat &R)
{
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
     
    return  norm(I, shouldBeIdentity) < 1e-6;
     
}

void integrateOdometryStereo(int frame_i, cv::Mat& rigid_body_transformation, cv::Mat& frame_pose, const cv::Mat& rotation, const cv::Mat& translation_stereo)
{

    // std::cout << "rotation" << rotation << std::endl;
    // std::cout << "translation_stereo" << translation_stereo << std::endl;

    
    cv::Mat addup = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);

    cv::hconcat(rotation, translation_stereo, rigid_body_transformation);
    cv::vconcat(rigid_body_transformation, addup, rigid_body_transformation);

    // std::cout << "rigid_body_transformation" << rigid_body_transformation << std::endl;

    double scale = sqrt((translation_stereo.at<double>(0))*(translation_stereo.at<double>(0)) 
                        + (translation_stereo.at<double>(1))*(translation_stereo.at<double>(1))
                        + (translation_stereo.at<double>(2))*(translation_stereo.at<double>(2))) ;

    // frame_pose = frame_pose * rigid_body_transformation;
    std::cout << "scale: " << scale << std::endl;

    // rigid_body_transformation = rigid_body_transformation.inv();
    // if ((scale>0.1)&&(translation_stereo.at<double>(2) > translation_stereo.at<double>(0)) && (translation_stereo.at<double>(2) > translation_stereo.at<double>(1))) 
    if (scale > 1. && scale < 10) 
    {
      // std::cout << "Rpose" << Rpose << std::endl;

      frame_pose = frame_pose * rigid_body_transformation;

    }
    else 
    {
     std::cout << "[WARNING] scale <0.1 too big scale or incorrect translation" << std::endl;
    }
}

    
// void display(int frame_id, cv::Mat& trajectory, cv::Mat& pose, std::vector<Matrix>& pose_matrix_gt, float fps, bool show_gt)
void display(int frame_id, cv::Mat& trajectory, cv::Mat& pose, float fps, bool show_gt)
{
    // draw estimated trajectory 
    int x = 600 - int(pose.at<double>(0))  ;
    int y = 400 - int(pose.at<double>(2)) ; //this is actually z
    circle(trajectory, cv::Point(x, y) ,1, CV_RGB(255,0,0), 2);

    // if (show_gt)
    // {
    //   // draw ground truth trajectory 
    //   cv::Mat pose_gt = cv::Mat::zeros(1, 3, CV_64F);
      
    //   pose_gt.at<double>(0) = pose_matrix_gt[frame_id].val[0][3];
    //   pose_gt.at<double>(1) = pose_matrix_gt[frame_id].val[0][7];
    //   pose_gt.at<double>(2) = pose_matrix_gt[frame_id].val[0][11];
    //   x = int(pose_gt.at<double>(0)) + 300;
    //   y = int(pose_gt.at<double>(2)) + 100;
    //   circle(trajectory, cv::Point(x, y) ,1, CV_RGB(255,255,0), 2);
    // }
    // print info

    // rectangle( traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
    // sprintf(text, "FPS: %02f", fps);
    // putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

    char text[200];
    sprintf(text, "x:%5.2f  y:%5.2f   z:%5.2f ",pose.at<double>(0),pose.at<double>(1),pose.at<double>(2));
    cv::putText(trajectory,text,cv::Point(50,50),CV_FONT_HERSHEY_COMPLEX,1,cv::Scalar::all(255),2,8);
    cv::imshow( "Trajectory", trajectory );
    


    cv::waitKey(100);
    trajectory.release();
    trajectory = cv::Mat::zeros(600,1200,CV_8UC3);
}

void pclDisplay(const cv::Mat &triangulated1, const cv::Mat &triangulated0, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1,pcl::PointCloud<pcl::PointXYZ>::Ptr cloud0,boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer){
    //boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    //pcl::visualization::PCLVisualizer viewer("Viewer");
    viewer->setBackgroundColor (255,255,255);

    
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>);
    cloud1->width = triangulated1.rows;
    cloud1->height =1;
    cloud1->is_dense = false;
    cloud1->points.resize(triangulated1.rows);

    for(int i = 0; i <triangulated1.rows; i++)
    {
        pcl::PointXYZ &point = cloud1->points[i];
        point.x = triangulated1.at<float>(i,0);
        point.y = -triangulated1.at<float>(i,1);
        point.z = -triangulated1.at<float>(i,2);
        //point.r = 0;
        //point.g = 0;
        //point.b = 255;
        std::cout<<"pointcloud blue: " <<cloud1->points[i] <<std::endl;
    }
    std::cout<<std::endl;
    
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud0(new pcl::PointCloud<pcl::PointXYZ>);
    cloud0->width = triangulated0.rows;
    cloud0->height =1;
    cloud0->is_dense = false;
    cloud0->points.resize(triangulated0.rows);

    for(int i = 0; i <triangulated0.rows; i++)
    {
        pcl::PointXYZ &point = cloud0->points[i];
        point.x = triangulated0.at<float>(i,0);
        point.y = -triangulated0.at<float>(i,1);
        point.z = -triangulated0.at<float>(i,2);
        //point.r = 0;
        //point.g = 0;
        //point.b = 255;
        std::cout<<"pointcloud red: " <<cloud0->points[i] <<std::endl;
    }

    //blue
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color1(cloud1, 0, 0, 255);
    viewer->addPointCloud(cloud1,single_color1,"Triangulated Point Cloud1");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,5,"Triangulated Point Cloud1");
    //red
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color0(cloud0, 255, 0, 0);
    viewer->addPointCloud(cloud0,single_color0,"Triangulated Point Cloud0");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,5,"Triangulated Point Cloud0");
    viewer->addCoordinateSystem(300.0);

    // E0gen::Vector4f centroid1;
    // pcl::computeCentroid(cloud1,centroid1);
    // cout<<"centroid of cloud1"<<centroid1<<endl;
    // pcl::CentroidPoint<pcl::PointXYZ> centroid0;
    // pcl::computeCentroid(cloud0,centroid0);
    // cout<<"centroid of cloud0"<<centroid0<<endl;
    viewer->spinOnce();


//     while (!viewer->wasStopped())
//    {
//         viewer.spinOnce(100);
//    }
    // char save_cloud[100];
    // sprintf(save_cloud,"./newpointsmove/test1_%02d.pcd",cloud_count);
    // pcl::io::savePCDFileASCII(save_cloud,*cloud);
    // cloud_count++;
    // return cloud;
}

void getOdometryMatch(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud0,const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1){
    pcl::IterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ> icp;
    icp.setInputSource(cloud0);
    icp.setInputTarget(cloud1);
    icp.setMaximumIterations(300);
    icp.setTransformationEpsilon(1e-9);
    icp.setMaxCorrespondenceDistance(150);
    icp.setEuclideanFitnessEpsilon(1);
    icp.setRANSACOutlierRejectionThreshold(1.5);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);    
    std::cout<<"hasconverged:" << icp.hasConverged() << std::endl;
    std::cout<<"score: " << icp.getFitnessScore() << std::endl;
    //cout<< "transform between" <<g_count2 <<"and"<< g_count2+1 <<endl;
    std::cout<< icp.getFinalTransformation() << std::endl;        

}

