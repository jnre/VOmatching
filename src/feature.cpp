#include "feature.hpp"
#include "bucket.hpp"

void appendNewFeatures(cv::Mat& image, FeatureSet& current_features,cv::Rect &validRoi,cv::Ptr<cv::ORB> &orb_detector)
{
    std::vector<cv::KeyPoint>  points_new;
    cv::Mat new_descriptors;
    featureDetectionORB(image,points_new, validRoi,orb_detector,new_descriptors);
    //featureDetectionFast(image, points_new, validRoi);
    current_features.keypoints.insert(current_features.keypoints.end(), points_new.begin(), points_new.end());
    std::vector<int>  ages_new(points_new.size(), 0);
    current_features.ages.insert(current_features.ages.end(), ages_new.begin(), ages_new.end());
    current_features.descriptors.push_back(new_descriptors);
}

void appendNewFeatures(std::vector<cv::KeyPoint> keypoints_new, FeatureSet& current_features,cv::Rect &validRoi)
{
    current_features.keypoints.insert(current_features.keypoints.end(), keypoints_new.begin(), keypoints_new.end());
    std::vector<int>  ages_new(keypoints_new.size(), 0);
    current_features.ages.insert(current_features.ages.end(), ages_new.begin(), ages_new.end());
}

void featureDetectionFast(cv::Mat image, std::vector<cv::Point2f>& points,cv::Rect &validRoi)  
{   
//uses FAST as for feature dection, modify parameters as necessary
  std::vector<cv::KeyPoint> keypoints;
  //int fast_threshold = 20;
  //bool nonmaxSuppression = true;
  cv::Mat mask = cv::Mat::zeros(image.size(),CV_8U);
  cv::Mat roi(mask,validRoi);
  roi = cv::Scalar(255);
  cv::Ptr<cv::FastFeatureDetector> fast_detector = cv::FastFeatureDetector::create(20,true);
  fast_detector->detect(image,keypoints,mask);
  std::sort(keypoints.begin(),keypoints.end(),[](cv::KeyPoint const &a, cv::KeyPoint const &b){
        return a.response >b.response;
    });
  //cv::FAST(image, keypoints, fast_threshold, nonmaxSuppression);  //gives int points
  cv::KeyPoint::convert(keypoints, points, std::vector<int>());
  
  
  cv::drawKeypoints(image,keypoints,image);
  cv::imshow("image123",image);
  cv::waitKey();
  
}

void featureDetectionORB(cv::Mat image, std::vector<cv::KeyPoint>& keypoints,cv::Rect &validRoi,cv::Ptr<cv::ORB> &orb_detector,cv::Mat & new_descriptor)  
{   

  // std::vector<cv::KeyPoint> keypoints;
  cv::Mat mask = cv::Mat::zeros(image.size(),CV_8U);
  cv::Mat roi(mask,validRoi);
  roi = cv::Scalar(255);
  orb_detector->detect(image,keypoints,mask);
  orb_detector->compute(image,keypoints,new_descriptor);
  // std::sort(keypoints.begin(),keypoints.end(),[](cv::KeyPoint const &a, cv::KeyPoint const &b){
  //       return a.response >b.response;
  //   });

  
}

void bucketingFeatures(cv::Mat& image, FeatureSet& current_features, int bucket_size, int features_per_bucket)
{
// This function buckets features
// image: only use for getting dimension of the image
// bucket_size: bucket size in pixel is bucket_size*bucket_size
// features_per_bucket: number of selected features per bucket
    int image_height = image.rows;
    int image_width = image.cols;
    int buckets_nums_height = image_height/bucket_size ;
    int buckets_nums_width = image_width/bucket_size ;
    //int buckets_number = buckets_nums_height * buckets_nums_width;

    std::vector<Bucket> Buckets;
  
    // initialize all the buckets(buckets_number + smaller boxes at edges)(200 buckets)
    for (int buckets_idx_height = 0; buckets_idx_height <= buckets_nums_height; buckets_idx_height++)
    {
      for (int buckets_idx_width = 0; buckets_idx_width <= buckets_nums_width; buckets_idx_width++)
      {
        Buckets.push_back(Bucket(features_per_bucket)); //bucket of size 4 initalized
      }
    }

    int buckets_number = Buckets.size();

    // bucket all current features into buckets by their location divide image to boxes and bucket_idx is the box number starting from 0, 192 buckets
    int buckets_nums_height_idx, buckets_nums_width_idx, buckets_idx;
    for (int i = 0; i < current_features.keypoints.size(); ++i)
    {
      buckets_nums_height_idx = current_features.keypoints[i].pt.y/bucket_size;
      buckets_nums_width_idx = current_features.keypoints[i].pt.x/bucket_size;
      buckets_idx = buckets_nums_height_idx* (buckets_nums_width +1) + buckets_nums_width_idx;
      Buckets[buckets_idx].add_feature(current_features.keypoints[i], current_features.ages[i],current_features.descriptors.row(i));
      
    }

    // get features back from buckets
    current_features.clear();
    for (int buckets_idx_height = 0; buckets_idx_height <= buckets_nums_height; buckets_idx_height++)
    {
      for (int buckets_idx_width = 0; buckets_idx_width <= buckets_nums_width; buckets_idx_width++)
      {
         buckets_idx = buckets_idx_height*(buckets_nums_width+1) + buckets_idx_width  ;
         Buckets[buckets_idx].get_features(current_features);
      }
    }

    std::cout << "current features number after bucketing: " << current_features.size() << std::endl;

}

void LRMatching(cv::Mat &img_l_0, cv::Mat &img_r_0,
                      std::vector<cv::KeyPoint>& keypoints_l_0, std::vector<cv::KeyPoint>& keypoints_r_0,
                      // std::vector<cv::Point2f>& keypoints_l_1, std::vector<cv::Point2f>& keypoints_r_1,
                      // std::vector<cv::Point2f>& keypoints_l_0_return,
                      FeatureSet& current_features,cv::Mat &F,cv::Rect & validRoi,cv::Ptr<cv::ORB> &orb_detector) { 
  
  //this function automatically gets rid of points for which tracking fails
  keypoints_l_0 = current_features.keypoints;
  for(int i =0; i <current_features.size();i++){ //this is actually for keypoints_l_1
    current_features.ages[i] += 1;   
  }
                   
  // cv::Size winSize=cv::Size(21,21);                                                                                             
  // cv::TermCriteria termcrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
  cv::Mat descriptors_l_0, descriptors_r_0;
  descriptors_l_0 = current_features.descriptors;
  
  //epipolar matching for feature between L and R----------------------------------------------------------
  //cv::Ptr<cv::FastFeatureDetector> fast_detector = cv::FastFeatureDetector::create(20,true);
  // orb_detector->compute(img_l_0,keypoints_l_0,descriptors_l_0); //lost keypoint and adds new points zzz
  //current_features.keypoints = keypoints_l_0;

  
  std::vector<cv::KeyPoint> matched_left_0,matched_right_0;
  std::vector<cv::DMatch> inline_matches;
  std::vector<bool> boolll(keypoints_l_0.size());
  
  for(int i =0; i< keypoints_l_0.size();i++){
    boolll[i] = false;
    cv::Mat mask = cv::Mat::zeros(img_r_0.size(),CV_8U);
    cv::Mat roi(mask,cv::Rect(0,keypoints_l_0[i].pt.y-1,keypoints_l_0[i].pt.x,2 ));// height of 1, on that line +1/-1, width is lesser than l_0.pt.x
    roi = cv::Scalar(255);
    orb_detector->detect(img_r_0,keypoints_r_0,mask);
    orb_detector->compute(img_r_0,keypoints_r_0,descriptors_r_0); 

    if(keypoints_r_0.size()>0){
      std::vector<cv::DMatch> matches;
      cv::Ptr<cv::BFMatcher> orb_matcher = cv::BFMatcher::create(cv::NORM_HAMMING,true);
      orb_matcher->match(descriptors_l_0.row(i),descriptors_r_0,matches);

      //if(matches[0].distance<30){
      //keypoint in first frame stored in matched 0, passing distance thres
        boolll[i] = true;
        int new_i = static_cast<int>(matched_left_0.size());
        matched_left_0.push_back(keypoints_l_0[i]);
        //keypoint in current frame stored in matched 1, passing distance thres
        matched_right_0.push_back(keypoints_r_0[matches[0].trainIdx]);
        inline_matches.push_back(cv::DMatch(new_i,new_i,matches[0].distance));
    }
  }

  
  int counter = 0;
  for(int i =0; i <keypoints_l_0.size();i++){
    
    if(boolll[i] == false){
      current_features.keypoints.erase(current_features.keypoints.begin() + counter);
      current_features.ages.erase(current_features.ages.begin() + counter);  
      cv::Mat top, bottom;

      
      // current_features.descriptors(cv::Range(i+1,row_range),cv::Range(0,32)).copyTo(current_features.descriptors);

        current_features.descriptors(cv::Range(0,counter),cv::Range(0,32)).copyTo(top);
        if(top.empty()){
          current_features.descriptors(cv::Range(counter+1,current_features.descriptors.rows),cv::Range(0,32)).copyTo(current_features.descriptors);
          continue;
        }
        else if(counter+1<current_features.size()){
          current_features.descriptors(cv::Range(counter+1,current_features.descriptors.rows),cv::Range(0,32)).copyTo(bottom);
          vconcat(top,bottom,current_features.descriptors);
          continue;
        }
        else{
          current_features.descriptors = top;
          break;
        }
      
    
    }
    else{
      counter++;
    }   
  }

  //ransac 
    std::vector<cv::DMatch> inline_matches_ransac;
    cv::Mat inline_mask, homography;
    std::vector<cv::KeyPoint> inline_left_0,inline_right_0;
    std::vector<cv::Point2f> converted_matched_old, converted_matched_new;
    cv::KeyPoint::convert(matched_left_0,converted_matched_old);
    cv::KeyPoint::convert(matched_right_0,converted_matched_new);
    //const double ransac_matching_thresh = 1.0f;
    //must have more than 4 points
    if(matched_left_0.size()>=4){
        homography = findHomography(converted_matched_old, converted_matched_new,cv::RANSAC,3.0,inline_mask);
    }
    else
    {
        throw "too little points"; 
    // vector<KeyPoint> emptykeypoints;
    // cout << "lost transformation" <<endl;
    
    }  
    
    for(int i =0; i <matched_left_0.size();i++){
        //see if mask gives a value of one, where one means it is correct matched
        if(inline_mask.at<uchar>(i)){
            int new_i = static_cast<int>(inline_left_0.size());
            inline_left_0.push_back(matched_left_0[i]);
            inline_right_0.push_back(matched_right_0[i]);
            //store matches of correct RANSAC
            inline_matches_ransac.push_back(cv::DMatch(new_i,new_i,0));
        }
    }

  cv::Mat heha;
  cv::drawMatches(img_l_0,inline_left_0,img_r_0,inline_right_0,inline_matches_ransac,
  heha,cv::Scalar::all(-1),cv::Scalar::all(-1),std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  //cv::imshow("draw",heha);
  //cv::waitKey(1);

  
  int counter2 = 0;
  for(int i =0; i<matched_left_0.size();i++){
    
    if(!inline_mask.at<uchar>(i)){
      current_features.keypoints.erase(current_features.keypoints.begin() + counter2);
      current_features.ages.erase(current_features.ages.begin() + counter2);
      cv::Mat top, bottom;

      current_features.descriptors(cv::Range(0,counter2),cv::Range(0,32)).copyTo(top);
      if(top.empty()){
        current_features.descriptors(cv::Range(counter2+1,current_features.descriptors.rows),cv::Range(0,32)).copyTo(current_features.descriptors);
        continue;
      }
      else if(counter2+1<current_features.size()){
        current_features.descriptors(cv::Range(counter2+1,current_features.descriptors.rows),cv::Range(0,32)).copyTo(bottom);
        vconcat(top,bottom,current_features.descriptors);
        continue;
      }
      else{
        current_features.descriptors = top;
        break;
      }

    }
    else{
      counter2++;      
    }
  }
  keypoints_r_0 = inline_right_0;
  keypoints_l_0 = inline_left_0;


  // cv::Mat res;
  // cv::drawMatches(img_l_0,matched_left_0,img_r_0,matched_right_0,inline_matches,res,cv::Scalar(255,0,0),cv::Scalar(255,0,0),std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  
  //cv::waitKey();



  // cv::calcOpticalFlowPyrLK(img_l_0, img_r_0, points_l_0, points_r_0, status0, err, winSize, 3, termcrit, 0, 0.001);
  // cv::calcOpticalFlowPyrLK(img_r_0, img_r_1, points_r_0, points_r_1, status1, err, winSize, 3, termcrit, 0, 0.001);
  // cv::calcOpticalFlowPyrLK(img_r_1, img_l_1, points_r_1, points_l_1, status2, err, winSize, 3, termcrit, 0, 0.001);
  // cv::calcOpticalFlowPyrLK(img_l_1, img_l_0, points_l_1, points_l_0_return, status3, err, winSize, 3, termcrit, 0, 0.001);
  
  
  // deleteUnmatchFeaturesCircle(points_l_0, points_r_0, points_r_1, points_l_1, points_l_0_return,
                        // status0, status1, status2, status3, current_features.ages);

  // std::cout << "points : " << points_l_0.size() << " "<< points_r_0.size() << " "<< points_r_1.size() << " "<< points_l_1.size() << " "<<std::endl;
}

void LeftMatching(cv::Mat &imageLeft_t0,cv::Mat &imageLeft_t1,std::vector<cv::KeyPoint> & keypoints_l_0 ,
std::vector<cv::KeyPoint> & keypoints_l_1,std::vector<cv::KeyPoint> & keypoints_r_0,cv::Rect & validRoi,FeatureSet& current_features
,cv::Ptr<cv::ORB> &orb_detector,cv::Mat& checker){

  cv::Mat descriptors_l_0, descriptors_l_1;
  std::vector<cv::KeyPoint> temp_keypoints_r_0;
  descriptors_l_0 = current_features.descriptors;


  std::vector<cv::KeyPoint> matched_left_0,matched_left_1;
  std::vector<cv::DMatch> matches;
  cv::Ptr<cv::BFMatcher> orb_matcher = cv::BFMatcher::create(cv::NORM_HAMMING,true);
  featureDetectionORB(imageLeft_t1,keypoints_l_1, validRoi,orb_detector,descriptors_l_1);
  // orb_detector->compute(imageLeft_t1,keypoints_l_1,descriptors_l_1);
  orb_matcher->match(descriptors_l_0,descriptors_l_1,matches);
  

  FeatureSet temp_current_features;
  //no sort
  for(int i = 0;i<matches.size();i++){

        matched_left_0.push_back(keypoints_l_0[matches[i].queryIdx]);
        matched_left_1.push_back(keypoints_l_1[matches[i].trainIdx]);
        temp_current_features.keypoints.push_back(keypoints_l_1[matches[i].trainIdx]);
        temp_current_features.ages.push_back(current_features.ages[matches[i].queryIdx]); //takes ages position for t_0
        temp_current_features.descriptors.push_back(descriptors_l_1.row(matches[i].trainIdx));
        temp_keypoints_r_0.push_back(keypoints_r_0[matches[i].queryIdx]);
  }
  cv::Mat hoho;
  cv::drawMatches(imageLeft_t0,keypoints_l_0,imageLeft_t1,keypoints_l_1,matches,hoho);
  //cv::imshow("hoho",hoho);
  //cv::waitKey(1);
  keypoints_r_0=temp_keypoints_r_0;
  std::vector<cv::DMatch> inline_matches;


  cv::Mat inline_mask, homography;
  std::vector<cv::KeyPoint> RANSAC_l_0,RANSAC_l_1;
  // std::vector<cv::Point2f> converted_matched_old, converted_matched_new;
  std::vector<cv::Point2f> converted_matched_l_0, converted_matched_l_1;
  // KeyPoint::convert(matched_old,converted_matched_old);
  // KeyPoint::convert(matched_newest,converted_matched_new);
  cv::KeyPoint::convert(matched_left_0,converted_matched_l_0);
  cv::KeyPoint::convert(matched_left_1,converted_matched_l_1);
  if(matched_left_0.size()>=4){
      homography = findHomography(converted_matched_l_0, converted_matched_l_1,cv::RANSAC,3,inline_mask);
  }
  else
  {
      throw "too little points"; 
  
  }
  //FeatureSet temp2_current_features;
  for(int i =0; i <matched_left_0.size();i++){
    //see if mask gives a value of one, where one means it is correct matched
    if(inline_mask.at<uchar>(i)){
        int new_i = static_cast<int>(RANSAC_l_0.size());
        RANSAC_l_0.push_back(matched_left_0[i]);
        RANSAC_l_1.push_back(matched_left_1[i]);
        //store matches of correct RANSAC
        inline_matches.push_back(cv::DMatch(new_i,new_i,0));

    }
  }

  checker = inline_mask;
  cv::Mat hehe;
  cv::drawMatches(imageLeft_t0,RANSAC_l_0,imageLeft_t1,RANSAC_l_1,inline_matches,hehe,
  cv::Scalar(255,0,0),cv::Scalar(255,0,0),std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  //cv::imshow("ransac 0 and 1",hehe);
  //cv::waitKey(1);

  int counter2 = 0;
  for(int i =0; i<matched_left_0.size();i++){
    
    if(!inline_mask.at<uchar>(i)){
      temp_current_features.keypoints.erase(temp_current_features.keypoints.begin() + counter2);
      temp_current_features.ages.erase(temp_current_features.ages.begin() + counter2);

      cv::Mat top, bottom;

      temp_current_features.descriptors(cv::Range(0,counter2),cv::Range(0,32)).copyTo(top);
      if(top.empty()){
        temp_current_features.descriptors(cv::Range(counter2+1,temp_current_features.descriptors.rows),cv::Range(0,32)).copyTo(temp_current_features.descriptors);
        continue;
      }
      else if(counter2+1<temp_current_features.size()){
        temp_current_features.descriptors(cv::Range(counter2+1,temp_current_features.descriptors.rows),cv::Range(0,32)).copyTo(bottom);
        vconcat(top,bottom,temp_current_features.descriptors);
        continue;
      }
      else{
        temp_current_features.descriptors = top;
        break;
      }



    }
    else{
      counter2++;      
    }
  }

  keypoints_l_0=RANSAC_l_0;
  keypoints_l_1=RANSAC_l_1;
  current_features.clear();
  current_features.keypoints= temp_current_features.keypoints;
  current_features.ages= temp_current_features.ages;
  current_features.descriptors= temp_current_features.descriptors;
  //current features is now for t1.
  

}

void LRChecker(std::vector<cv::KeyPoint>& keypoints_l_0, std::vector<cv::KeyPoint>& keypoints_r_0,cv::Mat & checker){
      
    int counter = 0;  
    for(int i = 0; i < checker.rows;i++){
      if(!checker.at<uchar>(i)){
      keypoints_r_0.erase(keypoints_r_0.begin() + counter);
      }       
      else{
        counter++;
      }

    }  
        
}

void deleteUnmatchFeaturesCircle(std::vector<cv::Point2f>& points0, std::vector<cv::Point2f>& points1,
                          std::vector<cv::Point2f>& points2, std::vector<cv::Point2f>& points3,
                          std::vector<cv::Point2f>& points0_return,
                          std::vector<uchar>& status0, std::vector<uchar>& status1,
                          std::vector<uchar>& status2, std::vector<uchar>& status3,
                          std::vector<int>& ages){
  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  for (int i = 0; i < ages.size(); ++i)
  {
     ages[i] += 1;
  }

  int indexCorrection = 0;
  for( int i=0; i<status3.size(); i++)  //size of all status is the same of size of currentVOfeatures-pointsLeft_t0
     {  cv::Point2f pt0 = points0.at(i- indexCorrection);
        cv::Point2f pt1 = points1.at(i- indexCorrection);
        cv::Point2f pt2 = points2.at(i- indexCorrection);
        cv::Point2f pt3 = points3.at(i- indexCorrection);
        cv::Point2f pt0_r = points0_return.at(i- indexCorrection);
        
        if ((status3.at(i) == 0)||(pt3.x<0)||(pt3.y<0)||
            (status2.at(i) == 0)||(pt2.x<0)||(pt2.y<0)||
            (status1.at(i) == 0)||(pt1.x<0)||(pt1.y<0)||
            (status0.at(i) == 0)||(pt0.x<0)||(pt0.y<0))   
        {
          // if((pt0.x<0)||(pt0.y<0)||(pt1.x<0)||(pt1.y<0)||(pt2.x<0)||(pt2.y<0)||(pt3.x<0)||(pt3.y<0))    
          // {
          //   status3.at(i) = 0;
          // }
          points0.erase (points0.begin() + (i - indexCorrection));
          points1.erase (points1.begin() + (i - indexCorrection));
          points2.erase (points2.begin() + (i - indexCorrection));
          points3.erase (points3.begin() + (i - indexCorrection));
          points0_return.erase (points0_return.begin() + (i - indexCorrection));

          ages.erase (ages.begin() + (i - indexCorrection));
          indexCorrection++;
        }

     }  
}