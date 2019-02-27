#include "bucket.hpp"

Bucket::Bucket(int size){
    max_size = size;
    //cv::Mat features.descriptors(4,32,CV_8U);

}

int Bucket::size(){
    return features.keypoints.size();
}

void Bucket::add_feature(cv::KeyPoint point, int age,cv::Mat row_descriptor){
    //wont add feature with age>10;
    int age_threshold = 10;
    if (age<age_threshold){
        

        //insert any feature before bucket is full
        if (size()<max_size)
        {
            features.keypoints.push_back(point);
            features.ages.push_back(age);
            features.descriptors.push_back(row_descriptor);
            
              
        }
        // insert feature with old age and remove youngest ????????????????????????????????
        else
        {
            
            int age_min = features.ages[0];
            int age_min_idx = 0;

            for (int i =0;i <size();i++)
            {
                if (features.ages[i]<age_min)
                {
                    age_min = features.ages[i];
                    age_min_idx = i;
                }
            }
            if(age>age_min){
                features.keypoints[age_min_idx] = point;
                features.ages[age_min_idx] = age;
                features.descriptors.row(age_min_idx) = row_descriptor;
            }
            
        }
        

    }
    
    // std::cout<<"features.descriptor"<<row_descriptor<<std::endl;
    // std::cout<<"features.descriptor"<<features.descriptors<<std::endl;
}

void Bucket::get_features(FeatureSet& current_features){

    current_features.keypoints.insert(current_features.keypoints.end(), features.keypoints.begin(), features.keypoints.end());
    current_features.ages.insert(current_features.ages.end(), features.ages.begin(), features.ages.end());
    if(current_features.descriptors.empty()){
        if(features.descriptors.empty()){
            return;   
        }
        else{ 
            current_features.descriptors=features.descriptors;
            return;
        }
    }
    else if(features.descriptors.empty()){
        return;
    }
    else if(!features.descriptors.empty()){
        cv::vconcat(current_features.descriptors,features.descriptors,current_features.descriptors);
    }
  

}

Bucket::~Bucket(){
}