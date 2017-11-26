#include <iostream>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <math.h>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "boost/filesystem.hpp"

using namespace std;
using namespace cv;

//#define PROCESS_EUROC_DATASET
//#define PROCESS_IPHONE_DATASET
#define PROCESS_ANDROID_DATASET

double readIphoneImgTimestamp(const std::string& file_path);

void readEurocTimestamp(const std::string& file_path);

void readAndroidTimestamp(const std::string& file_path);

void prepareDataset(const std::string& dataset_path);

void processDataset(const std::string& dataset_path);

void trackWithOpticalFlow(cv::Mat& last_frame, cv::Mat& curr_frame, 
                size_t frame_idx, std::list<cv::Point2f>& keypoints);

void trackWithFeatureMatching(cv::Mat& frame, size_t frame_idx);

std::map<unsigned int, double> iphone_timestamp_storage;
size_t iphone_img_counter = 0;
size_t iphone_img_num = 0;
std::vector<long> euroc_timestamp_storage;
std::vector<long> android_timestamp_storage;

cv::Ptr<cv::ORB> orb_detector;
std::vector<KeyPoint> orb_keypoints_1, orb_keypoints_2;
cv::Mat orb_descriptors_1, orb_descriptors_2;
std::vector<cv::DMatch> orb_matches;
std::vector<cv::DMatch> good_matches;
cv::Ptr<BFMatcher> orb_matcher;

int main(int argc, char** argv) {

    std::cout << "Hello: from Slam Playground" << std::endl;

    std::string m_dataset_path = argv[1];

    cv::namedWindow("Frame", CV_WINDOW_AUTOSIZE);

    prepareDataset(m_dataset_path);

    processDataset(m_dataset_path);

    return 0;

}

void prepareDataset(const std::string& dataset_path) {

#ifdef PROCESS_EUROC_DATASET
    const std::string euroc_dataset_path = dataset_path + "/cam0/data.csv";
    readEurocTimestamp(euroc_dataset_path);
#endif 

#ifdef PROCESS_IPHONE_DATASET
    const std::string iphone_timestamps_dir(dataset_path + "/IMAGE_TIME/");
    iphone_img_num = countIphoneImgNumber(iphone_timestamps_dir);
#endif 

#ifdef PROCESS_ANDROID_DATASET
    const std::string android_dataset_path = dataset_path + "/image_timestamps.txt";
    readAndroidTimestamp(android_dataset_path);
#endif 

}

void processDataset(const std::string& dataset_path) {

#ifdef PROCESS_IPHONE_DATASET
    while (iphone_img_counter < iphone_img_num) {
        
        const std::string img_path(dataset_path + "/IMAGE/" + std::to_string(iphone_img_counter));

        // std::cout << "Image path: " << img_path << "\n";

        cv::Mat input_frame = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);

        if (input_frame.cols == 0 || input_frame.rows == 0) continue;

        // std::cout << "Image w: " << input_frame.cols << " h: " << input_frame.rows << "\n";

        iphone_img_counter++;

        cv::imshow("Frame", input_frame);

        cv::waitKey(30);

    }
#endif

#ifdef PROCESS_EUROC_DATASET  
    for (size_t i = 0; i < euroc_timestamp_storage.size(); ++i) {
        
        const std::string img_path(dataset_path + "/cam0/data/" + std::to_string(euroc_timestamp_storage[i]) + ".png");
        
        //std::cout << "Image path: " << img_path << "\n";
        
        cv::Mat input_frame = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);

        if (input_frame.cols == 0 || input_frame.rows == 0) continue;
        
        //std::cout << "Image w: " << input_frame.cols << " h: " << input_frame.rows << "\n";

        cv::imshow("Frame", input_frame);

        cv::waitKey(30);
    }
#endif 

#ifdef PROCESS_ANDROID_DATASET  
    std::list<cv::Point2f> keypoints;
    cv::Mat gray_frame, last_frame, img_matched;
    for (size_t i = 0; i < android_timestamp_storage.size(); ++i) {
        
        const std::string img_path(dataset_path + "/image/" + std::to_string(android_timestamp_storage[i]) + ".png");
        
        //std::cout << "Image path: " << img_path << "\n";
        
        cv::Mat input_frame = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);

        if (input_frame.cols == 0 || input_frame.rows == 0) continue;
        
        //std::cout << "Image w: " << input_frame.cols << " h: " << input_frame.rows << "\n";

        cv::cvtColor(input_frame, gray_frame, CV_RGBA2GRAY);
        // trackWithOpticalFlow(last_frame, gray_frame, i, keypoints);
        
        // for (auto kp : keypoints) {
        //     cv::circle(input_frame, kp, 3, cv::Scalar(0, 255, 0), 1);
        // }

        trackWithFeatureMatching(gray_frame, i);
        last_frame = input_frame.clone();
        
        if (i % 2 == 0)
        {
            if (i != 0) 
            {
                drawMatches(last_frame, orb_keypoints_2, input_frame, orb_keypoints_1, good_matches, img_matched);
                cv::imshow("Frame", img_matched);
            }
        }
        else 
        {
            drawMatches(last_frame, orb_keypoints_1, input_frame, orb_keypoints_2, good_matches, img_matched);
            cv::imshow("Frame", img_matched);
        }

        //cv::imshow("Frame", input_frame);

        cv::waitKey(1);
    }
#endif 

    std::cout << "imgLoop exits ...\n";
}

size_t countIphoneImgNumber(const std::string &dir_path) {

    std::cout << "countIphoneImgNumber starts ...\n";

    std::vector<boost::filesystem::directory_entry> files;

    if (boost::filesystem::is_directory(dir_path)) {
        std::copy(
            boost::filesystem::directory_iterator(dir_path),
            boost::filesystem::directory_iterator(),
            std::back_inserter(files)
        );    

        std::cout << dir_path << " is a directory containing:\n";

        for (std::vector<boost::filesystem::directory_entry>::const_iterator iter = files.begin(); 
                iter != files.end(); ++iter) {

            //std::cout <<(*iter).path().stem().string() << "\n";
            unsigned int img_idx = std::stoi((*iter).path().stem().string());
            const std::string timestamp_path = (*iter).path().string();
            double img_timestamp = readIphoneImgTimestamp(timestamp_path);

            iphone_timestamp_storage.insert(std::pair<int, double>(img_idx, img_timestamp));
        }
    }

    std::cout << "Total timestamps: " << iphone_timestamp_storage.size() << "\n";

    return files.size();
}


double readIphoneImgTimestamp(const std::string& file_path) {

    std::cout << "readIphoneImgTimestamp from: " << file_path << "\n";

    double timestamp = 0;

    std::ifstream in_stream(file_path, ios::in | ios::binary);

    if (in_stream.is_open()) {
        std::cout << "Timestamp file is opened..\n";

        char time_buf[8];

        in_stream.read(time_buf, 8);

        timestamp = *reinterpret_cast<double*>(time_buf);

        printf("Image timestamp: %f\n", timestamp);

        in_stream.close();
    }

    return timestamp;

}

void readEurocTimestamp(const std::string& file_path) {

    std::ifstream in_stream(file_path, ios::in);

    std::string curr_line;

    if (in_stream.is_open()) {

        getline(in_stream, curr_line);
        
        while (getline(in_stream, curr_line)) {

            //std::cout << curr_line << "\n";

            istringstream iss(curr_line);

            long timestamp_long;

            while (iss >> timestamp_long) {

                //std::cout << "Timestamp is: " << timestamp_long << "\n";

                euroc_timestamp_storage.push_back(timestamp_long);

                if (iss.peek() == ',') {
                    break;
                }

            }
        }

        in_stream.close();
    }

    printf("readEurocTimestamp found %lu images\n", euroc_timestamp_storage.size());

}

void readAndroidTimestamp(const std::string& file_path) {
    std::ifstream in_stream(file_path, ios::in);
    
        std::string curr_line;
    
        if (in_stream.is_open()) {
                
            while (getline(in_stream, curr_line)) {
    
                //std::cout << curr_line << "\n";
    
                istringstream iss(curr_line);
    
                long timestamp_long;
    
                while (iss >> timestamp_long) {
    
                    //std::cout << "Timestamp is: " << timestamp_long << "\n";
    
                    android_timestamp_storage.push_back(timestamp_long);
    
                    break;
    
                }
            }
    
            in_stream.close();
        }
    
        printf("readAndroidTimestamp found %lu images\n", android_timestamp_storage.size());
}

void trackWithOpticalFlow(cv::Mat& last_frame, cv::Mat& curr_frame, 
                size_t frame_idx, std::list<cv::Point2f>& keypoints) {
    double t1 = 0.0, t2 = 0.0;
    if (frame_idx % 3 == 0) 
    {
        t1 = cv::getTickCount();
        keypoints.clear();
        // std::vector<cv::KeyPoint> kpts;
        // cv::Ptr<cv::FastFeatureDetector> fast_detector = 
        //         cv::FastFeatureDetector::create(20, true, cv::FastFeatureDetector::TYPE_9_16);
        // fast_detector->detect(curr_frame, kpts);

        // for (KeyPoint kp : kpts) {
        //     keypoints.push_back(kp.pt);
        // }
        std::vector<cv::Point2f> kpts;
        cv::goodFeaturesToTrack(curr_frame, kpts, 70, 0.01, 30);
        for (auto kp : kpts) {
            keypoints.push_back(kp);
        }
        t2 = cv::getTickCount();
        std::cout << "[Frame " << frame_idx << "] Detect " << keypoints.size() << " feature points in " 
                << (t2 - t1) * 1000.0 / cv::getTickFrequency() << " ms" << std::endl;
    }
    else 
    {
        t1 = cv::getTickCount();
        std::vector<cv::Point2f> prev_keypoints;
        std::vector<cv::Point2f> curr_keypoints;

        for (cv::Point2f pt : keypoints) {
            prev_keypoints.push_back(pt);
        }

        std::vector<unsigned char> status;
        std::vector<float> error;

        cv::calcOpticalFlowPyrLK(last_frame, curr_frame, prev_keypoints, curr_keypoints, status, error);

        // update the maintained keypoints
        size_t i = 0;
        for (auto iter = keypoints.begin(); iter != keypoints.end(); ++i) {
            if (status[i] == 0) {
                iter = keypoints.erase(iter);
                continue;
            }
            *iter = curr_keypoints[i];
            iter++;
        }

        t2 = cv::getTickCount();
        if (keypoints.size() == 0) {
            std::cout << "All keypoints lost at Frame " << frame_idx << std::endl;
        } else {
            std::cout << "[Frame " << frame_idx << "] Track " << keypoints.size() << " feature points in " 
                    << (t2 - t1) * 1000.0 / cv::getTickFrequency() << " ms" << std::endl;
        }
    }

    last_frame = curr_frame.clone();
}

void trackWithFeatureMatching(cv::Mat& frame, size_t frame_idx) {
    double t1 = 0.0, t2 = 0.0;
    t1 = cv::getTickCount();
    if (!orb_detector) {
        orb_detector = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    }

    if (!orb_matcher) {
        orb_matcher = cv::BFMatcher::create(NORM_HAMMING, false);
    }

    orb_matches.clear();
    good_matches.clear();

    if (frame_idx % 2 == 0) 
    {
        orb_keypoints_1.clear();
        orb_detector->detect(frame, orb_keypoints_1);
        orb_detector->compute(frame, orb_keypoints_1, orb_descriptors_1);

        if (frame_idx != 0) 
        {
            orb_matcher->match(orb_descriptors_2, orb_descriptors_1, orb_matches);
        } 
        else 
        {
            return;
        }
    } 
    else {
        orb_keypoints_2.clear();
        orb_detector->detect(frame, orb_keypoints_2);
        orb_detector->compute(frame, orb_keypoints_2, orb_descriptors_2);
        orb_matcher->match(orb_descriptors_1, orb_descriptors_2, orb_matches);
    }

    double min_dist = 10000, max_dist = 0;

    for (size_t i = 0; i < orb_matches.size(); ++i) 
    {
        double dist = orb_matches[i].distance;
        if (dist > max_dist) max_dist = dist;
        if (dist < min_dist) min_dist = dist;
    }

    for (size_t i = 0; i < orb_descriptors_1.rows; ++i)
    {
        if (orb_matches[i].distance <= std::max(2 * min_dist, 15.0)) 
        {
            good_matches.push_back(orb_matches[i]);
        }
    }
    t2 = cv::getTickCount();
    std::cout << "Frame [" << frame_idx << "] Tracking uses " << (t2 - t1) * 1000.0 /cv::getTickFrequency()
             << " ms. Max dist: " << max_dist << " Min dist: " << min_dist 
             << " Matches: " << orb_matches.size() << " Good matches: " << good_matches.size() << std::endl;

}