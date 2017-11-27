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

void estimateCameraPose(const std::vector<cv::KeyPoint>& prev_keypoints, 
                        const std::vector<cv::KeyPoint>& curr_keypoints, 
                        const std::vector<DMatch>& matches);

void validateCameraPose(const cv::Mat& rotation_matrix,
                        const cv::Mat& translation_matrix,
                        const std::vector<cv::KeyPoint>& prev_keypoints, 
                        const std::vector<cv::KeyPoint>& curr_keypoints, 
                        const std::vector<DMatch>& matches);

cv::Point2f pixel2camera(const cv::Point2f& pixel_point, const cv::Mat& camera_matrix);

void triangulation(const cv::Mat& rotation_matrix,
                    const cv::Mat& translation_matrix,
                    const std::vector<cv::KeyPoint>& prev_keypoints, 
                    const std::vector<cv::KeyPoint>& curr_keypoints, 
                    const std::vector<DMatch>& matches,
                    std::vector<cv::Point3d>& points_3d);

void validateTriangulation(const cv::Mat& rotation_matrix,
                            const cv::Mat& translation_matrix,
                            const std::vector<cv::KeyPoint>& prev_keypoints, 
                            const std::vector<cv::KeyPoint>& curr_keypoints, 
                            const std::vector<DMatch>& matches,
                            const std::vector<cv::Point3d>& points_3d);

std::map<unsigned int, double> iphone_timestamp_storage;
size_t iphone_img_counter = 0;
size_t iphone_img_num = 0;
std::vector<long> euroc_timestamp_storage;
std::vector<long> android_timestamp_storage;

// ********* Optical Flow ************
cv::Ptr<cv::FastFeatureDetector> fast_detector;
std::vector<cv::KeyPoint> fast_keypoints;
std::vector<cv::Point2f> good_features_keypoints;
std::vector<cv::Point2f> prev_optflw_keypoints;
std::vector<cv::Point2f> curr_optflw_keypoints;
std::vector<unsigned char> optflw_status;
std::vector<float> optflw_error;
std::list<cv::Point2f> tracked_keypoints;

// ********* Feature Matching ********
cv::Ptr<cv::ORB> orb_detector;
std::vector<KeyPoint> orb_keypoints_1, orb_keypoints_2;
cv::Mat orb_descriptors_1, orb_descriptors_2;
std::vector<cv::DMatch> orb_matches;
std::vector<cv::DMatch> good_matches;
cv::Ptr<DescriptorMatcher> orb_matcher;

// ********* Pose Estimation *********
cv::Mat camera_intrinsics, fundamental_matrix, essential_matrix, homography_matrix, R, t;
std::vector<cv::Point2f> matched_keypoints_1;
std::vector<cv::Point2f> matched_keypoints_2;
std::vector<cv::Point3d> world_points;

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
    cv::Mat gray_frame, last_frame, img_matched;
    for (size_t i = 0; i < android_timestamp_storage.size(); ++i) {
        
        const std::string img_path(dataset_path + "/image/" + std::to_string(android_timestamp_storage[i]) + ".png");
        
        //std::cout << "Image path: " << img_path << "\n";
        
        cv::Mat input_frame = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);

        if (input_frame.cols == 0 || input_frame.rows == 0) continue;
        
        //std::cout << "Image w: " << input_frame.cols << " h: " << input_frame.rows << "\n";

        cv::cvtColor(input_frame, gray_frame, CV_RGBA2GRAY);
        // trackWithOpticalFlow(last_frame, gray_frame, i, tracked_keypoints);
        
        // for (size_t i = 0; i < optflw_status.size(); ++i)
        // {
        //     if (optflw_status[i] != 0) 
        //     {
        //         cv::line(input_frame, prev_optflw_keypoints[i], curr_optflw_keypoints[i], cv::Scalar(0, 255, 0), 1);
        //     }
        // }

        // for (auto kp : tracked_keypoints) 
        // {
        //     cv::circle(input_frame, kp, 3, cv::Scalar(0, 255, 0), 1);
        // }

        // cv::imshow("Frame", input_frame);

        trackWithFeatureMatching(input_frame, i);
        last_frame = input_frame.clone();
        
        if (i % 2 == 0)
        {
            if (i != 0) 
            {
                estimateCameraPose(orb_keypoints_2, orb_keypoints_1, good_matches);
                validateCameraPose(R, t, orb_keypoints_2, orb_keypoints_1, good_matches);
                triangulation(R, t, orb_keypoints_2, orb_keypoints_1, good_matches, world_points);
                validateTriangulation(R, t, orb_keypoints_2, orb_keypoints_1, good_matches, world_points);
                drawMatches(last_frame, orb_keypoints_2, input_frame, orb_keypoints_1, good_matches, img_matched);
                cv::imshow("Frame", img_matched);
            }
        }
        else 
        {
            estimateCameraPose(orb_keypoints_1, orb_keypoints_2, good_matches);
            validateCameraPose(R, t, orb_keypoints_1, orb_keypoints_2, good_matches);
            triangulation(R, t, orb_keypoints_1, orb_keypoints_2, good_matches, world_points);
            validateTriangulation(R, t, orb_keypoints_1, orb_keypoints_2, good_matches, world_points);
            drawMatches(last_frame, orb_keypoints_1, input_frame, orb_keypoints_2, good_matches, img_matched);
            cv::imshow("Frame", img_matched);
        }

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
        // fast_keypoints.clear();
        // if (!fast_detector) 
        // {
        //     fast_detector = cv::FastFeatureDetector::create(20, true, cv::FastFeatureDetector::TYPE_9_16);
        // }
        // fast_detector->detect(curr_frame, fast_keypoints);

        // for (KeyPoint kp : fast_keypoints) {
        //     keypoints.push_back(kp.pt);
        // }
        good_features_keypoints.clear();
        cv::goodFeaturesToTrack(curr_frame, good_features_keypoints, 70, 0.01, 30);
        for (auto kp : good_features_keypoints) {
            keypoints.push_back(kp);
        }
        t2 = cv::getTickCount();
        std::cout << "[Frame " << frame_idx << "] Detect " << keypoints.size() << " feature points in " 
                << (t2 - t1) * 1000.0 / cv::getTickFrequency() << " ms" << std::endl;
    }
    else 
    {
        t1 = cv::getTickCount();

        prev_optflw_keypoints.clear();
        curr_optflw_keypoints.clear();

        for (cv::Point2f pt : keypoints) {
            prev_optflw_keypoints.push_back(pt);
        }

        optflw_status.clear();
        optflw_error.clear();

        cv::calcOpticalFlowPyrLK(last_frame, curr_frame, prev_optflw_keypoints, curr_optflw_keypoints, optflw_status, optflw_error);

        // update the maintained keypoints
        size_t i = 0;
        for (auto iter = keypoints.begin(); iter != keypoints.end(); ++i) {
            if (optflw_status[i] == 0) {
                iter = keypoints.erase(iter);
                continue;
            }
            *iter = curr_optflw_keypoints[i];
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
        orb_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
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
    else 
    {
        orb_keypoints_2.clear();
        orb_detector->detect(frame, orb_keypoints_2);
        orb_detector->compute(frame, orb_keypoints_2, orb_descriptors_2);
        orb_matcher->match(orb_descriptors_1, orb_descriptors_2, orb_matches);
    }

    double min_dist = 10000, max_dist = 0;

    for (size_t i = 0; i < orb_descriptors_1.rows; ++i) 
    {
        double dist = orb_matches[i].distance;
        if (dist > max_dist) max_dist = dist;
        if (dist < min_dist) min_dist = dist;
    }

    for (size_t i = 0; i < orb_descriptors_1.rows; ++i)
    {
        if (orb_matches[i].distance <= std::max(2 * min_dist, 30.0)) 
        {
            good_matches.push_back(orb_matches[i]);
        }
    }
    t2 = cv::getTickCount();
    std::cout << "Frame [" << frame_idx << "] Tracking uses " << (t2 - t1) * 1000.0 /cv::getTickFrequency()
             << " ms. Max dist: " << max_dist << " Min dist: " << min_dist 
             << " Matches: " << orb_matches.size() << " Good matches: " << good_matches.size() << std::endl;

}

void estimateCameraPose(const std::vector<cv::KeyPoint>& prev_keypoints, 
                        const std::vector<cv::KeyPoint>& curr_keypoints, 
                        const std::vector<DMatch>& matches) {
    
    if (camera_intrinsics.empty())
    {
        std::cout << "Setup camera intrinsics" << std::endl;
        camera_intrinsics = (cv::Mat_<double>(3, 3) << 639.073, 0.0, 320.0, 0.0, 639.073, 240.0, 0.0, 0.0, 1.0);
    }

    matched_keypoints_1.clear();
    matched_keypoints_2.clear();

    for (size_t i = 0; i < matches.size(); ++i)
    {
        matched_keypoints_1.push_back(prev_keypoints[matches[i].queryIdx].pt);
        matched_keypoints_2.push_back(curr_keypoints[matches[i].trainIdx].pt);
    }

    fundamental_matrix = cv::findFundamentalMat(matched_keypoints_1, matched_keypoints_2, CV_FM_8POINT);

    cv::Point2d principal_point(320.0, 240.0);
    //double focal_length = 639.073;
    int focal_length = 639;

    essential_matrix = cv::findEssentialMat(matched_keypoints_1, matched_keypoints_2, focal_length, principal_point, RANSAC);

    homography_matrix = cv::findHomography(matched_keypoints_1, matched_keypoints_2, RANSAC, 3, cv::noArray(), 2000, 0.99);

    cv::recoverPose(essential_matrix, matched_keypoints_1, matched_keypoints_2, R, t, focal_length, principal_point);

    std::cout << "---------- Pose Estimation ----------" << std::endl;

    std::cout << "fundamental_matrix: " << std::endl;
    std::cout << fundamental_matrix << std::endl << std::endl;

    std::cout << "essential_matrix: " << std::endl;
    std::cout << essential_matrix << std::endl << std::endl;

    std::cout << "homography_matrix: " << std::endl;
    std::cout << homography_matrix << std::endl << std::endl;

    std::cout << "R: " << std::endl;
    std::cout << R << std::endl;

    std::cout << "t: " << std::endl;
    std::cout << t << std::endl;

    std::cout << "--------------------------------------" << std::endl << std::endl;
}

cv::Point2f pixel2camera(const cv::Point2f& pixel_point, const cv::Mat& camera_matrix) {
    
    return cv::Point2f(
        (pixel_point.x - camera_matrix.at<double>(0, 2)) / camera_matrix.at<double>(0, 0),
        (pixel_point.y - camera_matrix.at<double>(1, 2)) / camera_matrix.at<double>(1, 1)
    );

}

void validateCameraPose(const cv::Mat& rotation_matrix,
                        const cv::Mat& translation_matrix,
                        const std::vector<cv::KeyPoint>& prev_keypoints, 
                        const std::vector<cv::KeyPoint>& curr_keypoints, 
                        const std::vector<DMatch>& matches) {
    
    cv::Mat t_x = (cv::Mat_<double>(3, 3) << 0.0, -translation_matrix.at<double>(2, 0), translation_matrix.at<double>(1, 0), 
                                             translation_matrix.at<double>(2, 0), 0.0, -translation_matrix.at<double>(0, 0),
                                             -translation_matrix.at<double>(1, 0), translation_matrix.at<double>(0, 0), 0.0);

    std::cout << "t_x^R: " << std::endl << t_x * rotation_matrix << std::endl << std::endl;

    for (DMatch match : matches)
    {
        cv::Point2f pt1 = pixel2camera(prev_keypoints[match.queryIdx].pt, camera_intrinsics);
        cv::Point2f pt2 = pixel2camera(curr_keypoints[match.trainIdx].pt, camera_intrinsics);
        
        cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);

        //std::cout << "epipolar constraint: " << y2.t() * t_x * rotation_matrix * y1 << std::endl;
    }

}

void triangulation(const cv::Mat& rotation_matrix,
                    const cv::Mat& translation_matrix,
                    const std::vector<cv::KeyPoint>& prev_keypoints, 
                    const std::vector<cv::KeyPoint>& curr_keypoints, 
                    const std::vector<DMatch>& matches,
                    std::vector<cv::Point3d>& points_3d) {

    cv::Mat T1 = (cv::Mat_<float>(3, 4) << 1.0, 0.0, 0.0, 0.0,
                                            0.0, 1.0, 0.0, 0.0,
                                            0.0, 0.0, 1.0, 0.0);

    cv::Mat T2 = (cv::Mat_<float>(3, 4) << 
        rotation_matrix.at<double>(0, 0), rotation_matrix.at<double>(0, 1), rotation_matrix.at<double>(0, 2), translation_matrix.at<double>(0, 0),
        rotation_matrix.at<double>(1, 0), rotation_matrix.at<double>(1, 1), rotation_matrix.at<double>(1, 2), translation_matrix.at<double>(1, 0),
        rotation_matrix.at<double>(2, 0), rotation_matrix.at<double>(2, 1), rotation_matrix.at<double>(2, 2), translation_matrix.at<double>(2, 0)   
    );

    std::vector<cv::Point2f> pts_1, pts_2;
    for (DMatch match : matches) 
    {
        pts_1.push_back ( pixel2camera( prev_keypoints[match.queryIdx].pt, camera_intrinsics) );
        pts_2.push_back ( pixel2camera( curr_keypoints[match.trainIdx].pt, camera_intrinsics) );
    }

    cv::Mat pts_4d;
    // std::cout << "T1: " << T1 << std::endl;
    // std::cout << "T2: " << T2 << std::endl;
    // std::cout << "pts_1: " << pts_1 << std::endl;
    // std::cout << "pts_2: " << pts_2 << std::endl;
    // std::cout << "pts_4d: " << pts_4d << std::endl;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d); // hmm... buggy. 

    points_3d.clear();

    for (size_t i = 0; i < pts_4d.cols; ++i)
    {
        cv::Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0); // was <double>, cause strange behavior in triangulatePoints return matrix... hmm... 
        cv::Point3d point_3d(
            x.at<float>(0, 0),
            x.at<float>(1, 0),
            x.at<float>(2, 0)
        );
        points_3d.push_back(point_3d);
        //std::cout << "p: " << point_3d << std::endl;
    }
}

void validateTriangulation(const cv::Mat& rotation_matrix,
                            const cv::Mat& translation_matrix,
                            const std::vector<cv::KeyPoint>& prev_keypoints, 
                            const std::vector<cv::KeyPoint>& curr_keypoints, 
                            const std::vector<DMatch>& matches,
                            const std::vector<cv::Point3d>& points_3d) {

    for (size_t i = 0; i < matches.size(); ++i)
    {
        cv::Point2f pt1_cam = pixel2camera(prev_keypoints[matches[i].queryIdx].pt, camera_intrinsics);

        cv::Point2d pt1_cam_3d(
            points_3d[i].x / points_3d[i].z,
            points_3d[i].y / points_3d[i].z
        );

        cv::Point2f pt2_cam = pixel2camera(curr_keypoints[matches[i].trainIdx].pt, camera_intrinsics);
        cv::Mat pt2_trans = rotation_matrix * (cv::Mat_<double>(3, 1) << points_3d[i].x, points_3d[i].y, points_3d[i].z) + translation_matrix;
        pt2_trans /= pt2_trans.at<double>(2, 0); 

        std::cout << "Point in first camera frame: " << pt1_cam << std::endl;
        std::cout << "Point reprojected from 3D: " << pt1_cam_3d << " d = " << points_3d[i].z << std::endl;

        std::cout << "Point in second camera frame: " << pt2_cam << std::endl;
        std::cout << "Point reprojected from second frame: " << pt2_trans.t() << std::endl << std::endl;
    }

}