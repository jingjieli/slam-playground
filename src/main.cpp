#include <iostream>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <math.h>
#include "opencv2/opencv.hpp"
#include "boost/filesystem.hpp"

using namespace std;

//#define PROCESS_EUROC_DATASET
//#define PROCESS_IPHONE_DATASET
#define PROCESS_ANDROID_DATASET

double readIphoneImgTimestamp(const std::string& file_path);

void readEurocTimestamp(const std::string& file_path);

void readAndroidTimestamp(const std::string& file_path);

void prepareDataset(const std::string& dataset_path);

void processDataset(const std::string& dataset_path);

std::map<unsigned int, double> iphone_timestamp_storage;
size_t iphone_img_counter = 0;
size_t iphone_img_num = 0;
std::vector<long> euroc_timestamp_storage;
std::vector<long> android_timestamp_storage;

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
    for (size_t i = 0; i < android_timestamp_storage.size(); ++i) {
        
        const std::string img_path(dataset_path + "/image/" + std::to_string(android_timestamp_storage[i]) + ".png");
        
        //std::cout << "Image path: " << img_path << "\n";
        
        cv::Mat input_frame = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);

        if (input_frame.cols == 0 || input_frame.rows == 0) continue;
        
        //std::cout << "Image w: " << input_frame.cols << " h: " << input_frame.rows << "\n";

        cv::imshow("Frame", input_frame);

        cv::waitKey(30);
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