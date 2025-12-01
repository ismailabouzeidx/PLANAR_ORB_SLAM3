/**
* Monocular-Inertial ORB-SLAM3 for INS-COI dataset
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<sstream>

#include<opencv2/core/core.hpp>

#include<System.h>
#include "ImuTypes.h"

using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps);
void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_inertial_inscoi path_to_vocabulary path_to_settings path_to_sequence" << endl;
        cerr << "Example: ./mono_inertial_inscoi Vocabulary/ORBvoc.txt Examples/Monocular-Inertial/ins_coi.yaml /path/to/C4_30" << endl;
        return 1;
    }

    // Load images (rgb.txt format: timestamp filepath)
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[3]) + "/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();
    cout << "Loaded " << nImages << " images" << endl;

    // Load IMU (imu.txt format: timestamp_ns,gx,gy,gz,ax,ay,az)
    vector<double> vTimestampsImu;
    vector<cv::Point3f> vAcc, vGyro;
    string strImuFile = string(argv[3]) + "/imu.txt";
    LoadIMU(strImuFile, vTimestampsImu, vAcc, vGyro);

    int nImu = vTimestampsImu.size();
    cout << "Loaded " << nImu << " IMU measurements" << endl;

    if(nImages <= 0 || nImu <= 0)
    {
        cerr << "ERROR: Failed to load images or IMU data" << endl;
        return 1;
    }

    // Find first IMU measurement that comes before first image
    int first_imu = 0;
    while(vTimestampsImu[first_imu] <= vTimestamps[0])
        first_imu++;
    first_imu--;

    // Create SLAM system
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images: " << nImages << ", IMU: " << nImu << endl << endl;

    cv::Mat im;
    vector<ORB_SLAM3::IMU::Point> vImuMeas;

    for(int ni = 0; ni < nImages; ni++)
    {
        // Read image
        im = cv::imread(string(argv[3]) + "/" + vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
        }

        // Collect IMU measurements between previous and current frame
        vImuMeas.clear();
        if(ni > 0)
        {
            while(vTimestampsImu[first_imu] <= vTimestamps[ni])
            {
                vImuMeas.push_back(ORB_SLAM3::IMU::Point(
                    vAcc[first_imu].x, vAcc[first_imu].y, vAcc[first_imu].z,
                    vGyro[first_imu].x, vGyro[first_imu].y, vGyro[first_imu].z,
                    vTimestampsImu[first_imu]));
                first_imu++;
            }
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass image and IMU to SLAM system
        SLAM.TrackMonocular(im, tframe, vImuMeas);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        // Wait to load next frame
        double T = 0;
        if(ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if(ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if(ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    // Shutdown
    SLAM.Shutdown();

    // Save trajectory
    SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");

    // Print timing stats
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for(int ni = 0; ni < nImages; ni++)
        totaltime += vTimesTrack[ni];

    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;

    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // Skip header lines (starting with #)
    string s;
    while(getline(f, s))
    {
        if(s.empty() || s[0] == '#')
            continue;

        stringstream ss(s);
        double t;
        string filename;
        ss >> t >> filename;
        vTimestamps.push_back(t);
        vstrImageFilenames.push_back(filename);
    }
}

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    ifstream fImu;
    fImu.open(strImuPath.c_str());

    while(!fImu.eof())
    {
        string s;
        getline(fImu, s);
        
        if(s.empty() || s[0] == '#')
            continue;

        // Parse CSV: timestamp_ns,gx,gy,gz,ax,ay,az
        double data[7];
        int count = 0;
        size_t pos = 0;
        string item;
        
        while((pos = s.find(',')) != string::npos && count < 6)
        {
            item = s.substr(0, pos);
            data[count++] = stod(item);
            s.erase(0, pos + 1);
        }
        data[6] = stod(s);

        vTimeStamps.push_back(data[0] / 1e9);  // ns to seconds
        vGyro.push_back(cv::Point3f(data[1], data[2], data[3]));
        vAcc.push_back(cv::Point3f(data[4], data[5], data[6]));
    }
}



