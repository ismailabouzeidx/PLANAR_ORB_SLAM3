/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include"Atlas.h"
#include"MapPoint.h"
#include"KeyFrame.h"
#include "Settings.h"
#include<pangolin/pangolin.h>
#include<Eigen/Dense>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<mutex>
#include<string>

// Forward declaration
namespace ORB_SLAM3 {
    class BEVGenerator;
}

namespace ORB_SLAM3
{

class Settings;

class MapDrawer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MapDrawer(Atlas* pAtlas, const string &strSettingPath, Settings* settings);
    ~MapDrawer();

    void newParameterLoader(Settings* settings);

    Atlas* mpAtlas;

    void DrawMapPoints();
    void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph, const bool bDrawOptLba);
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
    void SetCurrentCameraPose(const Sophus::SE3f &Tcw);
    void SetReferenceKeyFrame(KeyFrame *pKF);
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw);
    void EstimateGroundPlane();
    void InvalidateGroundPlane(); // Invalidate ground plane when map changes
    bool HasGroundPlane() const;
    bool HasAlignment() const;
    Eigen::Matrix3f GetAlignRotation() const;
    Eigen::Vector3f GetAlignTranslation() const;
    
    // Get aligned camera pose (world->camera) for a KeyFrame
    // Returns: (R_cw, t_cw) where R_cw is rotation and t_cw is translation
    // R_cw: aligned world -> camera rotation (3x3)
    // t_cw: origin of aligned world expressed in camera frame (3x1)
    void GetAlignedCameraPose(KeyFrame* pKF, Eigen::Matrix3f& R_cw, Eigen::Vector3f& t_cw);
    
    // Compute homography from ground plane to image for a KeyFrame
    // H_plane2img: 3x3 matrix mapping [X, Z, 1]^T (ground coords) to [u, v, 1]^T (pixel coords)
    // H_img2ground: 3x3 matrix mapping [u, v, 1]^T (pixel coords) to [X, Z, 1]^T (ground coords)
    bool ComputeGroundHomography(KeyFrame* pKF, Eigen::Matrix3f& H_plane2img, Eigen::Matrix3f& H_img2ground);
    
    // Convert pixel coordinates to ground plane coordinates (in aligned world meters)
    // Returns true if successful, false if alignment not available
    // X_ground, Z_ground: ground coordinates in meters (Y = 0 in aligned frame)
    bool PixelToGround(KeyFrame* pKF, float u, float v, float& X_ground, float& Z_ground);
    
    // BEV (Bird's Eye View) functions - delegated to BEVGenerator
    BEVGenerator* GetBEVGenerator() { return mpBEVGenerator; }
    const BEVGenerator* GetBEVGenerator() const { return mpBEVGenerator; }
    
    // Convenience wrappers for backward compatibility
    void SetBEVWindow(float X_min, float X_max, float Z_min, float Z_max, float pixels_per_meter = 300.0f);
    void SetBEVOutputDirectory(const std::string& dir);
    void SaveAllKeyframeBEVs(const cv::Mat& current_img, KeyFrame* pCurrentKF);
           

protected:

    bool ParseViewerParamFile(cv::FileStorage &fSettings);

    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;

    Sophus::SE3f mCameraPose;

    std::mutex mMutexCamera;
    mutable std::mutex mMutexGroundPlane; // Mutex for thread-safe access to ground plane state (mutable for const accessors)

    float mfFrameColors[6][3] = {{0.0f, 0.0f, 1.0f},
                                {0.8f, 0.4f, 1.0f},
                                {1.0f, 0.2f, 0.4f},
                                {0.6f, 0.0f, 1.0f},
                                {1.0f, 1.0f, 0.0f},
                                {0.0f, 1.0f, 1.0f}};

    bool m_hasGroundPlane = false;
    Eigen::Vector3f m_groundNormal;
    float m_groundD = 0.0f;
    Eigen::Vector3f m_groundCentroid;

    bool m_hasAlignment = false;
    Eigen::Matrix3f m_R_align = Eigen::Matrix3f::Identity();
    Eigen::Vector3f m_t_align = Eigen::Vector3f::Zero();

    // BEV generator instance
    BEVGenerator* mpBEVGenerator;
    
    // Map change tracking for ground plane invalidation
    unsigned int m_lastMapChangeIndex = 0;
    bool m_groundPlaneEstimating = false; // Flag to prevent concurrent estimation
    int m_lastEstimationPointCount = 0; // Track point count when plane was last estimated
    

private:

    void ComputeWorldAlignmentFromPlane();

};

} //namespace ORB_SLAM

#endif // MAPDRAWER_H
