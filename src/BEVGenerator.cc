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

#include "BEVGenerator.h"
#include "MapDrawer.h"
#include "KeyFrame.h"
#include "Map.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <climits>
#include <limits>
#include <cmath>
#include <unistd.h>
#include <sys/wait.h>
#include <cstdlib>

namespace ORB_SLAM3
{

// Static constants
const int BEVGenerator::MAX_DIMENSION = 4000;
const size_t BEVGenerator::MAX_MEMORY_BYTES = 50 * 1024 * 1024; // 50MB

BEVGenerator::BEVGenerator(MapDrawer* pMapDrawer)
    : mpMapDrawer(pMapDrawer),
      m_X_min(-2.0f),
      m_X_max(2.0f),
      m_Z_min(2.0f),
      m_Z_max(3.5f),
      m_pixels_per_meter(300.0f),
      m_width(0),
      m_height(0),
      m_output_dir("./bev_output/")
{
    UpdateDimensions();
}

BEVGenerator::~BEVGenerator()
{
}

void BEVGenerator::SetWindow(float X_min, float X_max, float Z_min, float Z_max, float pixels_per_meter)
{
    m_X_min = X_min;
    m_X_max = X_max;
    m_Z_min = Z_min;
    m_Z_max = Z_max;
    m_pixels_per_meter = pixels_per_meter;
    UpdateDimensions();
}

void BEVGenerator::SetOutputDirectory(const std::string& dir)
{
    m_output_dir = dir;
    // Ensure directory ends with separator
    if (!m_output_dir.empty() && m_output_dir.back() != '/')
    {
        m_output_dir += "/";
    }
    
    // Create directory if it doesn't exist
    // Use filesystem API if available, otherwise use system call with timeout
    // For now, just set the directory - creation will happen lazily when saving
    // This avoids blocking during initialization
    std::cout << "[BEVGenerator] BEV output directory set to: " << m_output_dir << std::endl;
}

void BEVGenerator::UpdateDimensions()
{
    m_width = static_cast<int>((m_X_max - m_X_min) * m_pixels_per_meter);
    m_height = static_cast<int>((m_Z_max - m_Z_min) * m_pixels_per_meter);
}

bool BEVGenerator::ValidateDimensions() const
{
    if (m_width <= 0 || m_height <= 0 || 
        m_width > MAX_DIMENSION || m_height > MAX_DIMENSION ||
        !std::isfinite(static_cast<float>(m_width)) || !std::isfinite(static_cast<float>(m_height))) {
        return false;
    }
    
    // Check total memory requirement (width * height * 3 bytes for BGR)
    // Use safe multiplication to prevent overflow
    size_t width_s = static_cast<size_t>(m_width);
    size_t height_s = static_cast<size_t>(m_height);
    if (width_s > SIZE_MAX / height_s / 3) {
        return false; // Multiplication would overflow
    }
    size_t memory_bytes = width_s * height_s * 3;
    if (memory_bytes > MAX_MEMORY_BYTES || memory_bytes == 0) {
        return false;
    }
    
    return true;
}

bool BEVGenerator::ComputeHomography(KeyFrame* pKF, Eigen::Matrix3f& H_img2bev)
{
    std::cout << "[BEVGenerator::ComputeHomography] Starting for KF " << (pKF ? pKF->mnId : -1) << std::endl;
    
    if (!pKF || !mpMapDrawer) {
        std::cout << "[BEVGenerator::ComputeHomography] Early exit: invalid inputs" << std::endl;
        return false;
    }
    
    // Check alignment state
    if (!mpMapDrawer->HasAlignment()) {
        std::cout << "[BEVGenerator::ComputeHomography] Early exit: no alignment" << std::endl;
        return false;
    }

    std::cout << "[BEVGenerator::ComputeHomography] Computing ground homography..." << std::endl;
    
    // Get image->ground homography
    Eigen::Matrix3f H_plane2img, H_img2ground;
    if (!mpMapDrawer->ComputeGroundHomography(pKF, H_plane2img, H_img2ground)) {
        std::cout << "[BEVGenerator::ComputeHomography] Early exit: failed to compute ground homography" << std::endl;
        return false;
    }
    
    std::cout << "[BEVGenerator::ComputeHomography] Ground homography computed, validating..." << std::endl;
    
    // Validate homography values
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            if (!std::isfinite(H_img2ground(r, c)) || std::abs(H_img2ground(r, c)) > 1e6) {
                std::cout << "[BEVGenerator::ComputeHomography] Early exit: invalid ground homography values" << std::endl;
                return false;
            }
        }
    }

    std::cout << "[BEVGenerator::ComputeHomography] Getting aligned camera pose..." << std::endl;
    
    // Get camera position in aligned world coordinates
    Eigen::Matrix3f R_cw;
    Eigen::Vector3f t_cw;
    mpMapDrawer->GetAlignedCameraPose(pKF, R_cw, t_cw);
    
    std::cout << "[BEVGenerator::ComputeHomography] Camera pose: t_cw = [" 
              << t_cw(0) << ", " << t_cw(1) << ", " << t_cw(2) << "]" << std::endl;
    
    // Validate camera pose
    if (!std::isfinite(t_cw(0)) || !std::isfinite(t_cw(1)) || !std::isfinite(t_cw(2))) {
        std::cout << "[BEVGenerator::ComputeHomography] Early exit: invalid camera pose" << std::endl;
        return false;
    }
    
    // Camera position in aligned world (world->camera, so invert to get camera->world)
    Eigen::Matrix3f R_wc = R_cw.transpose();
    Eigen::Vector3f t_wc = -R_wc * t_cw;
    
    // Camera's Z position in aligned world coordinates
    float camera_Z = t_wc(2);  // Z coordinate of camera in aligned world
    
    std::cout << "[BEVGenerator::ComputeHomography] Camera Z position: " << camera_Z << std::endl;
    
    // Validate camera Z position
    if (!std::isfinite(camera_Z)) {
        std::cout << "[BEVGenerator::ComputeHomography] Early exit: invalid camera Z" << std::endl;
        return false;
    }
    
    // Make BEV window relative to camera position
    // m_Z_min and m_Z_max are now interpreted as offsets from camera
    float Z_min_abs = camera_Z + m_Z_min;  // Near distance from camera
    float Z_max_abs = camera_Z + m_Z_max;  // Far distance from camera
    
    // Validate Z bounds
    if (!std::isfinite(Z_min_abs) || !std::isfinite(Z_max_abs) || Z_min_abs >= Z_max_abs) {
        return false;
    }
    
    // Ensure minimum Z range to prevent numerical instability
    // Small Z ranges can cause large homography values and crashes
    float Z_range = Z_max_abs - Z_min_abs;
    if (Z_range < 0.5f) {  // Minimum 0.5 meter range
        return false;
    }
    
    // Ensure Z_min is not too close to camera (behind or very close)
    // This prevents projection of points behind camera or at infinity
    if (Z_min_abs < camera_Z - 0.1f) {  // At least 0.1m in front of camera
        return false;
    }
    
    // Build world→BEV transform T_bev
    // X = X_min → u_bev = 0
    // X = X_max → u_bev = bev_width
    // Z = Z_max_abs → v_bev = 0     (far = top)
    // Z = Z_min_abs → v_bev = bev_height  (near = bottom)
    float s = m_pixels_per_meter;
    
    if (!std::isfinite(s) || s <= 0 || s > 10000) {
        return false;
    }
    
    Eigen::Matrix3f T_bev = Eigen::Matrix3f::Identity();
    T_bev <<
        s,    0,  -s * m_X_min,
        0,   -s,   s * Z_max_abs,
        0,    0,   1;

    std::cout << "[BEVGenerator::ComputeHomography] Composing final homography..." << std::endl;
    
    // Compose: image → ground → BEV
    H_img2bev = T_bev * H_img2ground;
    
    std::cout << "[BEVGenerator::ComputeHomography] Final homography computed, validating..." << std::endl;
    
    // Final validation of result with stricter checks for small Z ranges
    // When Z range is small, homography values can be large, so we need tighter bounds
    float max_homography_value = (Z_range < 2.0f) ? 1e5f : 1e6f;  // Stricter for small ranges
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float val = H_img2bev(r, c);
            if (!std::isfinite(val) || std::abs(val) > max_homography_value) {
                std::cout << "[BEVGenerator::ComputeHomography] Early exit: invalid final homography at [" 
                          << r << "," << c << "] = " << val << std::endl;
                return false;
            }
        }
    }

    std::cout << "[BEVGenerator::ComputeHomography] Completed successfully" << std::endl;
    return true;
}

bool BEVGenerator::SaveKeyframeBEV(const cv::Mat& current_img, KeyFrame* pCurrentKF)
{
    if (current_img.empty() || !pCurrentKF || !mpMapDrawer) {
        return false;
    }
    
    // Check alignment state (fast check first)
    if (!mpMapDrawer->HasAlignment()) {
        return false;
    }
    
    // Validate input image dimensions (fast check)
    if (current_img.cols <= 0 || current_img.rows <= 0 || 
        current_img.cols > 10000 || current_img.rows > 10000) {
        return false;
    }
    
    // Only save BEV for the current keyframe (the one that matches the current image)
    // Check if already saved BEFORE acquiring lock (fast path)
    {
        std::lock_guard<std::mutex> lock(mMutexSaved);
        // Skip if already saved (check again after acquiring lock)
        if (m_saved_keyframes.find(pCurrentKF->mnId) != m_saved_keyframes.end()) {
            return false; // Already saved, exit quickly
        }
    }
    
    // Lock map mutex to safely access KeyFrame
    Map* pMap = pCurrentKF->GetMap();
    if (!pMap) {
        return false;
    }
    
    // Try to acquire lock with timeout to avoid blocking GUI thread
    // If we can't get the lock quickly, skip this save (will try again later)
    unique_lock<mutex> mapLock(pMap->mMutexMapUpdate, std::defer_lock);
    if (!mapLock.try_lock()) {
        // Couldn't acquire lock quickly, skip this save to avoid blocking
        return false;
    }
    
    // Validate keyframe is still valid after acquiring lock
    if (pCurrentKF->isBad() || !pCurrentKF->mpCamera) {
        return false;
    }
    
    // Update dimensions and validate
    UpdateDimensions();
    if (!ValidateDimensions()) {
        return false;
    }
    
    // Compute BEV homography for the current keyframe
    // This is expensive, so add a quick check to see if we should proceed
    Eigen::Matrix3f H_img2bev;
    if (!ComputeHomography(pCurrentKF, H_img2bev)) {
        return false;
    }
    
    // Debug: log that we're about to do expensive operations
    // (comment out in production if too verbose)
    // std::cout << "[BEVGenerator] Computing BEV for KF " << pCurrentKF->mnId << std::endl;
    
    // Convert Eigen matrix to OpenCV Mat and validate
    cv::Mat H_cv(3, 3, CV_32F);
    bool valid_h = true;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float val = H_img2bev(r, c);
            if (!std::isfinite(val) || std::abs(val) > 1e6) {
                valid_h = false;
                break;
            }
            H_cv.at<float>(r, c) = val;
        }
        if (!valid_h) break;
    }
    
    if (!valid_h) {
        return false;
    }
    
    // Prepare image (convert to BGR if needed) - do this BEFORE allocation
    cv::Mat im_bgr;
    if (current_img.channels() == 1) {
        try {
            cv::cvtColor(current_img, im_bgr, cv::COLOR_GRAY2BGR);
        } catch (const cv::Exception& e) {
            return false;
        } catch (const std::bad_alloc& e) {
            return false;
        }
    } else {
        im_bgr = current_img.clone();
    }
    
    if (im_bgr.empty() || im_bgr.cols <= 0 || im_bgr.rows <= 0 ||
        im_bgr.cols > 10000 || im_bgr.rows > 10000) {
        return false;
    }
    
    // Pre-allocate BEV image - use Mat::zeros for guaranteed allocation
    cv::Mat img_bev;
    try {
        // Double-check dimensions are still valid before allocation
        if (m_width <= 0 || m_height <= 0 || 
            m_width > MAX_DIMENSION || m_height > MAX_DIMENSION) {
            return false;
        }
        
        img_bev = cv::Mat::zeros(m_height, m_width, CV_8UC3);
        
        if (img_bev.empty() || img_bev.cols != m_width || img_bev.rows != m_height ||
            img_bev.data == nullptr || img_bev.cols <= 0 || img_bev.rows <= 0) {
            return false;
        }
    } catch (const cv::Exception& e) {
        return false;
    } catch (const std::bad_alloc& e) {
        return false;
    } catch (...) {
        return false;
    }
    
    // Final check before warping - ensure dimensions match
    if (m_width != img_bev.cols || m_height != img_bev.rows ||
        m_width <= 0 || m_height <= 0) {
        return false;
    }
    
    // Warp image to BEV - use pre-allocated size
    try {
        cv::warpPerspective(im_bgr, img_bev, H_cv, img_bev.size(),
                            cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    } catch (const cv::Exception& e) {
        return false;
    } catch (const std::bad_alloc& e) {
        return false;
    } catch (...) {
        return false;
    }
    
    // Validate warped image
    if (img_bev.empty() || img_bev.cols != m_width || img_bev.rows != m_height) {
        return false;
    }
    
    // Create directory lazily if it doesn't exist (only when we actually need to save)
    // Use a simple, fast method - mkdir should be very fast
    static bool dir_created = false;
    if (!dir_created && !m_output_dir.empty()) {
        std::string escaped_dir = m_output_dir;
        // Remove trailing slash for mkdir
        if (!escaped_dir.empty() && escaped_dir.back() == '/') {
            escaped_dir.pop_back();
        }
        
        // Create directory synchronously - this is fast and happens in background thread
        // The async task is already in a background thread, so blocking here is OK
        std::string cmd = "mkdir -p \"" + escaped_dir + "\" 2>/dev/null";
        int result = system(cmd.c_str());
        if (result == 0) {
            std::cout << "[BEVGenerator] Created output directory: " << escaped_dir << std::endl;
        }
        dir_created = true;
    }
    
    // Save the BEV image
    std::string bev_filename = m_output_dir + "kf_" + std::to_string(pCurrentKF->mnId) + "_bev_initial.png";
    try {
        if (cv::imwrite(bev_filename, img_bev)) {
            // Mark as saved only after successful save
            {
                std::lock_guard<std::mutex> lock(mMutexSaved);
                m_saved_keyframes.insert(pCurrentKF->mnId);
            }
            std::cout << "[BEVGenerator] Saved BEV image: " << bev_filename << std::endl;
            return true;
        } else {
            std::cerr << "[BEVGenerator] Warning: Failed to save BEV image: " << bev_filename << std::endl;
            return false;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "[BEVGenerator] OpenCV exception while saving BEV: " << e.what() << std::endl;
        return false;
    } catch (const std::bad_alloc& e) {
        std::cerr << "[BEVGenerator] Memory allocation failed while saving BEV" << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[BEVGenerator] Exception while saving BEV: " << e.what() << std::endl;
        return false;
    }
}

bool BEVGenerator::GenerateBEVImage(const cv::Mat& current_img, KeyFrame* pCurrentKF, cv::Mat& bev_image)
{
    std::cout << "[BEVGenerator] GenerateBEVImage called for KF " << (pCurrentKF ? pCurrentKF->mnId : -1) << std::endl;
    
    // Similar to SaveKeyframeBEV but returns the image instead of saving
    if (current_img.empty() || !pCurrentKF || !mpMapDrawer) {
        std::cout << "[BEVGenerator] Early exit: empty image=" << current_img.empty() 
                  << ", no KF=" << (!pCurrentKF) << ", no MapDrawer=" << (!mpMapDrawer) << std::endl;
        return false;
    }
    
    std::cout << "[BEVGenerator] Input image: " << current_img.cols << "x" << current_img.rows 
              << ", channels: " << current_img.channels() << std::endl;
    
    // Check alignment state (fast check first)
    if (!mpMapDrawer->HasAlignment()) {
        std::cout << "[BEVGenerator] Early exit: no alignment" << std::endl;
        return false;
    }
    
    std::cout << "[BEVGenerator] Alignment check passed" << std::endl;
    
    // Validate input image dimensions (fast check)
    if (current_img.cols <= 0 || current_img.rows <= 0 || 
        current_img.cols > 10000 || current_img.rows > 10000) {
        std::cout << "[BEVGenerator] Early exit: invalid image dimensions" << std::endl;
        return false;
    }
    
    // Lock map mutex to safely access KeyFrame
    Map* pMap = pCurrentKF->GetMap();
    if (!pMap) {
        std::cout << "[BEVGenerator] Early exit: no map" << std::endl;
        return false;
    }
    
    std::cout << "[BEVGenerator] Attempting to acquire map lock..." << std::endl;
    
    // Try to acquire lock with timeout to avoid blocking GUI thread
    unique_lock<mutex> mapLock(pMap->mMutexMapUpdate, std::defer_lock);
    if (!mapLock.try_lock()) {
        std::cout << "[BEVGenerator] Early exit: failed to acquire map lock" << std::endl;
        return false;
    }
    
    std::cout << "[BEVGenerator] Map lock acquired" << std::endl;
    
    // Validate keyframe is still valid after acquiring lock
    if (pCurrentKF->isBad() || !pCurrentKF->mpCamera) {
        std::cout << "[BEVGenerator] Early exit: KF is bad or no camera" << std::endl;
        return false;
    }
    
    std::cout << "[BEVGenerator] Updating dimensions..." << std::endl;
    
    // Update dimensions and validate
    UpdateDimensions();
    if (!ValidateDimensions()) {
        std::cout << "[BEVGenerator] Early exit: invalid dimensions" << std::endl;
        return false;
    }
    
    std::cout << "[BEVGenerator] Dimensions: " << m_width << "x" << m_height << std::endl;
    
    // Compute BEV homography for the current keyframe
    std::cout << "[BEVGenerator] Computing homography..." << std::endl;
    Eigen::Matrix3f H_img2bev;
    if (!ComputeHomography(pCurrentKF, H_img2bev)) {
        std::cout << "[BEVGenerator] Early exit: failed to compute homography" << std::endl;
        return false;
    }
    
    std::cout << "[BEVGenerator] Homography computed successfully" << std::endl;
    
    // Convert Eigen matrix to OpenCV Mat and validate
    std::cout << "[BEVGenerator] Converting homography matrix..." << std::endl;
    cv::Mat H_cv(3, 3, CV_32F);
    bool valid_h = true;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float val = H_img2bev(r, c);
            if (!std::isfinite(val) || std::abs(val) > 1e6) {
                valid_h = false;
                break;
            }
            H_cv.at<float>(r, c) = val;
        }
        if (!valid_h) break;
    }
    
    if (!valid_h) {
        std::cout << "[BEVGenerator] Early exit: invalid homography matrix" << std::endl;
        return false;
    }
    
    std::cout << "[BEVGenerator] Preparing image (BGR conversion)..." << std::endl;
    
    // Prepare image (convert to BGR if needed)
    cv::Mat im_bgr;
    if (current_img.channels() == 1) {
        try {
            cv::cvtColor(current_img, im_bgr, cv::COLOR_GRAY2BGR);
        } catch (const cv::Exception& e) {
            std::cout << "[BEVGenerator] Exception during color conversion: " << e.what() << std::endl;
            return false;
        } catch (const std::bad_alloc& e) {
            std::cout << "[BEVGenerator] Memory allocation failed during color conversion" << std::endl;
            return false;
        }
    } else {
        im_bgr = current_img.clone();
    }
    
    if (im_bgr.empty() || im_bgr.cols <= 0 || im_bgr.rows <= 0 ||
        im_bgr.cols > 10000 || im_bgr.rows > 10000) {
        std::cout << "[BEVGenerator] Early exit: invalid BGR image" << std::endl;
        return false;
    }
    
    std::cout << "[BEVGenerator] BGR image: " << im_bgr.cols << "x" << im_bgr.rows << std::endl;
    
    // Pre-allocate BEV image
    std::cout << "[BEVGenerator] Allocating BEV image buffer..." << std::endl;
    try {
        if (m_width <= 0 || m_height <= 0 || 
            m_width > MAX_DIMENSION || m_height > MAX_DIMENSION) {
            std::cout << "[BEVGenerator] Early exit: invalid BEV dimensions" << std::endl;
            return false;
        }
        
        bev_image = cv::Mat::zeros(m_height, m_width, CV_8UC3);
        
        if (bev_image.empty() || bev_image.cols != m_width || bev_image.rows != m_height ||
            bev_image.data == nullptr) {
            std::cout << "[BEVGenerator] Early exit: failed to allocate BEV image" << std::endl;
            return false;
        }
    } catch (const cv::Exception& e) {
        std::cout << "[BEVGenerator] OpenCV exception during allocation: " << e.what() << std::endl;
        return false;
    } catch (const std::bad_alloc& e) {
        std::cout << "[BEVGenerator] Memory allocation failed for BEV image" << std::endl;
        return false;
    } catch (...) {
        std::cout << "[BEVGenerator] Unknown exception during allocation" << std::endl;
        return false;
    }
    
    std::cout << "[BEVGenerator] BEV image allocated: " << bev_image.cols << "x" << bev_image.rows << std::endl;
    
    // Warp image to BEV
    std::cout << "[BEVGenerator] Warping image to BEV (this may take a moment)..." << std::endl;
    try {
        cv::warpPerspective(im_bgr, bev_image, H_cv, bev_image.size(),
                            cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        std::cout << "[BEVGenerator] Warping complete" << std::endl;
    } catch (const cv::Exception& e) {
        std::cout << "[BEVGenerator] OpenCV exception during warping: " << e.what() << std::endl;
        return false;
    } catch (const std::bad_alloc& e) {
        std::cout << "[BEVGenerator] Memory allocation failed during warping" << std::endl;
        return false;
    } catch (...) {
        std::cout << "[BEVGenerator] Unknown exception during warping" << std::endl;
        return false;
    }
    
    // Validate warped image
    if (bev_image.empty() || bev_image.cols != m_width || bev_image.rows != m_height) {
        std::cout << "[BEVGenerator] Early exit: invalid warped image" << std::endl;
        return false;
    }
    
    std::cout << "[BEVGenerator] GenerateBEVImage completed successfully" << std::endl;
    return true;
}

bool BEVGenerator::ParseParameters(cv::FileStorage &fSettings)
{
    // BEV parameters are optional - don't fail if they're missing
    cv::FileNode node = fSettings["BEV.X_min"];
    if(!node.empty())
    {
        m_X_min = node.real();
    }

    node = fSettings["BEV.X_max"];
    if(!node.empty())
    {
        m_X_max = node.real();
    }

    node = fSettings["BEV.Z_min"];
    if(!node.empty())
    {
        m_Z_min = node.real();
    }

    node = fSettings["BEV.Z_max"];
    if(!node.empty())
    {
        m_Z_max = node.real();
    }

    node = fSettings["BEV.pixels_per_meter"];
    if(!node.empty())
    {
        m_pixels_per_meter = node.real();
    }

    node = fSettings["BEV.output_dir"];
    if(!node.empty())
    {
        std::string dir = node.string();
        // Just set the directory path, don't create it yet (lazy creation)
        m_output_dir = dir;
        if (!m_output_dir.empty() && m_output_dir.back() != '/')
        {
            m_output_dir += "/";
        }
    }

    // Update dimensions with new parameters
    UpdateDimensions();

    return true; // Always return true since BEV params are optional
}

void BEVGenerator::Reset()
{
    std::lock_guard<std::mutex> lock(mMutexSaved);
    m_saved_keyframes.clear();
}

} // namespace ORB_SLAM3

