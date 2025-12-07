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
* GNU General License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef BEVGENERATOR_H
#define BEVGENERATOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>
#include <mutex>
#include <string>
#include <set>

namespace ORB_SLAM3
{

class KeyFrame;
class MapDrawer;

/**
 * @brief BEVGenerator - Generates Bird's Eye View (BEV) images from camera frames
 * 
 * This class handles all BEV-related operations including:
 * - Computing homographies from image to BEV space
 * - Warping camera images to BEV perspective
 * - Saving BEV images to disk
 * - Managing BEV configuration parameters
 */
class BEVGenerator
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief Constructor
     * @param pMapDrawer Pointer to MapDrawer for accessing alignment information
     */
    BEVGenerator(MapDrawer* pMapDrawer);

    /**
     * @brief Destructor
     */
    ~BEVGenerator();

    /**
     * @brief Set BEV window parameters
     * @param X_min Minimum X coordinate in aligned world (meters)
     * @param X_max Maximum X coordinate in aligned world (meters)
     * @param Z_min Minimum Z coordinate (depth) in aligned world (meters)
     * @param Z_max Maximum Z coordinate (depth) in aligned world (meters)
     * @param pixels_per_meter Resolution in pixels per meter
     */
    void SetWindow(float X_min, float X_max, float Z_min, float Z_max, float pixels_per_meter = 300.0f);

    /**
     * @brief Set output directory for BEV images
     * @param dir Directory path (will be created if it doesn't exist)
     */
    void SetOutputDirectory(const std::string& dir);

    /**
     * @brief Get BEV image width in pixels
     * @return Width in pixels
     */
    int GetWidth() const { return m_width; }

    /**
     * @brief Get BEV image height in pixels
     * @return Height in pixels
     */
    int GetHeight() const { return m_height; }

    /**
     * @brief Compute homography from image to BEV space
     * @param pKF KeyFrame to use for pose and calibration
     * @param H_img2bev Output homography matrix (3x3)
     * @return true if successful, false otherwise
     */
    bool ComputeHomography(KeyFrame* pKF, Eigen::Matrix3f& H_img2bev);

    /**
     * @brief Save BEV image for a keyframe
     * @param current_img Current camera image
     * @param pCurrentKF KeyFrame associated with the image
     * @return true if saved successfully, false otherwise
     */
    bool SaveKeyframeBEV(const cv::Mat& current_img, KeyFrame* pCurrentKF);
    
    /**
     * @brief Generate BEV image for a keyframe (without saving)
     * @param current_img Current camera image
     * @param pCurrentKF KeyFrame associated with the image
     * @param bev_image Output BEV image
     * @return true if generated successfully, false otherwise
     */
    bool GenerateBEVImage(const cv::Mat& current_img, KeyFrame* pCurrentKF, cv::Mat& bev_image);

    /**
     * @brief Parse BEV parameters from YAML config file
     * @param fSettings OpenCV FileStorage object
     * @return true if parsing successful (parameters are optional)
     */
    bool ParseParameters(cv::FileStorage &fSettings);

    /**
     * @brief Reset saved keyframes tracking (useful for map resets)
     */
    void Reset();

private:
    /**
     * @brief Validate BEV dimensions and memory requirements
     * @return true if dimensions are valid, false otherwise
     */
    bool ValidateDimensions() const;

    /**
     * @brief Update BEV dimensions based on current parameters
     */
    void UpdateDimensions();

    // Pointer to MapDrawer for accessing alignment information
    MapDrawer* mpMapDrawer;

    // BEV window parameters (in aligned world meters)
    float m_X_min;
    float m_X_max;
    float m_Z_min;
    float m_Z_max;
    float m_pixels_per_meter;

    // BEV image dimensions (computed from parameters)
    int m_width;
    int m_height;

    // Output directory for BEV images
    std::string m_output_dir;

    // Track which keyframes have had their BEV saved
    std::set<unsigned long> m_saved_keyframes;
    std::mutex mMutexSaved; // Mutex for thread-safe access to m_saved_keyframes

    // Constants
    static const int MAX_DIMENSION;
    static const size_t MAX_MEMORY_BYTES;
};

} // namespace ORB_SLAM3

#endif // BEVGENERATOR_H

