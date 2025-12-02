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

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <iostream>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// Simple least-squares plane fit: n·X + d = 0
struct PlaneLSQ {
    Eigen::Vector3f n; // unit normal
    float d;
    Eigen::Vector3f centroid;
    float avgResidual = 0.0f;
    float maxResidual = 0.0f;
};

PlaneLSQ fitPlaneLeastSquares(const std::vector<Eigen::Vector3f>& pts)
{
    PlaneLSQ plane;
    // 1) centroid
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (auto& p : pts) centroid += p;
    centroid /= static_cast<float>(pts.size());
    plane.centroid = centroid;

    // 2) covariance
    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    for (auto& p : pts) {
        Eigen::Vector3f q = p - centroid;
        cov += q * q.transpose();
    }

    // 3) smallest eigenvector = normal
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
    plane.n = solver.eigenvectors().col(0); // smallest eigenvalue
    plane.n.normalize();

    // 4) d = -n·centroid
    plane.d = -plane.n.dot(centroid);

    // 5) compute residuals
    float sumResidual = 0.0f;
    float maxRes = 0.0f;
    for (auto& p : pts) {
        float dist = std::abs(plane.n.dot(p) + plane.d);
        sumResidual += dist;
        maxRes = std::max(maxRes, dist);
    }
    plane.avgResidual = sumResidual / static_cast<float>(pts.size());
    plane.maxResidual = maxRes;

    return plane;
}

} // anonymous namespace

namespace ORB_SLAM3
{


MapDrawer::MapDrawer(Atlas* pAtlas, const string &strSettingPath, Settings* settings):mpAtlas(pAtlas)
{
    if(settings){
        newParameterLoader(settings);
    }
    else{
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        bool is_correct = ParseViewerParamFile(fSettings);

        if(!is_correct)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }
}

void MapDrawer::newParameterLoader(Settings *settings) {
    mKeyFrameSize = settings->keyFrameSize();
    mKeyFrameLineWidth = settings->keyFrameLineWidth();
    mGraphLineWidth = settings->graphLineWidth();
    mPointSize = settings->pointSize();
    mCameraSize = settings->cameraSize();
    mCameraLineWidth  = settings->cameraLineWidth();
}

bool MapDrawer::ParseViewerParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::FileNode node = fSettings["Viewer.KeyFrameSize"];
    if(!node.empty())
    {
        mKeyFrameSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.KeyFrameSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.KeyFrameLineWidth"];
    if(!node.empty())
    {
        mKeyFrameLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.KeyFrameLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.GraphLineWidth"];
    if(!node.empty())
    {
        mGraphLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.GraphLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.PointSize"];
    if(!node.empty())
    {
        mPointSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.PointSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.CameraSize"];
    if(!node.empty())
    {
        mCameraSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.CameraSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.CameraLineWidth"];
    if(!node.empty())
    {
        mCameraLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.CameraLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    return !b_miss_params;
}

void MapDrawer::DrawMapPoints()
{
    Map* pActiveMap = mpAtlas->GetCurrentMap();
    if(!pActiveMap)
        return;

    const vector<MapPoint*> &vpMPs = pActiveMap->GetAllMapPoints();
    const vector<MapPoint*> &vpRefMPs = pActiveMap->GetReferenceMapPoints();

    if(vpMPs.empty())
        return;

    // --- NEW: estimate plane once when map is non-empty ---
    if (!m_hasGroundPlane) {
        EstimateGroundPlane();
    }

    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);

    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
        Eigen::Vector3f pos = vpMPs[i]->GetWorldPos();

        if (m_hasAlignment) {
            pos = m_R_align * pos + m_t_align;
        }

        glVertex3f(pos(0), pos(1), pos(2));
    }
    glEnd();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);

    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        if((*sit)->isBad())
            continue;
        Eigen::Vector3f pos = (*sit)->GetWorldPos();

        if (m_hasAlignment) {
            pos = m_R_align * pos + m_t_align;
        }

        glVertex3f(pos(0), pos(1), pos(2));

    }

    glEnd();

    // === Draw ground plane grid (aligned) ===
    if (m_hasAlignment)
    {
        float halfSize = 5.0f;
        int steps = 10;

        glLineWidth(1.0f);
        glColor3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_LINES);

        for (int i=-steps; i<=steps; ++i)
        {
            float alpha = halfSize * float(i) / float(steps);

            // lines along X'
            glVertex3f(alpha, 0.0f, -halfSize);
            glVertex3f(alpha, 0.0f,  halfSize);

            // lines along Z'
            glVertex3f(-halfSize, 0.0f, alpha);
            glVertex3f( halfSize, 0.0f, alpha);
        }

        glEnd();
    }
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph, const bool bDrawOptLba)
{
    const float &w = mKeyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;

    Map* pActiveMap = mpAtlas->GetCurrentMap();
    // DEBUG LBA
    std::set<long unsigned int> sOptKFs = pActiveMap->msOptKFs;
    std::set<long unsigned int> sFixedKFs = pActiveMap->msFixedKFs;

    if(!pActiveMap)
        return;

    const vector<KeyFrame*> vpKFs = pActiveMap->GetAllKeyFrames();

    // Debug: print heights of a few keyframe camera centers
    if (m_hasAlignment && !vpKFs.empty())
    {
        int numToPrint = std::min(5, static_cast<int>(vpKFs.size()));
        std::cout << "[MapDrawer] KeyFrame heights (aligned Y coordinate):" << std::endl;
        for (int i = 0; i < numToPrint; ++i)
        {
            KeyFrame* pKF = vpKFs[i];
            Eigen::Vector3f Ow = pKF->GetCameraCenter();
            Eigen::Vector3f Ow_aligned = m_R_align * Ow + m_t_align;
            float h = Ow_aligned.y();
            std::cout << "  KF[" << i << "] (id=" << pKF->mnId << "): h = " << h << std::endl;
        }
    }

    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];
            Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();

            if (m_hasAlignment) {
                Eigen::Matrix4f T_align = Eigen::Matrix4f::Identity();
                T_align.block<3,3>(0,0) = m_R_align;
                T_align.block<3,1>(0,3) = m_t_align;
                Twc = T_align * Twc;
            }

            unsigned int index_color = pKF->mnOriginMapId;

            glPushMatrix();

            glMultMatrixf((GLfloat*)Twc.data());

            if(!pKF->GetParent()) // It is the first KF in the map
            {
                glLineWidth(mKeyFrameLineWidth*5);
                glColor3f(1.0f,0.0f,0.0f);
                glBegin(GL_LINES);
            }
            else
            {
                //cout << "Child KF: " << vpKFs[i]->mnId << endl;
                glLineWidth(mKeyFrameLineWidth);
                if (bDrawOptLba) {
                    if(sOptKFs.find(pKF->mnId) != sOptKFs.end())
                    {
                        glColor3f(0.0f,1.0f,0.0f); // Green -> Opt KFs
                    }
                    else if(sFixedKFs.find(pKF->mnId) != sFixedKFs.end())
                    {
                        glColor3f(1.0f,0.0f,0.0f); // Red -> Fixed KFs
                    }
                    else
                    {
                        glColor3f(0.0f,0.0f,1.0f); // Basic color
                    }
                }
                else
                {
                    glColor3f(0.0f,0.0f,1.0f); // Basic color
                }
                glBegin(GL_LINES);
            }

            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();

            glEnd();
        }
    }

    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        // cout << "-----------------Draw graph-----------------" << endl;
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
            Eigen::Vector3f Ow = vpKFs[i]->GetCameraCenter();
            if (m_hasAlignment) {
                Ow = m_R_align * Ow + m_t_align;
            }
            if(!vCovKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
                    Eigen::Vector3f Ow2 = (*vit)->GetCameraCenter();
                    if (m_hasAlignment) {
                        Ow2 = m_R_align * Ow2 + m_t_align;
                    }
                    glVertex3f(Ow(0),Ow(1),Ow(2));
                    glVertex3f(Ow2(0),Ow2(1),Ow2(2));
                }
            }

            // Spanning tree
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
                Eigen::Vector3f Owp = pParent->GetCameraCenter();
                if (m_hasAlignment) {
                    Owp = m_R_align * Owp + m_t_align;
                }
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }

            // Loops
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
                Eigen::Vector3f Owl = (*sit)->GetCameraCenter();
                if (m_hasAlignment) {
                    Owl = m_R_align * Owl + m_t_align;
                }
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owl(0),Owl(1),Owl(2));
            }
        }

        glEnd();
    }

    if(bDrawInertialGraph && pActiveMap->isImuInitialized())
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(1.0f,0.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        //Draw inertial links
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            Eigen::Vector3f Ow = pKFi->GetCameraCenter();
            if (m_hasAlignment) {
                Ow = m_R_align * Ow + m_t_align;
            }
            KeyFrame* pNext = pKFi->mNextKF;
            if(pNext)
            {
                Eigen::Vector3f Owp = pNext->GetCameraCenter();
                if (m_hasAlignment) {
                    Owp = m_R_align * Owp + m_t_align;
                }
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }
        }

        glEnd();
    }

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();

    if(bDrawKF)
    {
        for(Map* pMap : vpMaps)
        {
            if(pMap == pActiveMap)
                continue;

            vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

            for(size_t i=0; i<vpKFs.size(); i++)
            {
                KeyFrame* pKF = vpKFs[i];
                Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();

                if (m_hasAlignment) {
                    Eigen::Matrix4f T_align = Eigen::Matrix4f::Identity();
                    T_align.block<3,3>(0,0) = m_R_align;
                    T_align.block<3,1>(0,3) = m_t_align;
                    Twc = T_align * Twc;
                }

                unsigned int index_color = pKF->mnOriginMapId;

                glPushMatrix();

                glMultMatrixf((GLfloat*)Twc.data());

                if(!vpKFs[i]->GetParent()) // It is the first KF in the map
                {
                    glLineWidth(mKeyFrameLineWidth*5);
                    glColor3f(1.0f,0.0f,0.0f);
                    glBegin(GL_LINES);
                }
                else
                {
                    glLineWidth(mKeyFrameLineWidth);
                    glColor3f(mfFrameColors[index_color][0],mfFrameColors[index_color][1],mfFrameColors[index_color][2]);
                    glBegin(GL_LINES);
                }

                glVertex3f(0,0,0);
                glVertex3f(w,h,z);
                glVertex3f(0,0,0);
                glVertex3f(w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,h,z);

                glVertex3f(w,h,z);
                glVertex3f(w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(-w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(w,h,z);

                glVertex3f(-w,-h,z);
                glVertex3f(w,-h,z);
                glEnd();

                glPopMatrix();
            }
        }
    }
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}


void MapDrawer::SetCurrentCameraPose(const Sophus::SE3f &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.inverse();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw)
{
    Eigen::Matrix4f Twc;
    {
        unique_lock<mutex> lock(mMutexCamera);
        Twc = mCameraPose.matrix();
    }

    for (int i = 0; i<4; i++) {
        M.m[4*i] = Twc(0,i);
        M.m[4*i+1] = Twc(1,i);
        M.m[4*i+2] = Twc(2,i);
        M.m[4*i+3] = Twc(3,i);
    }

    MOw.SetIdentity();
    MOw.m[12] = Twc(0,3);
    MOw.m[13] = Twc(1,3);
    MOw.m[14] = Twc(2,3);
}

void MapDrawer::EstimateGroundPlane()
{
    Map* pActiveMap = mpAtlas->GetCurrentMap();
    if (!pActiveMap) return;

    const vector<MapPoint*> &vpMPs = pActiveMap->GetAllMapPoints();
    if (vpMPs.empty()) return;

    std::vector<Eigen::Vector3f> pts;
    pts.reserve(vpMPs.size());

    for (auto pMP : vpMPs)
    {
        if (!pMP || pMP->isBad()) continue;

        Eigen::Vector3f pos = pMP->GetWorldPos();

        if (!std::isfinite(pos.x()) || !std::isfinite(pos.y()) || !std::isfinite(pos.z()))
            continue;

        pts.push_back(pos);
    }

    if (pts.size() < 50) {
        std::cout << "[MapDrawer] Not enough points to estimate ground plane." << std::endl;
        return;
    }

    PlaneLSQ plane = fitPlaneLeastSquares(pts);

    m_groundNormal   = plane.n;
    m_groundD        = plane.d;
    m_groundCentroid = plane.centroid;
    m_hasGroundPlane = true;

    // diagnostics (optional)
    Eigen::Vector3f up(0.0f, 1.0f, 0.0f);
    float cosTheta = plane.n.dot(up);
    cosTheta = std::max(-1.0f, std::min(1.0f, cosTheta));
    float angleDeg = std::acos(cosTheta) * 180.0f / static_cast<float>(M_PI);

    std::cout << "[MapDrawer] Ground plane estimated:\n"
              << "  n = [" << m_groundNormal.transpose() << "]\n"
              << "  d = " << m_groundD << "\n"
              << "  centroid = [" << plane.centroid.transpose() << "]\n"
              << "  avg residual = " << plane.avgResidual
              << ", max residual = " << plane.maxResidual << "\n"
              << "  angle to +Y axis = " << angleDeg << " deg" << std::endl;

    // NEW: compute alignment (normal -> +Y, plane -> Y=0)
    ComputeWorldAlignmentFromPlane();
}

void MapDrawer::ComputeWorldAlignmentFromPlane()
{
    if (!m_hasGroundPlane)
        return;

    // 1. rotation: ground normal -> +Y
    Eigen::Vector3f up(0.0f, 1.0f, 0.0f);
    Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(m_groundNormal, up);
    m_R_align = q.toRotationMatrix();

    // 2. translation: send plane to Y=0 using centroid as reference point
    //    X0 is a point on the plane in original world
    Eigen::Vector3f X0 = m_groundCentroid;
    Eigen::Vector3f X0_rot = m_R_align * X0; // in aligned frame

    float y_shift = -X0_rot.y(); // so that this point ends up at Y=0
    m_t_align = Eigen::Vector3f(0.0f, y_shift, 0.0f);

    m_hasAlignment = true;

    std::cout << "[MapDrawer] Alignment computed:\n"
              << "  R_align = \n" << m_R_align << "\n"
              << "  t_align = [" << m_t_align.transpose() << "]" << std::endl;
}

} //namespace ORB_SLAM
