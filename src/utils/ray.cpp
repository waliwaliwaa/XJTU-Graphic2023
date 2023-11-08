#include "ray.h"

#include <cmath>
#include <array>

#include <Eigen/Dense>
#include <spdlog/spdlog.h>
#include <iostream>
#include "../utils/math.hpp"

using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Vector4f;
using std::numeric_limits;
using std::optional;
using std::size_t;

constexpr float infinity = 1e5f;
constexpr float eps      = 1e-5f;

Intersection::Intersection() : t(numeric_limits<float>::infinity()), face_index(0)
{
}

Ray generate_ray(int width, int height, int x, int y, Camera& camera, float depth)
{

    // The ratio between the specified plane (width x height)'s depth and the image plane's depth.

    // Transfer the view-space position to world space.
    // 计算归一化成像平面中的像素坐标 pos
    Vector2f pos((float)x + 0.5f, (float)y + 0.5f);
    Vector2f center((float)width / 2.0f, (float)height / 2.0f);
    // depth = 5;
    //  计算相机坐标系中的视图坐标 view_pos_specified
    Matrix4f inv_view = camera.view().inverse();
    Vector4f view_pos_specified =
        Vector4f(pos.x() - center.x(), -(pos.y() - center.y()), -depth, 1.0f);

    // Vector3f world_pos           = world_pos_specified.head<3>();

    // 根据 fov 计算归一化成像平面的高度，并计算视图坐标到世界坐标的转换比例 ratio
    float fov_y              = radians(camera.fov_y_degrees);
    float image_plane_height = 2 * std::tan(fov_y / 2.0f) / depth;
    float ratio              = image_plane_height / (float)height;

    // 将视图坐标乘以转换比例，得到最终的世界坐标 world_pos
    Vector4f view_pos = ratio * view_pos_specified;

    // 使用相机的逆视图矩阵将视图坐标转换为世界坐标 world_pos_specified
    Vector4f world_pos_specified = inv_view * view_pos;
    Vector3f world_pos           = world_pos_specified.head<3>();
    // world_pos         = view_pos.head<3>() / view_pos.w();
    // std::cout << camera.position << std::endl;
    return {camera.position, (world_pos - camera.position).normalized()};
}

optional<Intersection> ray_triangle_intersect(const Ray& ray, const GL::Mesh& mesh, size_t index)
{
    // 此函数内是model坐标系
    // 函数外都是世界坐标系
    
    Intersection result;
    result.t                   = infinity;
    std::array<size_t, 3> face = mesh.face(index);
    Vector3f a                 = mesh.vertex(face[0]);
    Vector3f b                 = mesh.vertex(face[1]);
    Vector3f c                 = mesh.vertex(face[2]);
    // a                          = (model * Vector4f(a.x(), a.y(), a.z(), 1)).head<3>();
    // b                          = (model * Vector4f(b.x(), b.y(), b.z(), 1)).head<3>();
    // c                          = (model * Vector4f(c.x(), c.y(), c.z(), 1)).head<3>();
    Vector3f E1 = b - a;
    Vector3f E2 = c - a;
    Vector3f S  = ray.origin - a;
    Vector3f S1 = ray.direction.cross(E2);
    Vector3f S2 = S.cross(E1);
    if (std::fabs(S1.dot(E1)) < eps)
        return std::nullopt;
    float coef = 1.0 / S1.dot(E1);
    float t    = coef * S2.dot(E2);
    float b1   = coef * S1.dot(S);
    float b2   = coef * S2.dot(ray.direction);

    if (t >= 0 && b1 >= 0 && b2 >= 0 && (1 - b1 - b2) >= 0) {
        
        if (t < eps)
            return std::nullopt;
        if (t < result.t && t > eps) {
            result.t                 = t;
            result.face_index        = index;
            result.barycentric_coord = {1 - b1 - b2, b1, b2};
            // 已知三角形三个顶点法向量，求三角形所在平面法向量。顶点法向量肯定是一样的，所以相加之后归一化就行
            result.normal = E1.cross(E2).normalized();
            //
        }
    }

    if (result.t - infinity < -eps) {
        return result;
    } else {
        return std::nullopt;
    }
}

optional<Intersection> naive_intersect(const Ray& ray, const GL::Mesh& mesh, const Matrix4f model)
{
    Intersection result;
    result.t = infinity;
    // https://blog.csdn.net/zhanxi1992/article/details/109903792
    for (size_t i = 0; i < mesh.faces.count(); ++i) {
        // Vertex a, b and c are assumed to be in counterclockwise order.
        std::array<size_t, 3> face = mesh.face(i);
        Vector3f a                 = mesh.vertex(face[0]);
        Vector3f b                 = mesh.vertex(face[1]);
        Vector3f c                 = mesh.vertex(face[2]);
        a                          = (model * Vector4f(a.x(), a.y(), a.z(), 1)).head<3>();
        b                          = (model * Vector4f(b.x(), b.y(), b.z(), 1)).head<3>();
        c                          = (model * Vector4f(c.x(), c.y(), c.z(), 1)).head<3>();
        Vector3f E1                = b - a;
        Vector3f E2                = c - a;
        Vector3f S                 = ray.origin - a;
        Vector3f S1                = ray.direction.cross(E2);
        Vector3f S2                = S.cross(E1);
        if (std::fabs(S1.dot(E1)) < eps)
            continue;
        float coef = 1.0 / S1.dot(E1);
        float t    = coef * S2.dot(E2);
        float b1   = coef * S1.dot(S);
        float b2   = coef * S2.dot(ray.direction);

        if (t >= 0 && b1 >= 0 && b2 >= 0 && (1 - b1 - b2) >= 0) {

            if (t < eps)
                continue;
            if (t < result.t && t > eps) {
                result.t                 = t;
                result.face_index        = i;
                result.barycentric_coord = {1 - b1 - b2, b1, b2};
                // 已知三角形三个顶点法向量，求三角形所在平面法向量。顶点法向量肯定是一样的，所以相加之后归一化就行
                result.normal = E1.cross(E2).normalized();
            }
        }
    }
    if (result.t - infinity < -eps) {
        return result;
    }
    return std::nullopt;
}
