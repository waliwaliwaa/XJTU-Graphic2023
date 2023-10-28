#include "rasterizer.h"

#include <array>
#include <limits>
#include <tuple>
#include <vector>
#include <algorithm>
#include <mutex>
#include <omp.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <spdlog/spdlog.h>

#include "triangle.h"
#include "../utils/math.hpp"
#include <iostream>
using Eigen::Matrix4f;
using Eigen::Vector2i;
using Eigen::Vector3f;
using Eigen::Vector4f;

std::mutex mtx;
std::atomic_int process = 0;
// void Rasterizer::draw_mt(const std::vector<Triangle>& TriangleList, const GL::Material& material,
//                          const std::list<Light>& lights, const Camera& camera)
// {
//     // 线程数
//     const int num_threads = 8;
//     std::vector<std::thread> threads(num_threads);

//     // tri_per_thread是每个线程渲染的三角形数量
//     int tri_per_thread = TriangleList.size() / num_threads;
//     auto Thread        = [&](int start, int end) {
//         // lambda表达式
//         for (int j = start; j < end; ++j) {
//             // 三角形渲染代码
//             Triangle t           = TriangleList[j];
//             Triangle newtriangle = t;
//             // for (const auto& t : TriangleList) {
//             //     Triangle newtriangle = t;

//             // transform vertex position to world space for interpolating
//             std::array<Vector3f, 3> worldspace_pos;

//             for (int i = 0; i < 3; ++i) {
//                 Vector4f tmp          = Rasterizer::model * t.vertex[i];
//                 worldspace_pos[i].x() = tmp.x();
//                 worldspace_pos[i].y() = tmp.y();
//                 worldspace_pos[i].z() = tmp.z();
//             }

//             // Use vetex_shader to transform vertex attributes(position & normals)
//             // to view port and set a new triangle
//             std::array<VertexShaderPayload, 3> vertex_payloads;
//             for (int i = 0; i < 3; ++i) {
//                 // 将模型坐标用vertexshader转换成为视点坐标
//                 vertex_payloads[i].position = t.vertex[i];
//                 vertex_payloads[i].normal   = t.normal[i];
//                 vertex_payloads[i]          = vertex_shader(vertex_payloads[i]);
//                 // 用转换后的视点坐标构建新三角形
//                 newtriangle.vertex[i] = vertex_payloads[i].position;
//                 newtriangle.normal[i] = vertex_payloads[i].normal;
//             }

//             rasterize_triangle_mt(newtriangle, worldspace_pos, material, lights, camera);
//             //}
//         };
//     };
//     for (int i = 0; i < num_threads; ++i) {
//         int start  = i * tri_per_thread;
//         int end    = (i + 1) * tri_per_thread;
//         threads[i] = std::thread(Thread, start, end);
//     }

//     for (int i = 0; i < num_threads; ++i) threads[i].join();

//     for (int j = num_threads * tri_per_thread; j < TriangleList.size(); ++j) {
//         Triangle t           = TriangleList[j];
//         Triangle newtriangle = t;
//         std::array<Vector3f, 3> worldspace_pos;

//         for (int i = 0; i < 3; ++i) {
//             Vector4f tmp          = Rasterizer::model * t.vertex[i];
//             worldspace_pos[i].x() = tmp.x();
//             worldspace_pos[i].y() = tmp.y();
//             worldspace_pos[i].z() = tmp.z();
//         }
//         std::array<VertexShaderPayload, 3> vertex_payloads;
//         for (int i = 0; i < 3; ++i) {
//             // 将模型坐标用vertexshader转换成为视点坐标
//             vertex_payloads[i].position = t.vertex[i];
//             vertex_payloads[i].normal   = t.normal[i];
//             vertex_payloads[i]          = vertex_shader(vertex_payloads[i]);
//             // 用转换后的视点坐标构建新三角形
//             newtriangle.vertex[i] = vertex_payloads[i].position;
//             newtriangle.normal[i] = vertex_payloads[i].normal;
//         }

//         rasterize_triangle_mt(newtriangle, worldspace_pos, material, lights, camera);
//     };
// }

void Rasterizer::draw_mt(const std::vector<Triangle>& TriangleList, const GL::Material& material,
                         const std::list<Light>& lights, const Camera& camera)
{
// iterate over all triangles in TriangleList
    #pragma omp parallel for
    for (const auto& t : TriangleList) {
        Triangle newtriangle = t;

        // transform vertex position to world space for interpolating
        std::array<Vector3f, 3> worldspace_pos;

        for (int i = 0; i < 3; ++i) {
            Vector4f tmp          = Rasterizer::model * t.vertex[i];
            worldspace_pos[i].x() = tmp.x();
            worldspace_pos[i].y() = tmp.y();
            worldspace_pos[i].z() = tmp.z();
        }

        // Use vetex_shader to transform vertex attributes(position & normals) to
        // view port and set a new triangle
        std::array<VertexShaderPayload, 3> vertex_payloads;
        for (int i = 0; i < 3; ++i) {
            // 将模型坐标用vertexshader转换成为视点坐标
            vertex_payloads[i].position = t.vertex[i];
            vertex_payloads[i].normal   = t.normal[i];
            vertex_payloads[i]          = vertex_shader(vertex_payloads[i]);
            // 用转换后的视点坐标构建新三角形
            newtriangle.vertex[i] = vertex_payloads[i].position;
            newtriangle.normal[i] = vertex_payloads[i].normal;
        }

        rasterize_triangle_mt(newtriangle, worldspace_pos, material, lights, camera);
    }
}

// Screen space rasterization
// void Rasterizer::rasterize_triangle_mt(const Triangle& t, const std::array<Vector3f, 3>&
// world_pos,
//                                        GL::Material material, const std::list<Light>& lights,
//                                        Camera camera)
// {
//     // 传参的三角形的坐标已经经过vertex shader变换为视点坐标
//     float min_x = std::min(std::min(t.vertex[0].x(), t.vertex[1].x()), t.vertex[2].x());
//     float min_y = std::min(std::min(t.vertex[0].y(), t.vertex[1].y()), t.vertex[2].y());
//     float max_x = std::max(std::max(t.vertex[0].x(), t.vertex[1].x()), t.vertex[2].x());
//     float max_y = std::max(std::max(t.vertex[0].y(), t.vertex[1].y()), t.vertex[2].y());
//     Vector3f normal[3];

//     for (int i = 0; i < 3; ++i) {
//         Vector4f tmp_normal =
//             Uniforms::inv_trans_M *
//             Vector4f(t.normal[i].x(), t.normal[i].y(), t.normal[i].z(), 1.0f).normalized();
//         normal[i] = tmp_normal.block<3, 1>(0, 0);
//     }
//     const int num_threads  = 8;
//     const int block_size_x = (max_x - min_x) / num_threads;
//     // const int block_size_y = (max_y - min_y) / num_threads;
//     std::vector<std::thread> threads(num_threads);

//     auto Thread = [&](int start_x, int end_x) {
//         for (int x = start_x; x <= end_x; ++x) {
//             for (int y = min_y; y <= max_y; ++y) {
//                 // 像素点的中心点在格子中间，需要加0.5
//                 // if current pixel is in current triange:
//                 if (inside_triangle(x + 0.5, y + 0.5, t.vertex)) {
//                     //mtx.lock();
//                     auto [alpha, beta, gamma] = compute_barycentric_2d(x, y, t.vertex);
//                     // 1. interpolate depth(use projection correction algorithm)
//                     float Zt = 1.0 / (alpha / t.vertex[0].w() + beta / t.vertex[1].w() +
//                                       gamma / t.vertex[2].w());
//                     float It = alpha * t.vertex[0].z() / t.vertex[0].w() +
//                                beta * t.vertex[1].z() / t.vertex[1].w() +
//                                gamma * t.vertex[2].z() / t.vertex[2].w();
//                     It *= Zt;
//                     //mtx.unlock();
//                     // 2. interpolate vertex positon & normal(use
//                     // function:interpolate())

//                     // depth
//                     // buffer也可以叫做z-buffer，用于判断像素点相较于观察点的前后关系

//                     if (It < depth_buf[get_index(x, y)]) {
//                         mtx.lock();
//                         depth_buf[get_index(x, y)] = It;
//                         mtx.unlock();
//                         Vector3f t_weight = {t.vertex[0].w(), t.vertex[1].w(), t.vertex[2].w()};
//                         Vector3f vertex_interpolate =
//                             interpolate(alpha, beta, gamma, world_pos[0], world_pos[1],
//                                         world_pos[2], t_weight, Zt);

//                         Vector3f normal_interpolate = interpolate(
//                             alpha, beta, gamma, normal[0], normal[1], normal[2], t_weight, Zt);
//                         FragmentShaderPayload payload(vertex_interpolate.normalized(),
//                                                       normal_interpolate.normalized());
//                         Vector3f pixel_color = fragment_shader(payload, material, lights,
//                         camera);
//                         // process++;
//                         // std::cout<<process<<std::endl;

//                         Eigen::Vector2i p(x, y);
//                         //mtx.lock();
//                         set_pixel(p, pixel_color);
//                         //mtx.unlock();
//                     }

//                 }
//             }
//         }
//     };
//     // auto Thread = [&](int start_x, int end_x, int start_y, int end_y) {};

//     for (int i = 0; i < num_threads; ++i) {
//         int start_x = min_x + i * block_size_x;
//         int end_x   = min_x + (i + 1) * block_size_x;
//         // int start_y = min_y + i * block_size_y;
//         // int end_y   = min_y + (i + 1) * block_size_y;
//         // threads[i]  = std::thread(Thread, start_x, end_x, start_y, end_y);
//         threads[i] = std::thread(Thread, start_x, end_x);
//     }
//     for (int i = 0; i < num_threads; ++i) threads[i].join();

//     //mtx.lock();
//     // 串行渲染剩下没有被渲染的部分
//     for (int x = min_x + num_threads * block_size_x; x <= max_x; ++x) {
//         for (int y = min_y; y <= max_y; ++y) {
//             // 像素点的中心点在格子中间，需要加0.5
//             // if current pixel is in current triange:
//             if (inside_triangle(x + 0.5, y + 0.5, t.vertex)) {
//                 auto [alpha, beta, gamma] = compute_barycentric_2d(x, y, t.vertex);
//                 // 1. interpolate depth(use projection correction algorithm)
//                 float Zt = 1.0 / (alpha / t.vertex[0].w() + beta / t.vertex[1].w() +
//                                   gamma / t.vertex[2].w());
//                 float It = alpha * t.vertex[0].z() / t.vertex[0].w() +
//                            beta * t.vertex[1].z() / t.vertex[1].w() +
//                            gamma * t.vertex[2].z() / t.vertex[2].w();
//                 It *= Zt;
//                 // 2. interpolate vertex positon & normal(use
//                 // function:interpolate())

//                 // depth
//                 // buffer也可以叫做z-buffer，用于判断像素点相较于观察点的前后关系

//                 if (It < depth_buf[get_index(x, y)]) {
//                     depth_buf[get_index(x, y)] = It;
//                     Vector3f t_weight = {t.vertex[0].w(), t.vertex[1].w(), t.vertex[2].w()};
//                     Vector3f vertex_interpolate = interpolate(
//                         alpha, beta, gamma, world_pos[0], world_pos[1], world_pos[2], t_weight,
//                         Zt);

//                     Vector3f normal_interpolate = interpolate(alpha, beta, gamma, normal[0],
//                                                               normal[1], normal[2], t_weight,
//                                                               Zt);
//                     FragmentShaderPayload payload(vertex_interpolate.normalized(),
//                                                   normal_interpolate.normalized());
//                     Vector3f pixel_color = fragment_shader(payload, material, lights, camera);
//                     // process++;
//                     // std::cout<<process<<std::endl;

//                     Eigen::Vector2i p(x, y);
//                     set_pixel(p, pixel_color);
//                 }
//             }
//         }
//     }
//     //mtx.unlock();
// }

void Rasterizer::rasterize_triangle_mt(const Triangle& t, const std::array<Vector3f, 3>& world_pos,
                                       GL::Material material, const std::list<Light>& lights,
                                       Camera camera)
{
    float min_x = std::min(std::min(t.vertex[0].x(), t.vertex[1].x()), t.vertex[2].x());
    float min_y = std::min(std::min(t.vertex[0].y(), t.vertex[1].y()), t.vertex[2].y());
    float max_x = std::max(std::max(t.vertex[0].x(), t.vertex[1].x()), t.vertex[2].x());
    float max_y = std::max(std::max(t.vertex[0].y(), t.vertex[1].y()), t.vertex[2].y());
    Vector3f normal[3];
    for (int i = 0; i < 3; ++i) {
        Vector4f tmp_normal =
            Uniforms::inv_trans_M *
            Vector4f(t.normal[i].x(), t.normal[i].y(), t.normal[i].z(), 1.0f).normalized();
        normal[i] = tmp_normal.block<3, 1>(0, 0);
    }

    #pragma omp parallel for
    for (int x = min_x; x <= max_x; ++x) {
        for (int y = min_y; y <= max_y; ++y) {

            if (inside_triangle(x + 0.5, y + 0.5, t.vertex)) {
                auto [alpha, beta, gamma] = compute_barycentric_2d(x, y, t.vertex);

                float Zt = 1.0 / (alpha / t.vertex[0].w() + beta / t.vertex[1].w() +
                                  gamma / t.vertex[2].w());
                float It = alpha * t.vertex[0].z() / t.vertex[0].w() +
                           beta * t.vertex[1].z() / t.vertex[1].w() +
                           gamma * t.vertex[2].z() / t.vertex[2].w();
                It *= Zt;

                if (It < depth_buf[get_index(x, y)]) {
                    depth_buf[get_index(x, y)] = It;
                    Vector3f t_weight = {t.vertex[0].w(), t.vertex[1].w(), t.vertex[2].w()};

                    Vector3f vertex_interpolate = interpolate(
                        alpha, beta, gamma, world_pos[0], world_pos[1], world_pos[2], t_weight, Zt);

                    Vector3f normal_interpolate = interpolate(alpha, beta, gamma, normal[0],
                                                              normal[1], normal[2], t_weight, Zt);
                    FragmentShaderPayload payload(vertex_interpolate.normalized(),
                                                  normal_interpolate.normalized());
                    Vector3f pixel_color = fragment_shader(payload, material, lights, camera);
                    Eigen::Vector2i p(x, y);
                    set_pixel(p, pixel_color);
                }
                // interpolate
            }
        }
    }
}