#include <array>
#include <limits>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mutex>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <spdlog/spdlog.h>

#include "rasterizer.h"
#include "triangle.h"
#include "render_engine.h"
#include "../utils/math.hpp"

using Eigen::Matrix4f;
using Eigen::Vector2i;
using Eigen::Vector3f;
using Eigen::Vector4f;
using std::fill;
using std::tuple;

double judge(Vector3f Point0, Vector3f Point1, Vector3f Point2)
{
    // 首先根据坐标计算p1p2和p1p3的向量，然后再计算叉�?
    // p1p2 向量表示�? (p2.x-p1.x,p2.y-p1.y)
    // p1p3 向量表示�? (p3.x-p1.x,p3.y-p1.y)
    return (Point1.x() - Point0.x()) * (Point2.y() - Point0.y()) -
           (Point1.y() - Point0.y()) * (Point2.x() - Point0.x());
}

// 给定坐标(x,y)以及三角形的三个顶点坐标，判�?(x,y)是否在三角形的内�?

// int -> float

bool Rasterizer::inside_triangle(int x, int y, const Vector4f* vertices)
{

    Vector3f v[3];
    for (int i = 0; i < 3; i++) v[i] = {vertices[i].x(), vertices[i].y(), 1.0};

    Vector3f p(float(x), float(y), 1.0f);

    if (judge(v[0], v[1], v[2]) < 0) {
        if (judge(v[0], v[2], p) > 0 && judge(v[2], v[1], p) > 0 && judge(v[1], v[0], p) > 0)
            return true;
        return false;
    } else {
        if (judge(v[0], v[1], p) > 0 && judge(v[1], v[2], p) > 0 && judge(v[2], v[0], p) > 0)
            return true;
        return false;
    }
}

// 给定坐标(x,y)以及三角形的三个顶点坐标，计�?(x,y)对应的重心坐标[alpha, beta, gamma]
tuple<float, float, float> Rasterizer::compute_barycentric_2d(float x, float y, const Vector4f* v)
{
    float c1 = 0.f, c2 = 0.f, c3 = 0.f;
    Vector3f tri[3];
    for (int i = 0; i < 3; i++) tri[i] = {v[i].x(), v[i].y(), 1.0};

    c1 = ((x - tri[1].x()) * (tri[2].y() - tri[1].y()) -
          (y - tri[1].y()) * (tri[2].x() - tri[1].x())) /
         ((tri[0].x() - tri[1].x()) * (tri[2].y() - tri[1].y()) -
          (tri[0].y() - tri[1].y()) * (tri[2].x() - tri[1].x()));
    c2 = ((x - tri[2].x()) * (tri[0].y() - tri[2].y()) -
          (y - tri[2].y()) * (tri[0].x() - tri[2].x())) /
         ((tri[1].x() - tri[2].x()) * (tri[0].y() - tri[2].y()) -
          (tri[1].y() - tri[2].y()) * (tri[0].x() - tri[2].x()));
    c3 = 1 - c1 - c2;
    return {c1, c2, c3};
}

// 对当前渲染物体的所有三角形面片进行遍历，进行几何变换以及光栅化
void Rasterizer::draw(const std::vector<Triangle>& TriangleList, const GL::Material& material,
                      const std::list<Light>& lights, const Camera& camera)
{
    // iterate over all triangles in TriangleList
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
            // 用转换后的视点坐标构建新三角�?
            newtriangle.vertex[i] = vertex_payloads[i].position;
            newtriangle.normal[i] = vertex_payloads[i].normal;
        }

        rasterize_triangle(newtriangle, worldspace_pos, material, lights, camera);
    }
}

// 对顶点的某一属性插�?
Vector3f Rasterizer::interpolate(float alpha, float beta, float gamma, const Eigen::Vector3f& vert1,
                                 const Eigen::Vector3f& vert2, const Eigen::Vector3f& vert3,
                                 const Eigen::Vector3f& weight, const float& Z)
{
    Vector3f interpolated_res;
    for (int i = 0; i < 3; i++) {
        interpolated_res[i] = alpha * vert1[i] / weight[0] + beta * vert2[i] / weight[1] +
                              gamma * vert3[i] / weight[2];
    }
    interpolated_res *= Z;
    return interpolated_res;
}

// 对当前三角形进行光栅化
void Rasterizer::rasterize_triangle(const Triangle& t, const std::array<Vector3f, 3>& world_pos,
                                    GL::Material material, const std::list<Light>& lights,
                                    Camera camera)
{
    // 传参的三角形的坐标已经经过vertex shader变换为视点坐�?
    // auto v = to_vec4(t);
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

    // these lines above are just for compiling and can be deleted
    // float alpha, beta, gamma;
    for (int x = min_x; x <= max_x; ++x) {
        for (int y = min_y; y <= max_y; ++y) {
            // 像素点的中心点在格子中间，需要加0.5
            // if current pixel is in current triange:
            if (inside_triangle(x + 0.5, y + 0.5, t.vertex)) {
                auto [alpha, beta, gamma] = compute_barycentric_2d(x, y, t.vertex);
                // 1. interpolate depth(use projection correction algorithm)
                float Zt = 1.0 / (alpha / t.vertex[0].w() + beta / t.vertex[1].w() +
                                  gamma / t.vertex[2].w());
                float It = alpha * t.vertex[0].z() / t.vertex[0].w() +
                           beta * t.vertex[1].z() / t.vertex[1].w() +
                           gamma * t.vertex[2].z() / t.vertex[2].w();
                It *= Zt;
                // 2. interpolate vertex positon & normal(use function:interpolate())

                // depth buffer也可以叫做z-buffer，用于判断像素点相较于观察点的前后关�?
                if (It < depth_buf[get_index(x, y)]) {
                    depth_buf[get_index(x, y)] = It;
                    // depth_buf[get_index(x, y)] = Zt;
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

// 初始化整个光栅化渲染
void Rasterizer::clear(BufferType buff)
{
    if ((buff & BufferType::Color) == BufferType::Color) {
        fill(frame_buf.begin(), frame_buf.end(), RenderEngine::background_color * 255.0f);
    }
    if ((buff & BufferType::Depth) == BufferType::Depth) {
        fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

Rasterizer::Rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

// 给定像素坐标(x,y)，计算frame buffer里对应的index
int Rasterizer::get_index(int x, int y)
{
    return (height - 1 - y) * width + x;
}

// 给定像素点以及fragement shader得到的结果，对frame buffer中对应存储位置进行赋值
void Rasterizer::set_pixel(const Vector2i& point, const Vector3f& res)
{
    int idx        = get_index(point.x(), point.y());
    frame_buf[idx] = res;
}
