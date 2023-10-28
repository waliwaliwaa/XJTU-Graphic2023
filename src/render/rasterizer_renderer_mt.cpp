#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <vector>
#include <chrono>
#include <mutex>
#include <thread>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <spdlog/spdlog.h>

#include "rasterizer.h"
#include "render_engine.h"
#include "shader.h"
#include "triangle.h"
#include "../scene/light.h"

using std::chrono::steady_clock;
using duration   = std::chrono::duration<float>;
using time_point = std::chrono::time_point<steady_clock, duration>;
using Eigen::Vector3f;
using Eigen::Vector4f;

void RasterizerRenderer::render_mt(const Scene& scene)
{
    int thread_num = 8;
    std::vector<std::thread> threads(thread_num);
    Rasterizer r(static_cast<int>(width), static_cast<int>(height));

    // choose the active shader for fragment shader
    std::function<Vector3f(FragmentShaderPayload, GL::Material, const std::list<Light>&, Camera)>
        active_shader = phong_fragment_shader;

    r.vertex_shader   = vertex_shader;
    r.fragment_shader = active_shader;

    // clear Color Buffer & Depth Buffer & rendering_res
    r.clear(BufferType::Color | BufferType::Depth);
    this->rendering_res.clear();

    time_point begin_time = steady_clock::now();
    //-----------------------------------------------------------------------------------------------
    //
    // multi-thread render_mt should be completed here
    for (const auto& group : scene.groups) {
        // int group_size = group->objects.size();
        // int thread_num = 8;
        // int block_size = group_size / 8;
        // std::vector<std::thread> threads(thread_num);
        Camera cam = scene.camera;
        // set r.view & r.projection
        r.view       = cam.view();
        r.projection = cam.projection();
        
        for (const auto& object : group->objects) {
            // set r.model
            r.model = object->model();
            // set Uniforms for vertex shader
            Uniforms::MVP         = r.projection * r.view * r.model;
            Uniforms::inv_trans_M = r.model.inverse().transpose();
            Uniforms::width       = r.width;
            Uniforms::height      = r.height;
            // input object->mesh's vertices & faces & normals data
            std::vector<Triangle> TriangleList;
            const std::vector<float>& vertices     = object->mesh.vertices.data;
            const std::vector<unsigned int>& faces = object->mesh.faces.data;
            const std::vector<float>& normals      = object->mesh.normals.data;
            // #pragma omp parallel for
            int block_size = faces.size() / thread_num;
            auto Thread    = [&](int start, int end) {
                for (int i = start; i < end; i += 3) {
                    // set triangle list(vertex & normal)
                    TriangleList.emplace_back();
                    Triangle& t = *TriangleList.rbegin();
                    for (int j = 0; j < 3; j++) {
                        unsigned int idx = faces[i + j];
                        t.vertex[j]      = Vector4f(vertices[3 * idx], vertices[3 * idx + 1],
                                                       vertices[3 * idx + 2], 1.0f);
                        t.normal[j] =
                            Vector3f(normals[3 * idx], normals[3 * idx + 1], normals[3 * idx + 2]);
                    }
                }
            };
            for (int i = 0; i < thread_num; ++i) {
                threads[i] = std::thread(Thread, i * block_size, (i + 1) * block_size);
            }
            for (int i = 0; i < thread_num; ++i) threads[i].join();
            for (unsigned int i = thread_num * block_size; i < faces.size(); i += 3) {
                // set triangle list(vertex & normal)
                TriangleList.emplace_back();
                Triangle& t = *TriangleList.rbegin();
                for (int j = 0; j < 3; j++) {
                    unsigned int idx = faces[i + j];
                    t.vertex[j]      = Vector4f(vertices[3 * idx], vertices[3 * idx + 1],
                                                vertices[3 * idx + 2], 1.0f);
                    t.normal[j] =
                        Vector3f(normals[3 * idx], normals[3 * idx + 1], normals[3 * idx + 2]);
                }
            }
            r.draw_mt(TriangleList, object->mesh.material, scene.lights, cam);
        }
    }

    //-------------------------------------------------------------------------------------------
    time_point end_time         = steady_clock::now();
    duration rendering_duration = end_time - begin_time;

    this->logger->info("rendering ({} threads) takes {:.6f} seconds", n_threads,
                       rendering_duration.count());

    // OutImage can be saved at the working directory as .ppm
    std::ofstream output_image;
    output_image.open("rasterizer_res.ppm");
    int nx = static_cast<int>(width);
    int ny = static_cast<int>(height);
    output_image << "P3\n" << nx << ' ' << ny << "\n255\n";

    for (long unsigned int i = 0; i < r.depth_buf.size(); i++) {
        rendering_res.push_back(static_cast<unsigned char>(r.frame_buf[i].x()));
        rendering_res.push_back(static_cast<unsigned char>(r.frame_buf[i].y()));
        rendering_res.push_back(static_cast<unsigned char>(r.frame_buf[i].z()));

        output_image << int(r.frame_buf[i].x()) << ' ' << int(r.frame_buf[i].y()) << ' '
                     << int(r.frame_buf[i].z()) << '\n';
    }
}
