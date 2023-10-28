#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <vector>
#include <optional>
#include <iostream>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "render_engine.h"
#include "../scene/light.h"
#include "../utils/math.hpp"
#include "../utils/ray.h"
#include "../utils/logger.h"

using std::chrono::steady_clock;
using duration   = std::chrono::duration<float>;
using time_point = std::chrono::time_point<steady_clock, duration>;
using Eigen::Vector3f;

// 最大的反射次数
constexpr int MAX_DEPTH        = 5;
constexpr float INFINITY_FLOAT = std::numeric_limits<float>::max();
// 考虑物体与光线相交点的偏移值
constexpr float EPSILON = 0.00001f;

// 当前物体的材质类型，根据不同材质类型光线会有不同的反射情况
enum class MaterialType
{
    DIFFUSE_AND_GLOSSY,
    REFLECTION
};

// 显示渲染的进度条
void UpdateProgress(float progress)
{
    int barwidth = 70;
    std::cout << "[";
    int pos = static_cast<int>(barwidth * progress);
    for (int i = 0; i < barwidth; i++) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "]" << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

WhittedRenderer::WhittedRenderer(RenderEngine& engine)
    : width(engine.width), height(engine.height), n_threads(engine.n_threads), use_bvh(false),
      rendering_res(engine.rendering_res)
{
    logger = get_logger("Whitted Renderer");
}

// whitted-style渲染的实现
void WhittedRenderer::render(Scene& scene)
{
    time_point begin_time = steady_clock::now();
    width                 = std::floor(width);
    height                = std::floor(height);

    // initialize frame buffer
    std::vector<Vector3f> framebuffer(static_cast<size_t>(width * height));
    for (auto& v : framebuffer) {
        v = Vector3f(0.0f, 0.0f, 0.0f);
    }

    int idx = 0;
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            // generate ray
            float depth = 1 / scene.camera.position.norm();
            Ray ray = generate_ray(static_cast<int>(width), static_cast<int>(height), i, j,
                                   scene.camera, depth);
            //std::cout << ray.origin << std::endl;                       
            // cast ray
            framebuffer[idx++] = cast_ray(ray, scene, 0);
        }
        UpdateProgress(j / height);
    }
    // save result to whitted_res.ppm
    FILE* fp = fopen("whitted_res.ppm", "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", (int)width, (int)height);
    static unsigned char color_res[3];
    rendering_res.clear();
    for (long unsigned int i = 0; i < framebuffer.size(); i++) {
        color_res[0] = static_cast<unsigned char>(255 * clamp(0.f, 1.f, framebuffer[i][0]));
        color_res[1] = static_cast<unsigned char>(255 * clamp(0.f, 1.f, framebuffer[i][1]));
        color_res[2] = static_cast<unsigned char>(255 * clamp(0.f, 1.f, framebuffer[i][2]));
        fwrite(color_res, 1, 3, fp);
        rendering_res.push_back(color_res[0]);
        rendering_res.push_back(color_res[1]);
        rendering_res.push_back(color_res[2]);
    }
    time_point end_time         = steady_clock::now();
    duration rendering_duration = end_time - begin_time;
    logger->info("rendering takes {:.6f} seconds", rendering_duration.count());
}

// 菲涅尔定理计算反射光线
float WhittedRenderer::fresnel(const Vector3f& I, const Vector3f& N, const float& ior)
{
    // 计算入射角和法线角
    // 从矢量角度来看，入射与法线之间夹角应该始终为钝角吧，所以cosi应该始终为负数
    // 角度cos值
    float cosi = I.dot(N) / (I.norm() * N.norm());
    // ni: 入射介质折射率, nt: 出射介质折射率
    float ni = 1.0f;
    float nt = ior;

    // 根据折射定律计算折射角，etai*sini = nt*sint
    float sint = ni / nt * sqrt(1.0f - cosi * cosi);
    float cost = sqrt(1.0f - sint * sint);
    cosi       = abs(cosi); // 取绝对值
    float Rs   = ((nt * cosi) - (ni * cost)) / ((nt * cosi) + (ni * cost));
    float Rp   = ((ni * cosi) - (nt * cost)) / ((ni * cosi) + (nt * cost));
    return (Rs * Rs + Rp * Rp) / 2.0f;
}

// 如果相交返回Intersection结构体，如果不相交则返回false
std::optional<std::tuple<Intersection, GL::Material>> WhittedRenderer::trace(const Ray& ray,
                                                                             const Scene& scene)
{

    std::optional<Intersection> payload;
    float tNear = INFINITY_FLOAT;
    //payload->t = INFINITY_FLOAT;
    // Eigen::Matrix4f M;
    GL::Material material;
    for (const auto& group : scene.groups) {
        for (const auto& object : group->objects) {
            auto intersection = naive_intersect(ray, object->mesh, object->model());
            if (intersection.has_value() && intersection->t < tNear) {
                // std::cout << "has" << std::endl;
                payload  = intersection;
                material = object->mesh.material;
                tNear = intersection->t;
            }
            // 有值并且交点更近
            // if (intersection.has_value() && (intersection->t < payload->t)) {

            // }
            // if use bvh(exercise 2.4): use object->bvh->intersect
            // else(exercise 2.3): use naive_intersect()
            // pay attention to the range of payload->t
        }
    }

    if (!payload.has_value()) {
        return std::nullopt;
    }
    // std::cout << "return" << std::endl;
    return std::make_tuple(payload.value(), material);
}

// Whitted-style的光线传播算法实现
Vector3f WhittedRenderer::cast_ray(const Ray& ray, const Scene& scene, int depth)
{
    if (depth >= MAX_DEPTH) {
        return Vector3f(0.0f, 0.0f, 0.0f);
    }
    // initialize hit color
    Vector3f hitcolor = RenderEngine::background_color;
    // Vector3f hitcolor = {255, 255, 255};
    // get the result of trace()
    auto result = trace(ray, scene);
    //Vector3f amibient(0.1, 0.1, 0.1);
    Vector3f diffuse(0, 0, 0);
    Vector3f specular(0, 0, 0);
    // std::cout << "0" << std::endl;
    if (result.has_value()) {
        // std::cout << "1" << std::endl;
        auto [intersection, material] = result.value();
        // 由于不知道三角形顶点坐标，用射线t*d + o代表交点
        Vector3f hitPoint = intersection.t * ray.direction + ray.origin;
        // material.shiness<1000 时为 diffuse，否则为 relection
        if (material.shininess >= 1000) {
            // 不知道ior是多少，自己手动设置的
            float kr = fresnel(ray.direction, intersection.normal, 0.8);
            Ray reflection;
            reflection.direction =
                reflect(ray.direction.normalized(), intersection.normal.normalized());
            // 使用出射光线方向与法线点乘，小于零为内侧，大于零为外侧

            if (reflection.direction.dot(intersection.normal) < 0) {
                reflection.origin = hitPoint - intersection.normal * EPSILON;
            } else {
                reflection.origin = hitPoint + intersection.normal * EPSILON;
            }
            hitcolor = cast_ray(reflection, scene, depth + 1) * kr;
        } 
        else {
            //Vector3f ka = material.ambient;
            //amibient    = ka.cwiseProduct(amibient);
            for (const Light& light : scene.lights) {

                Ray shadow_ray;
                shadow_ray.origin = hitPoint;

                // direction是从光源指向交点
                shadow_ray.direction = (light.position - shadow_ray.origin).normalized();
                // Vector3f reflection  = reflect(-shadow_ray.direction, intersection.normal);
                //  判断光源到交点中是否存在其他交点。如果存在，也就被遮挡，不着色；如果不存在交点，那就着色
                if(shadow_ray.direction.dot(intersection.normal) < 0)
                {
                    shadow_ray.origin = hitPoint - intersection.normal * EPSILON;
                }else
                {
                    shadow_ray.origin = hitPoint + intersection.normal * EPSILON;
                }
                auto shadow_result = trace(shadow_ray, scene);
                if (!shadow_result.has_value()) {
                    Vector3f view_dir = (scene.camera.position - hitPoint).normalized();

                    Vector3f kd              = material.diffuse;
                    Vector3f ks              = material.specular;
                    Vector3f light_intensity = {light.intensity, light.intensity, light.intensity};

                    float distance     = (light.position - hitPoint).norm();
                    float constant     = 1;
                    // float range              = 100;
                    float linear    = 0.35;
                    float quadratic = 0.5;
                    float attenuation_number =
                        1.0f / (constant + linear * distance + quadratic * distance * distance);
                    Vector3f attenuation = light_intensity * attenuation_number;
                    Vector3f light_dir = (light.position - hitPoint).normalized();
                    Vector3f half_vec  = (light_dir + view_dir).normalized();

                    // 光衰减不要了

                    diffuse += kd.cwiseProduct(attenuation) *
                               std::max(0.0f, light_dir.dot(intersection.normal));

                    specular += ks.cwiseProduct(attenuation) *
                                std::pow(std::max(0.0f, intersection.normal.dot(half_vec)),
                                         material.shininess);
                    // std::cout << "light_intensity" << light.intensity << std::endl;
                    // std::cout << "light_dir" << light_dir << std::endl;
                    // std::cout << "normal" << intersection.normal << std::endl;
                    // std::cout << "dot" << light_dir.dot(intersection.normal) << std::endl;
                    // std::cout << "diffuse" << diffuse << std::endl;
                    // std::cout << "spe" << specular << std::endl;
                }
            }
            hitcolor = (diffuse + specular);
        }

        // else{
        //     std::cout << "1" << std::endl;
        // }
        // if result.has_value():
        // 1.judge the material_type
        // 2.if REFLECTION:
        //(1)use fresnel() to get kr
        //(2)hitcolor = cast_ray*kr
        // if DIFFUSE_AND_GLOSSY:
        //(1)compute shadow result using trace()
        //(2)hitcolor = diffuse*kd + specular*ks
    }
    return hitcolor;
}
