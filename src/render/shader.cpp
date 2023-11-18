#include "shader.h"
#include "../utils/math.hpp"

#ifdef _WIN32
#undef min
#undef max
#endif

using Eigen::Vector3f;
using Eigen::Vector4f;

// vertex shader & fragement shader can visit
// all the static variables below from Uniforms structure
Eigen::Matrix4f Uniforms::MVP;
Eigen::Matrix4f Uniforms::inv_trans_M;
int Uniforms::width;
int Uniforms::height;

// vertex shader
VertexShaderPayload vertex_shader(const VertexShaderPayload& payload)
{
    // 输入为模型坐标系下的顶点位置和法线，输出为变换到视口空间的顶点位置（方便剪裁）和世界/相机坐标系下的法线（方便插值）
    VertexShaderPayload output_payload = payload;
    // 将顶点坐标变换到投影平面，再进行视口变换；同时将法线向量变换到相机坐标系用于后续插值

    // Vertex position transformation
    output_payload.position = Uniforms::MVP * payload.position;

    // Viewport transformation
    // 齐次除法
    output_payload.position.x() /= output_payload.position.w();
    output_payload.position.y() /= output_payload.position.w();
    output_payload.position.z() /= output_payload.position.w();
    // 屏幕映射
    output_payload.position.x() = (output_payload.position.x() + 1) * Uniforms::width / 2;
    output_payload.position.y() = (output_payload.position.y() + 1) * Uniforms::height / 2;

    // Vertex normal transformation
    // 法向量本来是vector3f, 没法与M(Matrix4f)直接相乘，所以扩展一位置0
    Eigen::Vector4f expanded_normal(payload.normal.x(), payload.normal.y(), payload.normal.z(),
                                    0.0f);
    // 取Vector4f前三个值 从(0,0)开始取大小为<3,1>的子矩阵
    expanded_normal           = Uniforms::inv_trans_M * expanded_normal;
    output_payload.normal.x() = expanded_normal.x();
    output_payload.normal.y() = expanded_normal.y();
    output_payload.normal.z() = expanded_normal.z();

    return output_payload;
}

Vector3f phong_fragment_shader(const FragmentShaderPayload& payload, GL::Material material,
                               const std::list<Light>& lights, Camera camera)
{
    Vector3f result = {0, 0, 0};

    // ka,kd,ks can be got from material.ambient,material.diffuse,material.specular
    Vector3f ka = material.ambient;
    Vector3f kd = material.diffuse;
    Vector3f ks = material.specular;

    // set ambient light intensity
    // 不知道环境光强度应该是多少
    Vector3f ambient_light_intensity(0.9, 0.9, 0.9);

    Vector3f ambient(0, 0, 0);
    Vector3f diffuse(0, 0, 0);
    Vector3f specular(0, 0, 0);
    // Light Direction
    Vector3f light_dir(0, 0, 0);
    // View Direction
    Vector3f view_dir(0, 0, 0);
    view_dir = (camera.position - payload.world_pos).normalized();
    ambient  = ka.cwiseProduct(ambient_light_intensity);
    for (const auto& light : lights) {
        Vector3f light_intensity = {light.intensity / 5, light.intensity / 5, light.intensity / 5};
        light_dir                = (light.position - payload.world_pos).normalized();

        // half_vec是角平分向量，画个图就知道其实就是(light_dir + view_dir)
        Vector3f half_vec = (light_dir + view_dir).normalized();

        float distance = (light.position - payload.world_pos).norm();
        //float distance = (payload.world_pos - light.position).norm();
        // Light Attenuation
        // 数据来源https://wiki.ogre3d.org/tiki-index.php?page=-Point+Light+Attenuation

        float constant           = 1;
        //float range              = 100;
        float linear             = 0.09;
        float quadratic          = 0.032;
        float attenuation_number = 1.0f  / (constant + linear * distance + quadratic * distance * distance); 
        Vector3f attenuation = light_intensity * attenuation_number;
        // Vector3f attenuation = light_intensity * attenuation_number;
        diffuse +=
            kd.cwiseProduct(attenuation) * std::max(0.0f, payload.world_normal.normalized().dot(light_dir));
        // diffuse +=
        //     kd.cwiseProduct(attenuation) * std::max(0.0f, payload.world_normal.dot(-light_dir));
        specular +=
            ks.cwiseProduct(attenuation) *
            std::pow(std::max(0.0f, payload.world_normal.normalized().dot(half_vec)), material.shininess);
        // specular +=
        //     ks *
        //     std::pow(std::max(0.0f, payload.world_normal.dot(half_vec)), material.shininess);
    }

    // set rendering result max threshold to 255
    // 255 RGB
    result = (ambient + diffuse + specular) * 255.f;
    if (result.x() > 255) {result.x() = 255;}
    if (result.y() > 255) {result.y() = 255;}
    if (result.z() > 255) {result.z() = 255;}
    return result;
}
