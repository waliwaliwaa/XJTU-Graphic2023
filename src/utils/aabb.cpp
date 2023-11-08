#include "aabb.h"

#include <array>

#include <Eigen/Geometry>

using Eigen::Vector3f;
constexpr float EPSILON = 0.00001f;
AABB::AABB()
{
    float min_num = std::numeric_limits<float>::lowest();
    float max_num = std::numeric_limits<float>::max();
    p_max         = Vector3f(min_num, min_num, min_num);
    p_min         = Vector3f(max_num, max_num, max_num);
}

AABB::AABB(const Vector3f& p1, const Vector3f& p2)
{
    p_min = Vector3f(fmin(p1.x(), p2.x()), fmin(p1.y(), p2.y()), fmin(p1.z(), p2.z()));
    p_max = Vector3f(fmax(p1.x(), p2.x()), fmax(p1.y(), p2.y()), fmax(p1.z(), p2.z()));
}
// 返回AABB的對角线长度
Vector3f AABB::diagonal() const
{
    return p_max - p_min;
}
// 返回AABB最长的一维
int AABB::max_extent() const
{
    Vector3f d = diagonal();
    if (d.x() > d.y() && d.x() > d.z())
        return 0;
    else if (d.y() > d.z())
        return 1;
    else
        return 2;
}
// 返回AABB的中心点
Vector3f AABB::centroid()
{
    return 0.5 * p_min + 0.5 * p_max;
}
// 判断当前射线是否与当前AABB相交
bool AABB::intersect(const Ray& ray, const Vector3f& inv_dir, const std::array<int, 3>& dir_is_neg)
{
    Vector3f t_min = (p_min - ray.origin).cwiseProduct(inv_dir);
    Vector3f t_max = (p_max - ray.origin).cwiseProduct(inv_dir);
    Vector3f dir   = ray.direction;
    if (dir_is_neg[0] == 0)
        std::swap(t_min.x(), t_max.x());
    if (dir_is_neg[1] == 0)
        std::swap(t_min.y(), t_max.y());
    if (dir_is_neg[2] == 0)
        std::swap(t_min.z(), t_max.z());
    float t_enter = 0.0f, t_exit = 0.0f;
    t_enter = std::max(t_min.x(), std::max(t_min.y(), t_min.z()));
    t_exit  = std::min(t_max.x(), std::min(t_max.y(), t_max.z()));
    if(t_enter - t_exit > EPSILON) return false;
    return (t_enter >=0 || t_exit >= 0);


    // const float eps = 1e-6f;
    // // t_min 和 t_max 分别表示沿射线的最近和最远的交点
    // float t_min = (p_min.x() - ray.origin.x()) * inv_dir.x();
    // float t_max = (p_max.x() - ray.origin.x()) * inv_dir.x();

    // if (dir_is_neg[0]) {
    //     std::swap(t_min, t_max);
    // }

    // float ty_min = (p_min.y() - ray.origin.y()) * inv_dir.y();
    // float ty_max = (p_max.y() - ray.origin.y()) * inv_dir.y();

    // if (dir_is_neg[1]) {
    //     std::swap(ty_min, ty_max);
    // }

    // // 更新 t_min 和 t_max 以包含 y 轴的交点
    // t_min = std::max(t_min, ty_min);
    // t_max = std::min(t_max, ty_max);

    // // 如果 t_min 大于 t_max，说明没有交点
    // if (t_min - t_max > eps) {
    //     return false;
    // }

    // // 对 z 轴重复上述过程
    // float tz_min = (p_min.z() - ray.origin.z()) * inv_dir.z();
    // float tz_max = (p_max.z() - ray.origin.z()) * inv_dir.z();

    // if (dir_is_neg[2]) {
    //     std::swap(tz_min, tz_max);
    // }

    // t_min = std::max(t_min, tz_min);
    // t_max = std::min(t_max, tz_max);

    // // 如果 t_min 大于 t_max，说明没有交点
    // if (t_min - t_max > eps) {
    //     return false;
    // }

    // // 如果射线的起点在包围盒内部，t_min 可能会是负数，我们也认为这是相交的
    // return t_min >= 0.0f || t_max >= 0.0f;
}
// 获取当前图元对应AABB
AABB get_aabb(const GL::Mesh& mesh, size_t face_idx)
{
    std::array<size_t, 3> face = mesh.face(face_idx);
    std::array<Vector3f, 3> v;
    v[0] = mesh.vertex(face[0]).homogeneous().topRows(3); // a
    v[1] = mesh.vertex(face[1]).homogeneous().topRows(3); // b
    v[2] = mesh.vertex(face[2]).homogeneous().topRows(3); // c
    return union_AABB(AABB(v[0], v[1]), v[2]);
}
// 将两个AABB用一个AABB包住
AABB union_AABB(const AABB& b1, const AABB& b2)
{
    AABB ret;
    ret.p_min = Vector3f(fmin(b1.p_min.x(), b2.p_min.x()), fmin(b1.p_min.y(), b2.p_min.y()),
                         fmin(b1.p_min.z(), b2.p_min.z()));
    ret.p_max = Vector3f(fmax(b1.p_max.x(), b2.p_max.x()), fmax(b1.p_max.y(), b2.p_max.y()),
                         fmax(b1.p_max.z(), b2.p_max.z()));
    return ret;
}
// 将一个AABB和一个点用一个AABB包住
AABB union_AABB(const AABB& b, const Vector3f& p)
{
    AABB ret;
    ret.p_min =
        Vector3f(fmin(b.p_min.x(), p.x()), fmin(b.p_min.y(), p.y()), fmin(b.p_min.z(), p.z()));
    ret.p_max =
        Vector3f(fmax(b.p_max.x(), p.x()), fmax(b.p_max.y(), p.y()), fmax(b.p_max.z(), p.z()));
    return ret;
}
