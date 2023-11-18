#include "bvh.h"

#include <cassert>
#include <iostream>
#include <optional>

#include <Eigen/Geometry>
#include "formatter.hpp"
#include <spdlog/spdlog.h>

#include "math.hpp"

using Eigen::Vector3f;
using Eigen::Vector4f;
using std::optional;
using std::vector;

BVHNode::BVHNode() : left(nullptr), right(nullptr), face_idx(0)
{
}

BVH::BVH(const GL::Mesh& mesh) : root(nullptr), mesh(mesh)
{
}

// 建立bvh，将需要建立BVH的图元索引初始化
void BVH::build()
{
    if (mesh.faces.count() == 0) {
        root = nullptr;
        return;
    }

    primitives.resize(mesh.faces.count());
    for (size_t i = 0; i < mesh.faces.count(); i++) primitives[i] = i;

    root = recursively_build(primitives);
    return;
}
// 删除bvh
void BVH::recursively_delete(BVHNode* node)
{
    if (node == nullptr)
        return;
    recursively_delete(node->left);
    recursively_delete(node->right);
    delete node;
    node = nullptr;
}
// 统计BVH树建立的节点个数
size_t BVH::count_nodes(BVHNode* node)
{
    if (node == nullptr)
        return 0;
    else
        return count_nodes(node->left) + count_nodes(node->right) + 1;
}

// 递归建立BVH
BVHNode* BVH::recursively_build(vector<size_t> faces_idx)
{
    BVHNode* node = new BVHNode();

    AABB aabb;
    for (size_t i = 0; i < faces_idx.size(); i++) {
        aabb = union_AABB(aabb, get_aabb(mesh, faces_idx[i]));
    }
    if (faces_idx.size() == 1) {
        node->face_idx = faces_idx[0];
        node->left     = nullptr;
        node->right    = nullptr;
        node->aabb     = get_aabb(mesh, faces_idx[0]);
        return node;
    } else if (faces_idx.size() == 2) {
        node->left  = recursively_build(std::vector{faces_idx[0]});
        node->right = recursively_build(std::vector{faces_idx[1]});
        node->aabb  = union_AABB(node->left->aabb, node->right->aabb);
    } else {
        AABB centroid_aabb;
        for (size_t i = 0; i < faces_idx.size(); ++i) {
            centroid_aabb = union_AABB(centroid_aabb, get_aabb(mesh, faces_idx[i]).centroid());
        }
        int longest_dimension = centroid_aabb.max_extent();
        // auto m = this->mesh;
        switch (longest_dimension) {
        case 0:
            std::sort(faces_idx.begin(), faces_idx.end(), [&](size_t i, size_t j) {
                return get_aabb(mesh, i).centroid().x() < get_aabb(mesh, j).centroid().x();
            });
            break;
        case 1:
            std::sort(faces_idx.begin(), faces_idx.end(), [&](size_t i, size_t j) {
                return get_aabb(mesh, i).centroid().y() < get_aabb(mesh, j).centroid().y();
            });
            break;
        case 2:
            std::sort(faces_idx.begin(), faces_idx.end(), [&](size_t i, size_t j) {
                return get_aabb(mesh, i).centroid().z() < get_aabb(mesh, j).centroid().z();
            });
            break;
        default: break;
        }
        auto faces_begin = faces_idx.begin();
        auto faces_end   = faces_idx.end();
        auto faces_mid   = faces_idx.begin() + faces_idx.size() / 2;
        auto left_tree   = vector<size_t>(faces_begin, faces_mid);
        // 防止中间的物体重复索引，虽然但是好像不太需要
        auto right_tree = vector<size_t>(faces_mid, faces_end);
        node->left      = recursively_build(left_tree);
        node->right     = recursively_build(right_tree);
        node->aabb      = union_AABB(node->left->aabb, node->right->aabb);
    }
    // if faces_idx.size()==1: return node;
    // if faces_idx.size()==2: recursively_build() & union_AABB(node->left->aabb,
    // node->right->aabb); else:
    // choose the longest dimension among x,y,z
    // devide the primitives into two along the longest dimension
    // recursively_build() & union_AABB(node->left->aabb, node->right->aabb)
    return node;
}
// 使用BVH求交
optional<Intersection> BVH::intersect(const Ray& ray, [[maybe_unused]] const GL::Mesh& mesh,
                                      const Eigen::Matrix4f obj_model)
{
    model = obj_model;
    optional<Intersection> isect;
    if (!root) {
        isect = std::nullopt;
        return isect;
    }

    Eigen::Matrix4f model_inv = this->model.inverse();
    Ray trans_ray;
    trans_ray.direction =
        (model_inv * Vector4f(ray.direction.x(), ray.direction.y(), ray.direction.z(), 0.0f))
            .head<3>()
            .normalized();
    trans_ray.origin =
        (model_inv * Vector4f(ray.origin.x(), ray.origin.y(), ray.origin.z(), 1.0f)).head<3>();
    isect = ray_node_intersect(root, trans_ray);
    if (isect.has_value()) {
        Vector3f world_intersection =
            (model * (isect->t * trans_ray.direction + trans_ray.origin).homogeneous()).head<3>();
        // Vector3f world_intersection;
        float dis     = (world_intersection - ray.origin).norm();
        isect->t      = dis;
        isect->normal = (model_inv.transpose() *
                         Vector4f(isect->normal.x(), isect->normal.y(), isect->normal.z(), 0.0f))
                            .head<3>();
    }
    return isect;
}
// 发射的射线与当前节点求交，并递归获取最终的求交结果
optional<Intersection> BVH::ray_node_intersect(BVHNode* node, const Ray& ray) const
{
    // The node intersection is performed in the model coordinate system.
    // Therefore, the ray needs to be transformed into the model coordinate system.
    // The intersection attributes returned are all in the model coordinate system.
    // Therefore, They are need to be converted to the world coordinate system.
    // If the model shrinks, the value of t will also change.
    // The change of t can be solved by intersection point changing simultaneously
    Vector3f ray_inv_dir(1.0f / ray.direction.x(), 1.0f / ray.direction.y(),
                         1.0f / ray.direction.z());
    std::array<int, 3> dir_is_neg;
    dir_is_neg[0] = int(ray.direction.x() >= 0);
    dir_is_neg[1] = int(ray.direction.y() >= 0);
    dir_is_neg[2] = int(ray.direction.z() >= 0);

    if ((node == nullptr) || !(node->aabb.intersect(ray, ray_inv_dir, dir_is_neg)))
        return std::nullopt;

    optional<Intersection> isect;
    if (node->left == nullptr && node->right == nullptr) {
        isect = ray_triangle_intersect(ray, mesh, node->face_idx);
        return isect;
    } else {
        optional<Intersection> right_isect, left_isect;
        left_isect  = ray_node_intersect(node->left, ray);
        right_isect = ray_node_intersect(node->right, ray);

        if (left_isect.has_value() && right_isect.has_value()) {
            return (left_isect->t < right_isect->t) ? left_isect : right_isect;
        }
        // 如果只有一个子树有交点，返回该交点
        return left_isect.has_value() ? left_isect : right_isect;
    }
}
