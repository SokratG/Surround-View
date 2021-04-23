#pragma once
#include "meshgrid.hpp"

#include <stdint.h>
#include <limits>
#include <vector>
#include <cmath>




using uint = uint32_t;
using int32 = int32_t;
constexpr static float default_center[3]{0.f}; // for default value pass to constructor - argument center



class Bowl
{
private:
    constexpr static float PI = 3.14159265359f;
    constexpr static auto epsilon = std::numeric_limits<float>::epsilon();
    constexpr static int32 _num_vertices = 3; // x, y, z
private:
    float cen[3];
    float inner_rad;
    float rad;
    float param_a, param_b, param_c;
    float hole_rad;
    bool set_hole = false;
    bool useUV = false;
public:
    Bowl(const float inner_radius, const float radius, const float a, const float b, const float c, const float center[3] = default_center)
            : inner_rad(inner_radius), rad(radius), param_a(a), param_b(b), param_c(c), hole_rad(0.f)
    {
            cen[0] = center[0];
            cen[1] = center[1];
            cen[2] = center[2];
    }

    bool generate_mesh(const float max_size_vert, std::vector<float>& vertices, std::vector<uint>& indices)
    {
            set_hole = false;
            useUV = false;
            return generate_mesh_(max_size_vert, vertices, indices);
    }
    bool generate_mesh_uv(const float max_size_vert, std::vector<float>& vertices, std::vector<uint>& indices)
    {
            set_hole = false;
            useUV = true;
            return generate_mesh_(max_size_vert, vertices, indices);
    }

    bool generate_mesh_hole(const float max_size_vert, const float hole_radius, std::vector<float>& vertices, std::vector<uint>& indices)
    {
            set_hole = true;
            useUV = false;
            hole_rad = hole_radius;
            return generate_mesh_(max_size_vert, vertices, indices);
    }
    bool generate_mesh_uv_hole(const float max_size_vert, const float hole_radius, std::vector<float>& vertices, std::vector<uint>& indices)
    {
            set_hole = true;
            useUV = true;
            hole_rad = hole_radius;
            return generate_mesh_(max_size_vert, vertices, indices);
    }


protected:
    bool generate_mesh_(const float max_size_vert, std::vector<float>& vertices, std::vector<uint>& indices)
    {
            if (fabs(param_a) <= epsilon || fabs(param_b) <= epsilon || fabs(param_c) <= epsilon)
                    return false;
            if (rad <= 0.f || inner_rad <= 0.f)
                    return false;
            if (set_hole && hole_rad <= 0.f)
                    return false;
            auto a = param_a;
            auto b = param_b;
            auto c = param_c;

            vertices.clear();

            indices.clear();


            /*
                    prepare grid mesh in polar coordinate with r - radius and theta - angle
            */
            // texture coordinates generate (u, v) [0, 1]
            std::vector<float> texture_u = meshgen::linspace(0.f, 1.f, max_size_vert);
            auto texture_v = texture_u;

            auto r = meshgen::linspace(0.0f, rad, max_size_vert); // min_size = 0.f, max_size = 100.f,
            auto theta = meshgen::linspace(0.f, 2 * PI, max_size_vert);
            auto mesh_pair = meshgen::meshgrid(r, theta);

            auto R = std::get<0>(mesh_pair);
            auto THETA = std::get<1>(mesh_pair);
            size_t grid_size = R.size();
            std::vector<float> x_grid;
            std::vector<float> y_grid;
            std::vector<float> z_grid;

            // Convert to rectangular coordinates
            // x = r*cos(theta), z = r*sin(theta), y/c = (x^2)/(a^2) + (z^2)/(b^2);
            for (int i = 0; i < grid_size; ++i) {
                    for (int j = 0; j < grid_size; ++j) {
                            auto x = R(i, j) * cos(THETA(i, j));
                            auto z = R(i, j) * sin(THETA(i, j));
                            auto y = c * (pow((x / a), 2) + pow((z / b), 2));
                            x_grid.push_back(x);
                            z_grid.push_back(z);
                            y_grid.push_back(y);
                    }
            }

            /*
                    find start level - level when disk passes from to elliptic paraboloid
            */
            auto min_y = 0.f;
            for (int i = 0; i < grid_size; ++i) {
                    for (int j = 0; j < grid_size; ++j) {
                            auto x = x_grid[j + i * grid_size];
                            auto z = z_grid[j + i * grid_size];
                            if (lt_radius(x, z, inner_rad)) { // check level of paraboloid
                                    min_y = y_grid[j + i * grid_size];
                                    break;
                            }
                    }
            }


            /*
                    generate mesh vertices for disk and elliptic paraboloid
            */
            auto half_grid = grid_size / 2;
            auto vertices_size = 0;
            for (int i = 0; i < grid_size; ++i) {
                    for (int j = 0; j < grid_size; ++j) {
                            auto x = x_grid[j + i * grid_size];
                            auto z = z_grid[j + i * grid_size];

                            if (set_hole) { // check hole inside disk
                                    auto skip = lt_radius(x, z, hole_rad);
                                    if (skip)
                                            continue;
                            }

                            auto y = min_y;
                            if (gt_radius(x, z, inner_rad)) // check level of paraboloid
                                    y = y_grid[j + i * grid_size];

                            vertices.push_back(x + cen[0]);
                            vertices.push_back(y + cen[1]);
                            vertices.push_back(z + cen[2]);
                            vertices_size += 3;

                            if (useUV) { // texture coordinates
                                    auto u = texture_u[j];
                                    auto v = texture_v[i];
                                    if (i == 0 && j == 0) // center disk
                                        u = texture_u[half_grid];
                                    vertices.push_back(u);
                                    vertices.push_back(v);
                            }
                    }
            }


            /*
                    generate indices by y-order
            */
            int32 vert_size = vertices_size / _num_vertices;
            int32 last_vert = vertices_size / _num_vertices;

            bool oddRow = false;
            uint y = 0;
            while (vert_size > 0) {
                    if (!oddRow) // even rows: y == 0, y == 2; and so on
                    {
                            for (uint x = 0; x <= grid_size; ++x)
                            {
                                    vert_size--;
                                    auto current = y * grid_size + x;
                                    auto next = (y + 1) * grid_size + x;
                                    if (next >= last_vert)
                                            continue;
                                    indices.push_back(current);
                                    indices.push_back(next);
                                    if (vert_size <= 0)
                                            break;
                            }
                    }
                    else
                    {
                            for (int x = grid_size; x >= 0; --x)
                            {
                                    vert_size--;
                                    auto current = (y + 1) * grid_size + x;
                                    auto prev = y * grid_size + x;
                                    if (current >= last_vert)
                                            continue;
                                    indices.push_back(current);
                                    indices.push_back(prev);
                                    if (vert_size <= 0)
                                            break;
                            }
                    }
                    oddRow = !oddRow;
                    y++;
            }


            return true;

    }

private:
    // compare inner radius and outer radius
    bool lt_radius(const float x, const float z, const float radius) {
            auto r1 = pow((x - cen[0]), 2);
            auto r2 = pow((z - cen[2]), 2);
            auto lt = ((r1 + r2) <= pow(radius, 2));
            return lt;
    }
    bool gt_radius(const float x, const float z, const float radius) {
            auto r1 = pow((x - cen[0]), 2);
            auto r2 = pow((z - cen[2]), 2);
            auto gt = ((r1 + r2) > pow(radius, 2));
            return gt;
    }
};







class PartitionBowl
{
private:
    typedef std::tuple< meshgen::mesh_grid<float, 0, 2>, meshgen::mesh_grid<float, 1, 2>> grid_type;
    constexpr static float PI = 3.14159265359f;
    constexpr static auto epsilon = std::numeric_limits<float>::epsilon();
    constexpr static int32 _num_vertices = 3; // x, y, z
private:
    float cen[3];
    float inner_rad;
    float rad;
    float param_a, param_b, param_c;
    float hole_rad;
    bool set_hole = false;
    bool useUV = false;
public:
    PartitionBowl(const float inner_radius, const float radius, const float a, const float b, const float c, const float center[3]=default_center)
            : inner_rad(inner_radius), rad(radius), param_a(a), param_b(b), param_c(c), hole_rad(0.f)
    {
        cen[0] = center[0];
        cen[1] = center[1];
        cen[2] = center[2];
    }

    bool generate_mesh(const uint part_nums, const float max_size_vert, std::vector<std::vector<float>>& vertices, std::vector<std::vector<uint>>& indices)
    {
        set_hole = false;
        useUV = false;
        return generate_mesh_(part_nums, max_size_vert, vertices, indices);
    }

    bool generate_mesh_uv(const uint part_nums, const float max_size_vert, std::vector<std::vector<float>>& vertices, std::vector<std::vector<uint>>& indices)
    {
        set_hole = false;
        useUV = true;
        return generate_mesh_(part_nums, max_size_vert, vertices, indices);
    }

protected:
    bool generate_mesh_(const uint part_nums, const float max_size_vert, std::vector<std::vector<float>>& vertices, std::vector<std::vector<uint>>& indices)
    {

        if (fabs(param_a) <= epsilon || fabs(param_b) <= epsilon || fabs(param_c) <= epsilon)
                return false;
        if (rad <= 0.f || inner_rad <= 0.f)
                return false;
        if (set_hole && hole_rad <= 0.f)
                return false;
        if (part_nums <= 1)
                return false;
        auto a = param_a;
        auto b = param_b;
        auto c = param_c;

        vertices = std::move(std::vector<std::vector<float>>(part_nums));

        indices = std::move(std::vector<std::vector<uint>>(part_nums));


        float step_size = (2 * PI) / part_nums;

        /*
                prepare grid mesh
        */
        // texture coordinates generate (u, v) [0, 1]
        std::vector<float> texture_u = meshgen::linspace(0.f, 1.f, max_size_vert);
        auto texture_v = texture_u;

        std::vector<grid_type> mesh_pairs;
        auto r = meshgen::linspace(0.0f, rad, max_size_vert);
        for (auto i = 0, next = 1; i < part_nums; i++, next += 1) {
                auto theta = meshgen::linspace(i * step_size, next * step_size, max_size_vert);
                mesh_pairs.push_back(meshgen::meshgrid(r, theta));
        }

        std::vector<std::vector<float>> x_grid(part_nums);
        std::vector<std::vector<float>> y_grid(part_nums);
        std::vector<std::vector<float>> z_grid(part_nums);


        size_t grid_size = std::get<0>(mesh_pairs[0]).size();

        // Convert to rectangular coordinates
        for (auto k = 0; k < part_nums; ++k) {
                auto R = std::get<0>(mesh_pairs[k]);
                auto THETA = std::get<1>(mesh_pairs[k]);
                for (int i = 0; i < grid_size; ++i) {
                        for (int j = 0; j < grid_size; ++j) {
                                auto x = R(i, j) * cos(THETA(i, j));
                                auto z = R(i, j) * sin(THETA(i, j));
                                auto y = c * (pow((x / a), 2) + pow((z / b), 2));
                                x_grid[k].push_back(x);
                                z_grid[k].push_back(z);
                                y_grid[k].push_back(y);
                        }
                }
        }

        /*
                find start level
        */
        auto min_y = 0.f;
        auto idx_min_y = 0u;
        for (int i = 0; i < grid_size; ++i) {
                for (int j = 0; j < grid_size; ++j) {
                        auto x = x_grid[0][j + i * grid_size];
                        auto z = z_grid[0][j + i * grid_size];
                        if (lt_radius(x, z, inner_rad)) { // check level of paraboloid
                                min_y = y_grid[0][j + i * grid_size];
                                idx_min_y = i;
                                break;
                        }
                }
        }


        /*
                generate mesh vertices for disk and elliptic paraboloid
        */
        auto half_grid = grid_size / 2;
        for (auto k = 0; k < part_nums; ++k) {
                for (int i = 0; i < grid_size; ++i) {
                        for (int j = 0; j < grid_size; ++j) {
                                auto x = x_grid[k][j + i * grid_size];
                                auto z = z_grid[k][j + i * grid_size];

                                if (set_hole) { // check hole inside disk
                                    auto skip = lt_radius(x, z, hole_rad);
                                    if (skip)
                                            continue;
                                }

                                auto y = min_y;
                                if (gt_radius(x, z, inner_rad)) // check level of paraboloid
                                        y = y_grid[k][j + i * grid_size];

                                vertices[k].push_back(x + cen[0]);
                                vertices[k].push_back(y + cen[1]);
                                vertices[k].push_back(z + cen[2]);

                                if (useUV) { // texture coordinates
                                    auto u = texture_u[j];
                                    auto v = texture_v[i];
                                    if (i == 0 && j == 0) // center disk
                                        u = texture_u[half_grid];
                                    vertices[k].push_back(u);
                                    vertices[k].push_back(v);
                                }
                        }
                }
        }


        /*
                generate indices by y-order
        */

        for (auto i = 0; i < part_nums; ++i) {
                bool oddRow = false;
                for (uint y = 0; y < grid_size-1; ++y){

                        if (!oddRow) // even rows: y == 0, y == 2; and so on
                        {
                                for (uint x = 0; x < grid_size; ++x)
                                {
                                        auto current = y * grid_size + x;
                                        auto next = (y + 1) * grid_size + x;
                                        /* change order when change disk to elliptic paraboloid */
                                        if (y == idx_min_y && x == 0){
                                                std::swap(current, next);
                                                indices[i].push_back(next);
                                                indices[i].push_back(current);
                                                indices[i].push_back(current - grid_size);
                                                continue;
                                        }
                                        indices[i].push_back(current);
                                        indices[i].push_back(next);
                                }
                        }
                        else
                        {
                                for (int x = grid_size-1; x >= 0; --x)
                                {
                                        auto current = (y + 1) * grid_size + x;
                                        auto prev = y * grid_size + x;
                                        /* change order when change disk to elliptic paraboloid */
                                        if (y == idx_min_y && x == grid_size - 1) {
                                                std::swap(current, prev);
                                                indices[i].push_back(current);
                                                indices[i].push_back(prev);
                                                indices[i].push_back(prev-grid_size);
                                                continue;
                                        }
                                        indices[i].push_back(current);
                                        indices[i].push_back(prev);

                                }
                        }
                        oddRow = !oddRow;
                }
        }


        return true;
    }
private:
    // compare inner radius and outer radius
    bool lt_radius(const float x, const float z, const float radius) {
        auto r1 = pow((x - cen[0]), 2);
        auto r2 = pow((z - cen[2]), 2);
        auto lt = ((r1 + r2) <= pow(radius, 2));
        return lt;
    }
    bool gt_radius(const float x, const float z, const float radius) {
        auto r1 = pow((x - cen[0]), 2);
        auto r2 = pow((z - cen[2]), 2);
        auto gt = ((r1 + r2) > pow(radius, 2));
        return gt;
    }
};


