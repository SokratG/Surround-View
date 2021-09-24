#include "Bowl.hpp"


bool Bowl::generate_mesh_(const float max_size_vert, std::vector<float>& vertices, std::vector<uint>& indices)
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
        std::vector<float> texture_u = meshgen::linspace(0.f, (1.f + eps_uv), max_size_vert);
        auto texture_v = texture_u;

        auto r = meshgen::linspace(hole_rad, rad, max_size_vert); // min_size = 0.f, max_size = 100.f,
        auto theta = meshgen::linspace(0.f, polar_coord, max_size_vert);
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
        auto idx_min_y = 0u; // index y - component when transition between disk and paraboloid
        for (int i = 0; i < grid_size; ++i) {
                for (int j = 0; j < grid_size; ++j) {
                        auto x = x_grid[j + i * grid_size];
                        auto z = z_grid[j + i * grid_size];
                        if (lt_radius(x, z, inner_rad)) { // check level of paraboloid
                                min_y = y_grid[j + i * grid_size];
                                idx_min_y = i;
                                break;
                        }
                }
        }


        /*
                generate mesh vertices for disk and elliptic paraboloid
        */
        auto half_grid = grid_size / 2;
        auto vertices_size = 0;
        auto offset_idx_min_y = 0;
        for (int i = 0; i < grid_size; ++i) {
                for (int j = 0; j < grid_size; ++j) {
                        auto x = x_grid[j + i * grid_size];
                        auto z = z_grid[j + i * grid_size];


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
                                if (i == 0 && j == 0 &&  !set_hole) // center disk
                                    u = texture_u[half_grid];
                                vertices.push_back(u);
                                vertices.push_back(v);
                        }
                }
        }


        /*
                generate indices by y-order
        */

        idx_min_y -= offset_idx_min_y;
        int32 last_vert = vertices_size / _num_vertices;
        generate_indices(indices, grid_size, idx_min_y, last_vert);

        return true;
}


void Bowl::generate_indices(std::vector<uint>& indices, const uint grid_size, const uint idx_min_y, const int32 last_vert) {
    bool oddRow = false;

    for (uint y = 0; y < grid_size - 1; ++y) {
            if (!oddRow) // even rows: y == 0, y == 2; and so on
            {
                    for (uint x = 0; x < grid_size; ++x)
                    {
                            auto current = y * grid_size + x;
                            auto next = (y + 1) * grid_size + x;
                            /* change order when change disk to elliptic paraboloid */
                            if (y == idx_min_y && x == 0) {
                                    std::swap(current, next);
                                    indices.push_back(current - grid_size);
                                    indices.push_back(next);
                                    indices.push_back(current);
                                    continue;
                            }
                            if (set_hole && (current >= last_vert || next >= last_vert))
                                    continue;
                            indices.push_back(current);
                            indices.push_back(next);
                    }
            }
            else
            {
                    for (int x = grid_size - 1; x >= 0; --x)
                    {
                            auto current = (y + 1) * grid_size + x;
                            auto prev = y * grid_size + x;
                            /* change order when change disk to elliptic paraboloid */
                            if (y == idx_min_y && x == grid_size - 1) {
                                    indices.push_back(current - grid_size);
                                    indices.push_back(current);
                                    indices.push_back(prev);
                                    continue;
                            }
                            if (set_hole && (current >= last_vert || prev >= last_vert))
                                    continue;
                            indices.push_back(current);
                            indices.push_back(prev);

                    }
            }
            oddRow = !oddRow;
    }
}



bool HemiSphere::generate_mesh_(std::vector<float>& vertices, std::vector<uint>& indices)
{
        if (set_hole && hole_rad <= 0.f)
                return false;

        vertices.clear();

        indices.clear();

        for(int y = 0; y <= y_segment; ++y){
            for(int x = 0; x <= x_segment; ++x){
                float xSegm = (float)x / (float)x_segment;
                float ySegm = (float)y / (float)y_segment;
                float xPos = std::cos(xSegm * 2.0 * PI) * std::sin(ySegm * half_pi);
                float yPos = 1.0f - std::cos(ySegm * half_pi);
                float zPos = std::sin(xSegm * 2.0 * PI) * std::sin(ySegm * half_pi);

                vertices.push_back(xPos);
                vertices.push_back(yPos);
                vertices.push_back(zPos);
                vertices.push_back(xSegm);
                vertices.push_back(ySegm);
            }

        }


        bool oddRow = false;
        for(uint y = 0; y < y_segment - 1; ++y){
            if (!oddRow){

                for(uint x = 0; x <= x_segment; ++x){
                    indices.push_back(y * (x_segment + 1) + x);
                    indices.push_back((y + 1)* (x_segment + 1) + x);
                }
            }
            else{

                for(int x = x_segment; x >= 0; --x){
                    indices.push_back((y + 1)* (x_segment + 1) + x);
                    indices.push_back(y * (x_segment + 1) + x);
                }
            }
            oddRow = !oddRow;
        }


        return true;
}

