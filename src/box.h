#pragma once
#include "type.h"
#include <cmath>
#include <stdexcept>

struct Box
{
    // data[0..8] = box matrix (a,b,c as rows)
    // data[9..17] = inverse box matrix (precomputed)
    double data[18]{};
    double origin[3]{};
    int boundary[3]{};
    double thickness[3]{};
    bool triclinic{false};

    // ==========================================================
    // Volume
    // ==========================================================
    double get_volume() const
    {
        if (triclinic)
        {
            return data[0] * (data[4] * data[8] - data[5] * data[7]) -
                   data[1] * (data[3] * data[8] - data[5] * data[6]) +
                   data[2] * (data[3] * data[7] - data[4] * data[6]);
        }
        else
        {
            return data[0] * data[4] * data[8];
        }
    }

    // ==========================================================
    // Box length in each direction
    // ==========================================================
    double get_box_length(int dir) const
    {
        if (triclinic)
        {
            const double *v = data + dir * 3;
            return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        }
        else
        {
            return data[dir * 4]; // 0,4,8
        }
    }

    // ==========================================================
    // Thickness (for visualization or analysis)
    // ==========================================================
    double get_thickness(int dir) const
    {
        if (!triclinic)
            return data[dir * 4];

        double V = get_volume();
        const double *a = data;
        const double *b = data + 3;
        const double *c = data + 6;
        double m, n, k;

        if (dir == 0)
        {
            m = b[1] * c[2] - b[2] * c[1];
            n = b[2] * c[0] - b[0] * c[2];
            k = b[0] * c[1] - b[1] * c[0];
        }
        else if (dir == 1)
        {
            m = a[1] * c[2] - a[2] * c[1];
            n = a[2] * c[0] - a[0] * c[2];
            k = a[0] * c[1] - a[1] * c[0];
        }
        else if (dir == 2)
        {
            m = a[1] * b[2] - a[2] * b[1];
            n = a[2] * b[0] - a[0] * b[2];
            k = a[0] * b[1] - a[1] * b[0];
        }
        else
        {
            throw std::runtime_error("Invalid direction.");
        }

        return V / std::sqrt(m * m + n * n + k * k);
    }

    // ==========================================================
    // Periodic boundary condition (minimum image convention)
    // ==========================================================
    void pbc(double &xij, double &yij, double &zij) const
    {
        if (triclinic)
        {
            // Convert to fractional coordinates
            double x = xij * data[9] + yij * data[12] + zij * data[15];
            double y = xij * data[10] + yij * data[13] + zij * data[16];
            double z = xij * data[11] + yij * data[14] + zij * data[17];

            // Apply PBC with floor (safe for any distance)
            if (boundary[0])
                x -= std::floor(x + 0.5);
            if (boundary[1])
                y -= std::floor(y + 0.5);
            if (boundary[2])
                z -= std::floor(z + 0.5);

            // Back to Cartesian
            xij = x * data[0] + y * data[3] + z * data[6];
            yij = x * data[1] + y * data[4] + z * data[7];
            zij = x * data[2] + y * data[5] + z * data[8];
        }
        else
        {
            // Orthogonal box
            if (boundary[0])
                xij -= data[0] * std::floor(xij / data[0] + 0.5);
            if (boundary[1])
                yij -= data[4] * std::floor(yij / data[4] + 0.5);
            if (boundary[2])
                zij -= data[8] * std::floor(zij / data[8] + 0.5);
        }
    }

    // ==========================================================
    // Wrap coordinates into box (NEW!)
    // ==========================================================
    void wrap_into_box(double &x, double &y, double &z) const
    {
        if (triclinic)
        {
            double dx = x - origin[0];
            double dy = y - origin[1];
            double dz = z - origin[2];

            // 转换到分数坐标
            double nx = dx * data[9] + dy * data[12] + dz * data[15];
            double ny = dx * data[10] + dy * data[13] + dz * data[16];
            double nz = dx * data[11] + dy * data[14] + dz * data[17];

            // 应用周期边界条件
            if (boundary[0])
                nx -= std::floor(nx);
            if (boundary[1])
                ny -= std::floor(ny);
            if (boundary[2])
                nz -= std::floor(nz);

            // 转换回笛卡尔坐标
            x = origin[0] + nx * data[0] + ny * data[3] + nz * data[6];
            y = origin[1] + nx * data[1] + ny * data[4] + nz * data[7];
            z = origin[2] + nx * data[2] + ny * data[5] + nz * data[8];
        }
        else
        {
            // 正交盒子
            if (boundary[0])
            {
                double dx = x - origin[0];
                x = origin[0] + dx - data[0] * std::floor(dx / data[0]);
            }
            if (boundary[1])
            {
                double dy = y - origin[1];
                y = origin[1] + dy - data[4] * std::floor(dy / data[4]);
            }
            if (boundary[2])
            {
                double dz = z - origin[2];
                z = origin[2] + dz - data[8] * std::floor(dz / data[8]);
            }
        }
    }
};

// ==========================================================
// Compute inverse box (for triclinic boxes)
// ==========================================================
static void get_inverse_box(Box &box)
{
    double det = box.get_volume();
    if (std::abs(det) < 1e-12)
        throw std::runtime_error("The volume of the box is zero.");

    double inv_det = 1.0 / det;
    const double *m = box.data;

    // Compute adjugate matrix
    box.data[9] = (m[4] * m[8] - m[5] * m[7]) * inv_det;
    box.data[10] = -(m[1] * m[8] - m[2] * m[7]) * inv_det;
    box.data[11] = (m[1] * m[5] - m[2] * m[4]) * inv_det;

    box.data[12] = -(m[3] * m[8] - m[5] * m[6]) * inv_det;
    box.data[13] = (m[0] * m[8] - m[2] * m[6]) * inv_det;
    box.data[14] = -(m[0] * m[5] - m[2] * m[3]) * inv_det;

    box.data[15] = (m[3] * m[7] - m[4] * m[6]) * inv_det;
    box.data[16] = -(m[0] * m[7] - m[1] * m[6]) * inv_det;
    box.data[17] = (m[0] * m[4] - m[1] * m[3]) * inv_det;
}

// ==========================================================
// Build box from numpy (orthogonal or triclinic)
// ==========================================================
Box get_box(const RTwoArrayD box_py, const ROneArrayD origin, const ROneArrayI boundary)
{
    Box box{};
    // Fill box matrix
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            box.data[i * 3 + j] = box_py(i, j);
            if (i != j && std::abs(box.data[i * 3 + j]) > 1e-10)
                box.triclinic = true;
        }

    // If any length negative or non-orthogonal, treat as triclinic
    if (box.data[0] < 0 || box.data[4] < 0 || box.data[8] < 0)
        box.triclinic = true;

    if (box.triclinic)
    {
        get_inverse_box(box);
    }
    else
    {
        // Orthogonal box: just set inverse easily
        box.data[9] = 1.0 / box.data[0];
        box.data[13] = 1.0 / box.data[4];
        box.data[17] = 1.0 / box.data[8];
    }

    // Origin & boundary
    for (int i = 0; i < 3; ++i)
    {
        box.origin[i] = origin(i);
        box.boundary[i] = boundary(i);
        box.thickness[i] = box.get_thickness(i);
    }

    return box;
}