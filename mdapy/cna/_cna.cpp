// Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
// This file is from the mdapy project, released under the BSD 3-Clause License.
// Some part codes come from the ovito CNA modifier. see https://gitlab.com/stuko/ovito/-/tree/master/src/ovito/particles/modifier/analysis/cna?ref_type=heads

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <omp.h>
// #include <iostream>

namespace py = pybind11;

typedef unsigned int CNAPairBond;

struct NeighborBondArray
{
    /// Two-dimensional bit array that stores the bonds between neighbors.
    unsigned int neighborArray[32];

    /// Resets all bits.
    NeighborBondArray()
    {
        memset(neighborArray, 0, sizeof(neighborArray));
    }

    /// Returns whether two nearest neighbors have a bond between them.
    inline bool neighborBond(int neighborIndex1, int neighborIndex2) const
    {
        return (neighborArray[neighborIndex1] & (1 << neighborIndex2));
    }

    /// Sets whether two nearest neighbors have a bond between them.
    inline void setNeighborBond(int neighborIndex1, int neighborIndex2, bool bonded)
    {
        if (bonded)
        {
            neighborArray[neighborIndex1] |= (1 << neighborIndex2);
            neighborArray[neighborIndex2] |= (1 << neighborIndex1);
        }
        else
        {
            neighborArray[neighborIndex1] &= ~(1 << neighborIndex2);
            neighborArray[neighborIndex2] &= ~(1 << neighborIndex1);
        }
    }
};

/******************************************************************************
 * Find all atoms that are nearest neighbors of the given pair of atoms.
 ******************************************************************************/
int findCommonNeighbors(const NeighborBondArray &neighborArray, int neighborIndex, unsigned int &commonNeighbors)
{
    commonNeighbors = neighborArray.neighborArray[neighborIndex];
#ifndef MSVC
    // Count the number of bits set in neighbor bit-field.
    return __builtin_popcount(commonNeighbors);
#else
    // Count the number of bits set in neighbor bit-field.
    unsigned int v = commonNeighbors - ((commonNeighbors >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    return ((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
#endif
}

/******************************************************************************
 * Finds all bonds between common nearest neighbors.
 ******************************************************************************/
int findNeighborBonds(const NeighborBondArray &neighborArray, unsigned int commonNeighbors, int numNeighbors, CNAPairBond *neighborBonds)
{
    int numBonds = 0;

    unsigned int nib[32];
    int nibn = 0;
    unsigned int ni1b = 1;
    for (int ni1 = 0; ni1 < numNeighbors; ni1++, ni1b <<= 1)
    {
        if (commonNeighbors & ni1b)
        {
            unsigned int b = commonNeighbors & neighborArray.neighborArray[ni1];
            for (int n = 0; n < nibn; n++)
            {
                if (b & nib[n])
                {
                    neighborBonds[numBonds++] = ni1b | nib[n];
                }
            }
            nib[nibn++] = ni1b;
        }
    }
    return numBonds;
}

/******************************************************************************
 * Find all chains of bonds.
 ******************************************************************************/
static int getAdjacentBonds(unsigned int atom, CNAPairBond *bondsToProcess, int &numBonds, unsigned int &atomsToProcess, unsigned int &atomsProcessed)
{
    int adjacentBonds = 0;
    for (int b = numBonds - 1; b >= 0; b--)
    {
        if (atom & *bondsToProcess)
        {
            ++adjacentBonds;
            atomsToProcess |= *bondsToProcess & (~atomsProcessed);
            memmove(bondsToProcess, bondsToProcess + 1, sizeof(CNAPairBond) * b);
            numBonds--;
        }
        else
            ++bondsToProcess;
    }
    return adjacentBonds;
}

/******************************************************************************
 * Find all chains of bonds between common neighbors and determine the length
 * of the longest continuous chain.
 ******************************************************************************/
int calcMaxChainLength(CNAPairBond *neighborBonds, int numBonds)
{
    // Group the common bonds into clusters.
    int maxChainLength = 0;
    while (numBonds)
    {
        // Make a new cluster starting with the first remaining bond to be processed.
        numBonds--;
        unsigned int atomsToProcess = neighborBonds[numBonds];
        unsigned int atomsProcessed = 0;
        int clusterSize = 1;
        do
        {
#ifndef MSVC
            int nextAtomIndex = __builtin_ctz(atomsToProcess);
#else
            unsigned long nextAtomIndex;
            _BitScanForward(&nextAtomIndex, atomsToProcess);
#endif
            unsigned int nextAtom = 1 << nextAtomIndex;
            atomsProcessed |= nextAtom;
            atomsToProcess &= ~nextAtom;
            clusterSize += getAdjacentBonds(nextAtom, neighborBonds, numBonds, atomsToProcess, atomsProcessed);
        } while (atomsToProcess);
        if (clusterSize > maxChainLength)
            maxChainLength = clusterSize;
    }
    return maxChainLength;
}

double pbcdis_sq(
    py::detail::unchecked_reference<double, 2> c_pos,
    py::detail::unchecked_reference<double, 2> c_box,
    py::detail::unchecked_reference<double, 2> c_inverse_box,
    py::detail::unchecked_reference<bool, 1> c_boundary,
    int i, int j)
{

    double rij_0 = c_pos(j, 0) - c_pos(i, 0);
    double rij_1 = c_pos(j, 1) - c_pos(i, 1);
    double rij_2 = c_pos(j, 2) - c_pos(i, 2);
    double n_0 = rij_0 * c_inverse_box(0, 0) + rij_1 * c_inverse_box(1, 0) + rij_2 * c_inverse_box(2, 0);
    double n_1 = rij_0 * c_inverse_box(0, 1) + rij_1 * c_inverse_box(1, 1) + rij_2 * c_inverse_box(2, 1);
    double n_2 = rij_0 * c_inverse_box(0, 2) + rij_1 * c_inverse_box(1, 2) + rij_2 * c_inverse_box(2, 2);
    if (c_boundary(0))
    {
        if (n_0 > 0.5)
        {
            n_0 -= 1.;
        }
        else if (n_0 < -0.5)
        {
            n_0 += 1.;
        }
    }
    if (c_boundary(1))
    {
        if (n_1 > 0.5)
        {
            n_1 -= 1.;
        }
        else if (n_1 < -0.5)
        {
            n_1 += 1.;
        }
    }
    if (c_boundary(2))
    {
        if (n_2 > 0.5)
        {
            n_2 -= 1.;
        }
        else if (n_2 < -0.5)
        {
            n_2 += 1.;
        }
    }
    rij_0 = n_0 * c_box(0, 0) + n_1 * c_box(1, 0) + n_2 * c_box(2, 0);
    rij_1 = n_0 * c_box(0, 1) + n_1 * c_box(1, 1) + n_2 * c_box(2, 1);
    rij_2 = n_0 * c_box(0, 2) + n_1 * c_box(1, 2) + n_2 * c_box(2, 2);

    return rij_0 * rij_0 + rij_1 * rij_1 + rij_2 * rij_2;
}

void IdentifyDiamond(py::array pos, py::array box, py::array inverse_box, py::array boundary, py::array verlet_list, py::array new_verlet_list, py::array pattern)
{
    auto c_pos = pos.unchecked<double, 2>();
    auto c_box = box.unchecked<double, 2>();
    auto c_inverse_box = inverse_box.unchecked<double, 2>();
    auto c_boundary = boundary.unchecked<bool, 1>();
    auto c_verlet_list = verlet_list.unchecked<int, 2>();
    auto c_new_verlet_list = new_verlet_list.mutable_unchecked<int, 2>();
    auto c_pattern = pattern.mutable_unchecked<int, 1>();
    int N = c_verlet_list.shape(0);

#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        int count = 0;
        for (int m = 0; m < 4; m++)
        {
            int j = c_verlet_list(i, m);
            int nn = 0;
            for (int kk = 0; kk < 4; kk++)
            {
                int kkk = c_verlet_list(j, kk);
                if (kkk != i && nn < 3)
                {
                    c_new_verlet_list(i, count) = kkk;
                    nn += 1;
                    count += 1;
                }
            }
        }

        double rc = 0.;
        for (int m = 0; m < 12; m++)
        {
            int j = c_new_verlet_list(i, m);
            rc += std::sqrt(pbcdis_sq(c_pos, c_box, c_inverse_box, c_boundary, i, j));
        }
        rc /= 12;
        double localCutoff = rc * 1.2071068;

        double localCutoffSquared = localCutoff * localCutoff;

        NeighborBondArray neighborArray;
        for (int ni1 = 0; ni1 < 12; ni1++)
        {
            neighborArray.setNeighborBond(ni1, ni1, false);
            for (int ni2 = ni1 + 1; ni2 < 12; ni2++)
                neighborArray.setNeighborBond(ni1, ni2,
                                              pbcdis_sq(c_pos, c_box, c_inverse_box, c_boundary, c_new_verlet_list(i, ni1), c_new_verlet_list(i, ni2)) <= localCutoffSquared);
        }
        int n421 = 0;
        int n422 = 0;
        for (int ni = 0; ni < 12; ni++)
        {
            // Determine number of neighbors the two atoms have in common.
            unsigned int commonNeighbors;
            int numCommonNeighbors = findCommonNeighbors(neighborArray, ni, commonNeighbors);
            if (numCommonNeighbors != 4)
                break;

            // Determine the number of bonds among the common neighbors.
            CNAPairBond neighborBonds[12 * 12];
            int numNeighborBonds = findNeighborBonds(neighborArray, commonNeighbors, 12, neighborBonds);
            if (numNeighborBonds != 2)
                break;

            // Determine the number of bonds in the longest continuous chain.
            int maxChainLength = calcMaxChainLength(neighborBonds, numNeighborBonds);
            if (maxChainLength == 1)
                n421++;
            else if (maxChainLength == 2)
                n422++;
        }

        if (n421 == 12)
            c_pattern(i) = 1;
        else if (n421 == 6 && n422 == 6)
            c_pattern(i) = 4;
    }

    for (int i = 0; i < N; i++)
    {
        int ctype = c_pattern(i);
        if (ctype != 1 && ctype != 4)
            continue;
        for (int jj = 0; jj < 4; jj++)
        {
            int j = c_verlet_list(i, jj);
            if (c_pattern(j) == 0)
            {
                if (ctype == 1)
                    c_pattern(j) = 2;
                if (ctype == 4)
                    c_pattern(j) = 5;
            }
        }
    }
    for (int i = 0; i < N; i++)
    {
        int ctype = c_pattern(i);
        if (ctype != 2 && ctype != 5)
            continue;
        for (int jj = 0; jj < 4; jj++)
        {
            int j = c_verlet_list(i, jj);
            if (c_pattern(j) == 0)
            {
                if (ctype == 2)
                    c_pattern(j) = 3;
                if (ctype == 5)
                    c_pattern(j) = 6;
            }
        }
    }
}

void AdaptiveCNA(py::array pos, py::array box, py::array inverse_box, py::array boundary, py::array verlet_list, py::array pattern)
{
    auto c_pos = pos.unchecked<double, 2>();
    auto c_box = box.unchecked<double, 2>();
    auto c_inverse_box = inverse_box.unchecked<double, 2>();
    auto c_boundary = boundary.unchecked<bool, 1>();
    auto c_verlet_list = verlet_list.unchecked<int, 2>();
    auto c_pattern = pattern.mutable_unchecked<int, 1>();
    int N = c_verlet_list.shape(0);

#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {

        int nn = 12;
        double rc = 0.;
        for (int m = 0; m < nn; m++)
        {
            int j = c_verlet_list(i, m);
            rc += std::sqrt(pbcdis_sq(c_pos, c_box, c_inverse_box, c_boundary, i, j));
        }

        double localCutoff = rc / nn * (1.0 + std::sqrt(2.0)) * 0.5;
        double localCutoffSquared = localCutoff * localCutoff;

        NeighborBondArray neighborArray;
        for (int ni1 = 0; ni1 < nn; ni1++)
        {
            neighborArray.setNeighborBond(ni1, ni1, false);
            for (int ni2 = ni1 + 1; ni2 < nn; ni2++)
                neighborArray.setNeighborBond(ni1, ni2,
                                              pbcdis_sq(c_pos, c_box, c_inverse_box, c_boundary, c_verlet_list(i, ni1), c_verlet_list(i, ni2)) <= localCutoffSquared);
        }
        int n421 = 0;
        int n422 = 0;
        int n555 = 0;
        for (int ni = 0; ni < nn; ni++)
        {
            // Determine number of neighbors the two atoms have in common.
            unsigned int commonNeighbors;
            int numCommonNeighbors = findCommonNeighbors(neighborArray, ni, commonNeighbors);
            if (numCommonNeighbors != 4 && numCommonNeighbors != 5)
                break;

            // Determine the number of bonds among the common neighbors.
            CNAPairBond neighborBonds[14 * 14];
            int numNeighborBonds = findNeighborBonds(neighborArray, commonNeighbors, 12, neighborBonds);
            if (numNeighborBonds != 2 && numNeighborBonds != 5)
                break;

            // Determine the number of bonds in the longest continuous chain.
            int maxChainLength = calcMaxChainLength(neighborBonds, numNeighborBonds);
            if (numCommonNeighbors == 4 && numNeighborBonds == 2)
            {
                if (maxChainLength == 1)
                    n421++;
                else if (maxChainLength == 2)
                    n422++;
                else
                    break;
            }
            else if (numCommonNeighbors == 5 && numNeighborBonds == 5 && maxChainLength == 5)
                n555++;
            else
                break;
        }

        if (n421 == 12)
            c_pattern(i) = 1;
        else if (n421 == 6 && n422 == 6)
            c_pattern(i) = 2;
        else if (n555 == 12)
            c_pattern(i) = 4;

        if (c_pattern(i) == 0)
        {
            int nn = 14;
            double rc = 0.;
            for (int m = 0; m < 8; m++)
            {
                int j = c_verlet_list(i, m);
                rc += std::sqrt(pbcdis_sq(c_pos, c_box, c_inverse_box, c_boundary, i, j) / (3.0 / 4.0));
            }
            for (int m = 8; m < 14; m++)
            {
                int j = c_verlet_list(i, m);
                rc += std::sqrt(pbcdis_sq(c_pos, c_box, c_inverse_box, c_boundary, i, j));
            }

            double localCutoff = rc / nn * (1.0 + std::sqrt(2.0)) * 0.5;
            double localCutoffSquared = localCutoff * localCutoff;

            NeighborBondArray neighborArray;
            for (int ni1 = 0; ni1 < nn; ni1++)
            {
                neighborArray.setNeighborBond(ni1, ni1, false);
                for (int ni2 = ni1 + 1; ni2 < nn; ni2++)
                    neighborArray.setNeighborBond(ni1, ni2,
                                                  pbcdis_sq(c_pos, c_box, c_inverse_box, c_boundary, c_verlet_list(i, ni1), c_verlet_list(i, ni2)) <= localCutoffSquared);
            }
            int n444 = 0;
            int n666 = 0;
            for (int ni = 0; ni < nn; ni++)
            {
                // Determine number of neighbors the two atoms have in common.
                unsigned int commonNeighbors;
                int numCommonNeighbors = findCommonNeighbors(neighborArray, ni, commonNeighbors);
                if (numCommonNeighbors != 4 && numCommonNeighbors != 6)
                    break;

                // Determine the number of bonds among the common neighbors.
                CNAPairBond neighborBonds[14 * 14];
                int numNeighborBonds = findNeighborBonds(neighborArray, commonNeighbors, 14, neighborBonds);
                if (numNeighborBonds != 4 && numNeighborBonds != 6)
                    break;

                // Determine the number of bonds in the longest continuous chain.
                int maxChainLength = calcMaxChainLength(neighborBonds, numNeighborBonds);
                if (numCommonNeighbors == 4 && numNeighborBonds == 4 && maxChainLength == 4)
                    n444++;
                else if (numCommonNeighbors == 6 && numNeighborBonds == 6 && maxChainLength == 6)
                    n666++;
                else
                    break;
            }
            if (n666 == 8 && n444 == 6)
                c_pattern(i) = 3;
        }
    }
}

void FixedCNA(py::array pos, py::array box, py::array inverse_box, py::array boundary, py::array verlet_list, py::array neighbor_number, py::array pattern, double rc)
{
    auto c_pos = pos.unchecked<double, 2>();
    auto c_box = box.unchecked<double, 2>();
    auto c_inverse_box = inverse_box.unchecked<double, 2>();
    auto c_boundary = boundary.unchecked<bool, 1>();
    auto c_verlet_list = verlet_list.unchecked<int, 2>();
    auto c_neighbor_number = neighbor_number.unchecked<int, 1>();
    auto c_pattern = pattern.mutable_unchecked<int, 1>();
    int N = c_verlet_list.shape(0);

#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        int nn = c_neighbor_number(i);
        if (nn == 12 || nn == 14)
        {
            double localCutoffSquared = rc * rc;
            NeighborBondArray neighborArray;
            for (int ni1 = 0; ni1 < nn; ni1++)
            {
                neighborArray.setNeighborBond(ni1, ni1, false);
                for (int ni2 = ni1 + 1; ni2 < nn; ni2++)
                    neighborArray.setNeighborBond(ni1, ni2,
                                                  pbcdis_sq(c_pos, c_box, c_inverse_box, c_boundary, c_verlet_list(i, ni1), c_verlet_list(i, ni2)) <= localCutoffSquared);
            }
            int n421 = 0;
            int n422 = 0;
            int n555 = 0;
            int n444 = 0;
            int n666 = 0;
            for (int ni = 0; ni < nn; ni++)
            {
                // Determine number of neighbors the two atoms have in common.
                unsigned int commonNeighbors;
                int numCommonNeighbors = findCommonNeighbors(neighborArray, ni, commonNeighbors);
                CNAPairBond neighborBonds[14 * 14];

                int numNeighborBonds = findNeighborBonds(neighborArray, commonNeighbors, nn, neighborBonds);
                int maxChainLength = calcMaxChainLength(neighborBonds, numNeighborBonds);

                if (numCommonNeighbors == 4 && numNeighborBonds == 2)
                {
                    if (maxChainLength == 1)
                        n421++;
                    else if (maxChainLength == 2)
                        n422++;
                }
                else if (numCommonNeighbors == 5 && numNeighborBonds == 5 && maxChainLength == 5)
                    n555++;
                else if (numCommonNeighbors == 4 && numNeighborBonds == 4 && maxChainLength == 4)
                    n444++;
                else if (numCommonNeighbors == 6 && numNeighborBonds == 6 && maxChainLength == 6)
                    n666++;
            }
            // if (i < 10)
            //     std::cout << i << " " << n421 << " " << n444 << " " << n666 << std::endl;
            if (n421 == 12)
                c_pattern(i) = 1;
            else if (n421 == 6 && n422 == 6)
                c_pattern(i) = 2;
            else if (n555 == 12)
                c_pattern(i) = 4;
            else if (n666 == 8 && n444 == 6)
                c_pattern(i) = 3;
        }
    }
}

PYBIND11_MODULE(_cna, m)
{
    m.def("_ids", &IdentifyDiamond);
    m.def("_acna", &AdaptiveCNA);
    m.def("_fcna", &FixedCNA);
}
