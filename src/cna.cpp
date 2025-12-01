// Copyright (c) 2022-2025, Yongchao Wu in Aalto University
// This file is from the mdapy project, released under the BSD 3-Clause License.
// Some part codes come from the ovito CNA modifier. see https://gitlab.com/stuko/ovito/-/tree/master/src/ovito/particles/modifier/analysis/cna?ref_type=heads

#include "box.h"
#include "type.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cmath>
#include <omp.h>

namespace nb = nanobind;

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

double pbcdis_sq(const Box &box,
                 const double *x,
                 const double *y,
                 const double *z,
                 const int i,
                 const int j)
{
    double xij = x[j] - x[i];
    double yij = y[j] - y[i];
    double zij = z[j] - z[i];
    box.pbc(xij, yij, zij);
    return xij * xij + yij * yij + zij * zij;
}

void IdentifyDiamond(
    const ROneArrayD x_py,
    const ROneArrayD y_py,
    const ROneArrayD z_py,
    const RTwoArrayD box_py,
    const ROneArrayD origin_py,
    const ROneArrayI boundary_py,
    const RTwoArrayI verlet_list_py,
    TwoArrayI new_verlet_list_py,
    OneArrayI pattern_py)
{
    const double *x = x_py.data();
    const double *y = y_py.data();
    const double *z = z_py.data();
    Box box = get_box(box_py, origin_py, boundary_py);
    auto verlet_list = verlet_list_py.view();
    auto new_verlet_list = new_verlet_list_py.view();
    auto pattern = pattern_py.view();
    const int N{static_cast<int>(x_py.shape(0))};

#pragma omp parallel for firstprivate(x, y, z, verlet_list)
    for (int i = 0; i < N; i++)
    {
        int count = 0;
        for (int m = 0; m < 4; m++)
        {
            const int j = verlet_list(i, m);
            int nn = 0;
            for (int kk = 0; kk < 4; kk++)
            {
                int kkk = verlet_list(j, kk);
                if (kkk != i && nn < 3)
                {
                    new_verlet_list(i, count) = kkk;
                    nn++;
                    count++;
                }
            }
        }

        double rc = 0.;
        for (int m = 0; m < 12; m++)
        {
            int j = new_verlet_list(i, m);
            rc += std::sqrt(pbcdis_sq(box, x, y, z, i, j));
        }
        rc /= 12.0;

        const double localCutoff = rc * 1.2071068;
        const double localCutoffSquared = localCutoff * localCutoff;

        NeighborBondArray neighborArray;
        for (int ni1 = 0; ni1 < 12; ni1++)
        {
            neighborArray.setNeighborBond(ni1, ni1, false);
            for (int ni2 = ni1 + 1; ni2 < 12; ni2++)
                neighborArray.setNeighborBond(ni1, ni2,
                                              pbcdis_sq(box, x, y, z, new_verlet_list(i, ni1), new_verlet_list(i, ni2)) <= localCutoffSquared);
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
            pattern(i) = 1;
        else if (n421 == 6 && n422 == 6)
            pattern(i) = 4;
    }

    for (int i = 0; i < N; i++)
    {
        const int ctype = pattern(i);
        if (ctype != 1 && ctype != 4)
            continue;
        for (int jj = 0; jj < 4; jj++)
        {
            const int j = verlet_list(i, jj);
            if (pattern(j) == 0)
            {
                if (ctype == 1)
                    pattern(j) = 2;
                if (ctype == 4)
                    pattern(j) = 5;
            }
        }
    }
    for (int i = 0; i < N; i++)
    {
        const int ctype = pattern(i);
        if (ctype != 2 && ctype != 5)
            continue;
        for (int jj = 0; jj < 4; jj++)
        {
            const int j = verlet_list(i, jj);
            if (pattern(j) == 0)
            {
                if (ctype == 2)
                    pattern(j) = 3;
                if (ctype == 5)
                    pattern(j) = 6;
            }
        }
    }
}

void AdaptiveCNA(
    const ROneArrayD x_py,
    const ROneArrayD y_py,
    const ROneArrayD z_py,
    const RTwoArrayD box_py,
    const ROneArrayD origin_py,
    const ROneArrayI boundary_py,
    const RTwoArrayI verlet_list_py,
    OneArrayI pattern_py)
{
    const double *x = x_py.data();
    const double *y = y_py.data();
    const double *z = z_py.data();
    Box box = get_box(box_py, origin_py, boundary_py);
    auto verlet_list = verlet_list_py.view();
    auto pattern = pattern_py.view();
    const int N{static_cast<int>(x_py.shape(0))};

#pragma omp parallel for firstprivate(x, y, z, verlet_list)
    for (int i = 0; i < N; i++)
    {

        const int nn = 12;
        double rc = 0.;
        for (int m = 0; m < nn; m++)
        {
            int j = verlet_list(i, m);
            rc += std::sqrt(pbcdis_sq(box, x, y, z, i, j));
        }

        const double localCutoff = rc / nn * (1.0 + std::sqrt(2.0)) * 0.5;
        const double localCutoffSquared = localCutoff * localCutoff;

        NeighborBondArray neighborArray;
        for (int ni1 = 0; ni1 < nn; ni1++)
        {
            neighborArray.setNeighborBond(ni1, ni1, false);
            for (int ni2 = ni1 + 1; ni2 < nn; ni2++)
                neighborArray.setNeighborBond(ni1, ni2,
                                              pbcdis_sq(box, x, y, z, verlet_list(i, ni1), verlet_list(i, ni2)) <= localCutoffSquared);
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
            pattern(i) = 1;
        else if (n421 == 6 && n422 == 6)
            pattern(i) = 2;
        else if (n555 == 12)
            pattern(i) = 4;

        if (pattern(i) == 0)
        {
            const int nn = 14;
            double rc = 0.;
            for (int m = 0; m < 8; m++)
            {
                int j = verlet_list(i, m);
                rc += std::sqrt(pbcdis_sq(box, x, y, z, i, j) / (3.0 / 4.0));
            }
            for (int m = 8; m < 14; m++)
            {
                int j = verlet_list(i, m);
                rc += std::sqrt(pbcdis_sq(box, x, y, z, i, j));
            }

            const double localCutoff = rc / nn * (1.0 + std::sqrt(2.0)) * 0.5;
            const double localCutoffSquared = localCutoff * localCutoff;

            NeighborBondArray neighborArray;
            for (int ni1 = 0; ni1 < nn; ni1++)
            {
                neighborArray.setNeighborBond(ni1, ni1, false);
                for (int ni2 = ni1 + 1; ni2 < nn; ni2++)
                    neighborArray.setNeighborBond(ni1, ni2,
                                                  pbcdis_sq(box, x, y, z, verlet_list(i, ni1), verlet_list(i, ni2)) <= localCutoffSquared);
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
                pattern(i) = 3;
        }
    }
}

void FixedCNA(const ROneArrayD x_py,
              const ROneArrayD y_py,
              const ROneArrayD z_py,
              const RTwoArrayD box_py,
              const ROneArrayD origin_py,
              const ROneArrayI boundary_py,
              const RTwoArrayI verlet_list_py,
              const ROneArrayI neighbor_number_py,
              OneArrayI pattern_py,
              const double rc)
{
    const double *x = x_py.data();
    const double *y = y_py.data();
    const double *z = z_py.data();
    Box box = get_box(box_py, origin_py, boundary_py);
    auto verlet_list = verlet_list_py.view();
    auto neighbor_number = neighbor_number_py.view();
    auto pattern = pattern_py.view();
    const int N{static_cast<int>(x_py.shape(0))};

    const double localCutoffSquared = rc * rc;

#pragma omp parallel for firstprivate(x, y, z, verlet_list, neighbor_number)
    for (int i = 0; i < N; i++)
    {
        const int nn = neighbor_number(i);
        if (nn == 12 || nn == 14)
        {
            NeighborBondArray neighborArray;
            for (int ni1 = 0; ni1 < nn; ni1++)
            {
                neighborArray.setNeighborBond(ni1, ni1, false);
                for (int ni2 = ni1 + 1; ni2 < nn; ni2++)
                    neighborArray.setNeighborBond(ni1, ni2,
                                                  pbcdis_sq(box, x, y, z, verlet_list(i, ni1), verlet_list(i, ni2)) <= localCutoffSquared);
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

            if (n421 == 12)
                pattern(i) = 1;
            else if (n421 == 6 && n422 == 6)
                pattern(i) = 2;
            else if (n555 == 12)
                pattern(i) = 4;
            else if (n666 == 8 && n444 == 6)
                pattern(i) = 3;
        }
    }
}

NB_MODULE(_cna, m)
{
    m.def("acna", &AdaptiveCNA);
    m.def("fcna", &FixedCNA);
    m.def("ids", &IdentifyDiamond);
}
