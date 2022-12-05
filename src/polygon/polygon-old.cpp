// Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
// This file is from the mdapy project, released under the BSD 3-Clause License.

#include "polygon.hpp"
#include "voro++.hh"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <tuple>
#include <pybind11/stl.h>
#include <cmath>

namespace py = pybind11;
using namespace voro;

tuple<vector<vector<vector<int>>>, vector<vector<vector<double>>>, vector<double>, vector<double>> get_cell_info(py::array pos, py::array box, py::array boundary)
{
	voronoicell_neighbor c;

	int tnx, tny, tnz, ti;
	auto c_pos = pos.mutable_unchecked<double, 2>();
	auto c_box = box.mutable_unchecked<double, 2>();
	auto c_boundary = boundary.mutable_unchecked<bool, 1>();
	int N = c_pos.shape(0);
	vector<vector<vector<int>>> face_vertices;
	vector<vector<vector<double>>> vertices_pos;
	vector<double> vertice2seed, volume, cavity_radius;

	pre_container pcon(c_box(0, 0), c_box(0, 1), c_box(1, 0), c_box(1, 1), c_box(2, 0), c_box(2, 1), c_boundary(0), c_boundary(1), c_boundary(2));
	for (int i = 0; i < N; i++)
	{
		pcon.put(i, c_pos(i, 0), c_pos(i, 1), c_pos(i, 2));
	}
	pcon.guess_optimal(tnx, tny, tnz);
	container con(c_box(0, 0), c_box(0, 1), c_box(1, 0), c_box(1, 1), c_box(2, 0), c_box(2, 1), tnx, tny, tnz, c_boundary(0), c_boundary(1), c_boundary(2), 8);
	pcon.setup(con);
	c_loop_all cl(con);
	if (cl.start())
		do
			if (con.compute_cell(c, cl))
			{
				ti = cl.pid();
				// get cell volume
				volume.emplace_back(c.volume());
				// get maximum of cavity_radius
				cavity_radius.emplace_back(sqrt(c.max_radius_squared()));
				// get face_vertices
				vector<int> face_index_vector;
				c.face_vertices(face_index_vector);
				face_vertices.emplace_back(vector<vector<int>>());
				int j = 0;
				while (j < face_index_vector.size())
				{
					vector<int> temp;
					for (int i = 0; i < face_index_vector[j]; i++)
					{
						temp.emplace_back(face_index_vector[j + 1 + i]);
					}
					j = j + face_index_vector[j] + 1;
					face_vertices[ti].emplace_back(temp);
				}
				// get vertices_pos
				c.vertices(0., 0., 0., vertice2seed);
				int nverts = int(vertice2seed.size()) / 3;
				vertices_pos.emplace_back(vector<vector<double>>());
				for (int si = 0; si < nverts; si++)
				{
					vector<double> temp;
					int m = 0;
					for (int vi = si * 3; vi < (si * 3 + 3); vi++)
					{
						temp.emplace_back(vertice2seed[vi] + c_pos(ti, m));
						m++;
					}
					vertices_pos[ti].emplace_back(temp);
				}
			}
		while (cl.inc());
	return {face_vertices, vertices_pos, volume, cavity_radius};
}
