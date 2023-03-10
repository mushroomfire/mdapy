/*Copyright (c) 2016 PM Larsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/*
	structures:
		+ no match for each of them
		nonzero rmsd

	add permutations of points
		with alloy numbers too

done:
	SC, FCC, HCP, ICO, BCC
		translation offset
		scale
		rotation
		zero rmsd
		orientations + fundamental zones
		strain rotation must equal to rmsd rotation if no anisotropic distortion, when mapped into fundamental zone

	alloys
		disordered + ordered

	deformation gradients

	strains + polar decomposition rotations

	check det(U) > 0
*/

#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <cassert>
#include <cmath>
#include <algorithm>
#include "ptm_normalize_vertices.h"
#include "ptm_quat.h"
#include "ptm_functions.h"


namespace ptm {

#define RADIANS(x) (2.0 * M_PI * (x) / 360.0)
#define DEGREES(x) (360 * (x) / (2.0 * M_PI))


typedef struct
{
	int type;
	int check;
	int num_points;
	const double (*points)[3];
} structdata_t;

//                              { .type = PTM_MATCH_SC,  .check = PTM_CHECK_SC,  .num_points =  7, .points = ptm_template_sc },
structdata_t structdata[8] =  {	{ PTM_MATCH_FCC,  PTM_CHECK_FCC,  13, ptm_template_fcc  },
				{ PTM_MATCH_HCP,  PTM_CHECK_HCP,  13, ptm_template_hcp  },
				{ PTM_MATCH_BCC,  PTM_CHECK_BCC,  15, ptm_template_bcc  },
				{ PTM_MATCH_ICO,  PTM_CHECK_ICO,  13, ptm_template_ico  },
				{ PTM_MATCH_SC,   PTM_CHECK_SC,    7, ptm_template_sc   },
				{ PTM_MATCH_DCUB, PTM_CHECK_DCUB, 17, ptm_template_dcub },
				{ PTM_MATCH_DHEX, PTM_CHECK_DHEX, 17, ptm_template_dhex },
				{ PTM_MATCH_GRAPHENE, PTM_CHECK_GRAPHENE, 10, ptm_template_graphene },
};

typedef struct
{
	int32_t type;
	int32_t numbers[PTM_MAX_POINTS];
} alloytest_t;

alloytest_t sc_alloy_tests[] = {

	{ PTM_ALLOY_NONE,   {-1}},	//no test
};

alloytest_t fcc_alloy_tests[] = {

	{ PTM_ALLOY_NONE,   {-1}},	//no test

	{ PTM_ALLOY_NONE,   {0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},	//pure -defect
	{ PTM_ALLOY_NONE,   {4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4}},	//pure -defect
	{ PTM_ALLOY_NONE,   {3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 7}},	//L10 -defect
	{ PTM_ALLOY_NONE,   {3, 0, 3, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0}},	//L10 -defect
	{ PTM_ALLOY_NONE,   {3, 0, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3}},	//L10 -defect
	{ PTM_ALLOY_NONE,   {0, 3, 1, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0}},	//L12_CU -defect
	{ PTM_ALLOY_NONE,   {5, 5, 3, 5, 5, 3, 3, 3, 3, 5, 5, 5, 5}},	//L12_CU -defect
	{ PTM_ALLOY_NONE,   {9, 9, 2, 9, 9, 9, 9, 9, 9, 3, 3, 3, 3}},	//L12_CU -defect
	{ PTM_ALLOY_NONE,   {4, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},	//L12_AU -defect
	{ PTM_ALLOY_NONE,   {1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}},	//L12_AU -defect

	{ PTM_ALLOY_PURE,   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	{ PTM_ALLOY_PURE,   {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}},
	{ PTM_ALLOY_L10,    {3, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0}},
	{ PTM_ALLOY_L10,    {3, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0}},
	{ PTM_ALLOY_L10,    {3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3}},
	{ PTM_ALLOY_L12_CU, {0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0}},
	{ PTM_ALLOY_L12_CU, {5, 5, 3, 5, 5, 3, 5, 5, 3, 5, 5, 3, 5}},
	{ PTM_ALLOY_L12_CU, {9, 9, 9, 3, 9, 9, 3, 9, 9, 3, 9, 9, 3}},
	{ PTM_ALLOY_L12_AU, {4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	{ PTM_ALLOY_L12_AU, {1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}},
};

alloytest_t hcp_alloy_tests[] = {

	{ PTM_ALLOY_NONE,   {-1}},	//no test
};

alloytest_t ico_alloy_tests[] = {

	{ PTM_ALLOY_NONE,   {-1}},	//no test
};

alloytest_t bcc_alloy_tests[] = {

	{ PTM_ALLOY_NONE,   {-1}},	//no test

	{ PTM_ALLOY_NONE,   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}},	//pure -defect
	{ PTM_ALLOY_NONE,   {4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}},	//pure -defect
	{ PTM_ALLOY_NONE,   {4, 0, 0, 0, 0, 4, 0, 0, 0, 4, 4, 4, 4, 4, 4}},	//B2 -defect
	{ PTM_ALLOY_NONE,   {1, 2, 2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1}},	//B2 -defect

	{ PTM_ALLOY_PURE,   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	{ PTM_ALLOY_PURE,   {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}},
	{ PTM_ALLOY_B2,     {4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4}},
	{ PTM_ALLOY_B2,     {1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1}},
};

alloytest_t dcub_alloy_tests[] = {

	{ PTM_ALLOY_NONE,   {-1}},	//no test

	{ PTM_ALLOY_NONE,   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}},	//pure -defect
	{ PTM_ALLOY_NONE,   {4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}},	//pure -defect
	{ PTM_ALLOY_NONE,   {4, 0, 0, 0, 0, 4, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4}},	//SiC -defect
	{ PTM_ALLOY_NONE,   {1, 2, 2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1}},	//SiC -defect

	{ PTM_ALLOY_PURE,   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	{ PTM_ALLOY_PURE,   {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}},
	{ PTM_ALLOY_SIC,    {4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}},
	{ PTM_ALLOY_SIC,    {1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},

};

alloytest_t dhex_alloy_tests[] = {

	{ PTM_ALLOY_NONE,   {-1}},	//no test

	{ PTM_ALLOY_NONE,   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}},	//pure -defect
	{ PTM_ALLOY_NONE,   {4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}},	//pure -defect

	{ PTM_ALLOY_PURE,   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	{ PTM_ALLOY_PURE,   {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}},
};


alloytest_t graphene_alloy_tests[] = {

	{ PTM_ALLOY_NONE,   {-1}},	//no test

	{ PTM_ALLOY_NONE,   {0, 0, 0, 0, 0, 0, 1, 0, 0, 0}},	//pure -defect
	{ PTM_ALLOY_NONE,   {4, 1, 4, 4, 4, 4, 4, 4, 4, 4}},	//pure -defect
	{ PTM_ALLOY_NONE,   {4, 0, 0, 0, 4, 4, 0, 4, 4, 4}},	//BN -defect
	{ PTM_ALLOY_NONE,   {1, 2, 2, 2, 2, 3, 2, 2, 2, 1}},	//BN -defect

	{ PTM_ALLOY_PURE,   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	{ PTM_ALLOY_PURE,   {4, 4, 4, 4, 4, 4, 4, 4, 4, 4}},
	{ PTM_ALLOY_BN,     {4, 0, 0, 0, 4, 4, 4, 4, 4, 4}},
	{ PTM_ALLOY_BN,     {1, 2, 2, 2, 1, 1, 1, 1, 1, 1}},
};


typedef struct
{
	bool strain;
	double pre[9];
	double post[9];
} quattest_t;

quattest_t cubic_qtest[] = {	{	false, {1.00, 0.00, -0.00, 0.00},			{1.00, 0.00, -0.00, 0.00}			},
				{	false, {0.00, 1.00, -0.00, 0.00},			{1.00, 0.00, -0.00, 0.00}			},
				{	false, {0.00, 0.00, -1.00, 0.00},			{1.00, 0.00, -0.00, 0.00}			},
				{	false, {0.00, 0.00, -0.00, 1.00},			{1.00, 0.00, -0.00, 0.00}			},
				{	false, {0.987070, 0.020780, -0.031171, 0.155853},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.683270, 0.712658, 0.088164, 0.132246},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.720005, -0.095511, 0.675923, 0.124899},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.587759, -0.007347, -0.036735, 0.808168},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.808168, 0.036735, -0.007347, -0.587759},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.675923, 0.124899, -0.720005, 0.095511},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.712658, -0.683270, -0.132246, 0.088164},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.420803, 0.410413, 0.545486, 0.597437},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.576656, 0.441584, 0.566266, -0.389633},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.389633, 0.566266, -0.441584, 0.576656},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.545486, 0.597437, -0.420803, -0.410413},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.441584, -0.576656, 0.389633, 0.566266},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.597437, -0.545486, 0.410413, -0.420803},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.410413, -0.420803, -0.597437, 0.545486},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.566266, -0.389633, -0.576656, -0.441584},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.020780, -0.987070, -0.155853, -0.031171},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.007347, 0.587759, 0.808168, 0.036735},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.124899, -0.675923, -0.095511, -0.720005},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.095511, 0.720005, 0.124899, -0.675923},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.036735, -0.808168, 0.587759, -0.007347},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.031171, -0.155853, 0.987070, 0.020780},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.088164, 0.132246, -0.683270, -0.712658},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.132246, -0.088164, 0.712658, -0.683270},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	false, {0.155853, 0.031171, 0.020780, -0.987070},	{0.987070, 0.020780, -0.031171, 0.155853}	},
				{	true,  {1.1, 0.0, 0.0, -0.0, 1.0, -0.0, -0.0, 0.0, 1.0},	{-1, -1, -1, -1}			},
				{	true,  {1.05, 0.0, 0.0, -0.05, 0.95, -0.0, -0.02, 0.06, 0.94},	{-1, -1, -1, -1}			}	};

quattest_t ico_qtest[] = {	{	false, {0.103781, -0.709767, -0.544366, 0.434886},	{0.946549, -0.235192, -0.181237, -0.126026}	},
				{	false, {0.367185, 0.677864, -0.636562, 0.021554},	{0.957099, 0.037090, -0.186490, 0.218649}	},
				{	false, {0.190755, 0.875270, 0.172544, -0.409564},	{0.968785, -0.171273, -0.178955, 0.009798}	},
				{	false, {0.214153, -0.117098, -0.734199, -0.633545},	{0.960186, 0.190467, 0.005094, -0.204301}	},
				{	false, {0.423220, -0.522912, 0.065963, -0.736951},	{0.990945, -0.124176, -0.044386, -0.025249}	},
				{	false, {0.479500, -0.565900, -0.356178, 0.568308},	{0.971669, 0.109592, -0.205483, 0.040320}	},
				{	false, {0.680330, -0.529816, -0.505883, -0.023007},	{0.956954, 0.008905, 0.289826, -0.012650}	},
				{	false, {0.541528, 0.167491, -0.368713, -0.736713},	{0.986343, -0.023998, 0.158360, -0.038382}	},
				{	false, {0.902495, -0.222917, 0.333109, 0.157634},	{0.946696, -0.185034, -0.263576, 0.007488}	},
				{	false, {0.445357, 0.327588, 0.339277, 0.761075},	{0.991976, 0.116389, -0.029525, 0.039557}	},
				{	false, {0.354516, -0.481525, 0.799651, -0.054864},	{0.959420, -0.231172, 0.050190, -0.153471}	},
				{	false, {0.199248, 0.011552, -0.897134, 0.394104},	{0.970340, 0.136406, 0.175564, -0.094924}	},
				{	false, {0.844472, 0.269083, -0.281229, -0.367928},	{0.960138, -0.218696, 0.022905, -0.172576}	},
				{	false, {0.167894, 0.937494, -0.299389, 0.057298},	{0.937494, -0.167894, -0.057298, -0.299389}	},
				{	false, {0.750422, -0.527268, 0.095267, 0.387013},	{0.987948, -0.013526, 0.010065, 0.153866}	},
				{	false, {0.127694, -0.854596, -0.354997, -0.356843},	{0.975933, -0.217966, -0.001345, 0.006596}	},
				{	false, {0.471540, -0.350594, -0.240403, 0.772619},	{0.981832, 0.100397, 0.101439, 0.125045}	},
				{	false, {0.849248, 0.490675, -0.047140, -0.189193},	{0.989784, -0.009315, 0.065447, 0.126323}	},
				{	false, {0.539812, -0.821868, 0.181925, -0.006351},	{0.983664, 0.053669, 0.077447, 0.153383}	},
				{	false, {0.152879, 0.503682, 0.225435, 0.819824},	{0.985849, -0.135018, -0.006291, -0.099157}	},
				{	false, {0.662156, -0.548907, -0.470188, -0.197922},	{0.961265, 0.116446, 0.210810, -0.134046}	},
				{	false, {0.112504, -0.756711, -0.578261, -0.283454},	{0.952085, 0.257628, -0.163190, 0.023039}	},
				{	false, {0.149157, 0.352931, -0.012353, -0.923601},	{0.964134, 0.064962, 0.185811, 0.178042}	},
				{	false, {0.043936, -0.346654, 0.325834, 0.878483},	{0.973545, -0.112485, 0.137222, -0.143973}	},
				{	false, {0.412262, -0.594130, -0.382977, 0.574785},	{0.960977, 0.079247, -0.245995, 0.098644}	},
				{	false, {0.740561, 0.411247, 0.278195, -0.452827},	{0.963565, -0.015612, -0.058086, -0.260624}	},
				{	false, {0.231777, -0.600925, 0.729169, -0.231260},	{0.972254, 0.081175, 0.205778, 0.076084}	},
				{	false, {0.239717, -0.274414, -0.462570, -0.808246},	{0.957627, -0.037699, -0.190240, -0.212925}	},
				{	false, {0.255509, 0.664796, 0.126855, -0.690413},	{0.955897, 0.179780, -0.090383, -0.213943}	},
				{	false, {0.259564, 0.812528, 0.110003, 0.510220},	{0.983484, 0.128426, -0.101426, -0.077324}	},
				{	false, {0.817769, -0.230825, 0.186116, 0.493290},	{0.951537, -0.296138, 0.014895, -0.081592}	},
				{	false, {0.732367, -0.278766, 0.552232, 0.284550},	{0.957621, -0.221092, -0.011541, 0.184248}	},
				{	false, {0.333549, 0.832782, 0.328986, -0.294936},	{0.974904, 0.122463, 0.180129, -0.046034}	},
				{	false, {0.544500, -0.340298, 0.169793, 0.747587},	{0.964108, 0.199228, -0.149866, 0.091342}	},
				{	false, {0.141283, 0.815409, 0.491600, 0.271065},	{0.948635, 0.045027, -0.302340, -0.081573}	},
				{	false, {0.199176, -0.288193, -0.871345, 0.343557},	{0.951421, 0.144704, -0.082987, 0.258791}	},
				{	false, {0.854785, 0.467915, -0.122139, -0.188363},	{0.994850, -0.050346, 0.011239, 0.087253}	},
				{	false, {0.400491, 0.875614, 0.089294, -0.254820},	{0.973503, 0.061326, 0.003273, -0.220274}	},
				{	false, {0.186361, -0.516894, 0.404923, -0.730840},	{0.953643, 0.091192, 0.134590, 0.253248}	},
				{	false, {0.778485, 0.156941, 0.239332, -0.558615},	{0.958153, 0.267643, 0.101376, 0.005653}	},
				{	false, {0.660376, 0.209905, 0.592739, -0.410493},	{0.959131, -0.053033, 0.270334, -0.064617}	},
				{	false, {0.317279, -0.184247, 0.852733, 0.371797},	{0.981557, -0.103355, -0.084750, 0.136675}	},
				{	false, {0.292583, -0.885028, 0.262344, -0.249592},	{0.971804, -0.228798, -0.003881, -0.056861}	},
				{	false, {0.111907, 0.644420, 0.291936, 0.697835},	{0.963252, -0.190919, 0.181078, -0.053918}	},
				{	false, {0.465415, -0.040139, 0.154135, 0.870644},	{0.971852, -0.158995, 0.009456, -0.173592}	},
				{	false, {0.563950, 0.031183, 0.394821, 0.724641},	{0.958230, -0.122182, -0.244918, 0.082958}	},
				{	false, {0.976751, -0.058497, 0.171110, 0.115138},	{0.976751, -0.058497, 0.171110, 0.115138}	},
				{	false, {0.713546, -0.330358, -0.298234, 0.541085},	{0.948436, -0.198509, 0.142577, 0.201830}	},
				{	false, {0.086107, 0.481579, 0.128916, -0.862582},	{0.953602, 0.206744, 0.035624, 0.215943}	},
				{	false, {0.868157, -0.263057, -0.062338, -0.416195},	{0.953413, 0.272489, -0.048341, -0.120067}	},
				{	false, {0.093970, -0.598545, 0.779445, -0.159305},	{0.942380, -0.017614, -0.184115, -0.278769}	},
				{	false, {0.102984, 0.267494, -0.534110, -0.795341},	{0.957356, -0.036205, -0.173403, 0.228233}	},
				{	false, {0.268924, 0.304196, -0.682994, -0.607177},	{0.941340, -0.075679, 0.328735, 0.009229}	},
				{	false, {0.035537, 0.476906, -0.647822, 0.592979},	{0.947469, -0.002944, -0.269381, 0.172417}	},
				{	false, {0.599786, -0.106945, 0.175667, -0.773279},	{0.939759, 0.016734, 0.204979, -0.273051}	},
				{	false, {0.407305, 0.793108, -0.413804, -0.183979},	{0.954528, -0.212072, -0.209293, -0.009909}	},
				{	false, {0.977982, 0.093890, 0.053592, 0.178504},	{0.977982, 0.093890, 0.053592, 0.178504}	},
				{	false, {0.353735, 0.780830, -0.396049, -0.329122},	{0.985106, -0.126823, -0.095242, -0.066408}	},
				{	false, {0.277306, -0.112211, 0.456831, 0.837744},	{0.952798, 0.051824, -0.241240, -0.176897}	},
				{	false, {0.936002, 0.144177, -0.220122, -0.233794},	{0.936002, 0.144177, -0.220122, -0.233794}	},
				{	true,  {1.1, 0.0, 0.0, -0.0, 1.0, -0.0, -0.0, 0.0, 1.0}, 	{-1, -1, -1, -1}			},
				{	true,  {1.05, 0.0, 0.0, -0.05, 0.95, -0.0, -0.02, 0.06, 0.94},	{-1, -1, -1, -1}			}	};


quattest_t hcp_qtest[] = {	{	false, {1.00, 0.00, -0.00, 0.00},			{1.00, 0.00, -0.00, 0.00}			},
				{	false, {0.813035, -0.166389, 0.181571, -0.527562},	{0.863399, 0.074051, 0.234882, 0.440328}	},
				{	false, {0.023227, -0.154944, -0.861953, -0.482172},	{0.861953, 0.482172, 0.023227, -0.154944}	},
				{	false, {0.631878, 0.693281, 0.293658, -0.184001},	{0.747228, -0.639223, -0.156589, -0.092325}	},
				{	false, {0.167480, -0.812443, 0.552710, 0.079998},	{0.979951, 0.185041, -0.014460, -0.072439}	},
				{	false, {0.002364, -0.201656, 0.941752, 0.269132},	{0.941752, 0.269132, -0.002364, 0.201656}	},
				{	false, {0.122621, -0.141644, 0.064344, 0.980184},	{0.910175, -0.126546, -0.090495, 0.383899}	},
				{	true,  {1.1, 0.0, 0.0, -0.0, 1.0, -0.0, -0.0, 0.0, 1.0}, 	{-1, -1, -1, -1}			},
				{	true,  {1.05, 0.0, 0.0, -0.05, 0.95, -0.0, -0.02, 0.06, 0.94},	{-1, -1, -1, -1}			}	};

//#ifdef DEBUG
#define ERROR(msg, code) print_error(__FILE__, __PRETTY_FUNCTION__, __LINE__, msg, code)
#define CLEANUP(msg, code) {ret = code; print_error(__FILE__, __PRETTY_FUNCTION__, __LINE__, msg, code); goto cleanup;}
//#else
//#define ERROR(msg, code) code
//#define CLEANUP(msg, code) {ret = code; goto cleanup;}
//#endif

static int print_error(const char* file, const char* function, int line, const char* msg, int error_code)
{
	printf("\n\nerror\tfile: %s\n", file);
	printf("\tline: %d\n", line);
	printf("\tfunction: %s\n", function);
	printf("\terror message: %s\n", msg);
	printf("\terror code: %d\n", error_code);
	(void)fflush(stdout);
	return error_code;
}

static void matvec(double* A, double* x, double* b)
{
	b[0] = A[0] * x[0] + A[1] * x[1] + A[2] * x[2];
	b[1] = A[3] * x[0] + A[4] * x[1] + A[5] * x[2];
	b[2] = A[6] * x[0] + A[7] * x[1] + A[8] * x[2];
}

static void matmul(double* A, double* x, double* b)
{
	b[0] = A[0] * x[0] + A[1] * x[3] + A[2] * x[6];
	b[3] = A[3] * x[0] + A[4] * x[3] + A[5] * x[6];
	b[6] = A[6] * x[0] + A[7] * x[3] + A[8] * x[6];

	b[1] = A[0] * x[1] + A[1] * x[4] + A[2] * x[7];
	b[4] = A[3] * x[1] + A[4] * x[4] + A[5] * x[7];
	b[7] = A[6] * x[1] + A[7] * x[4] + A[8] * x[7];

	b[2] = A[0] * x[2] + A[1] * x[5] + A[2] * x[8];
	b[5] = A[3] * x[2] + A[4] * x[5] + A[5] * x[8];
	b[8] = A[6] * x[2] + A[7] * x[5] + A[8] * x[8];
}

static bool check_matrix_equality(double* A, double* B, double tolerance)
{
	for (int i=0;i<9;i++)
		if (fabs(A[i] - B[i]) > tolerance)
			return false;
	return true;
}

static double matrix_determinant(double* A)
{
	return    A[0] * (A[4] * A[8] - A[5] * A[7])
		- A[1] * (A[3] * A[8] - A[5] * A[6])
		+ A[2] * (A[3] * A[7] - A[4] * A[6]);
}

static double nearest_neighbour_rmsd(int num, double scale, double* A, double (*input_points)[3], const double (*template_points)[3])
{
	//transform template
	double transformed_template[PTM_MAX_POINTS][3];
	for (int i=0;i<num;i++)
	{
		double row[3] = {0, 0, 0};
		matvec(A, (double*)template_points[i], row);
		memcpy(transformed_template[i], row, 3 * sizeof(double));
	}

	//translate and scale input points
	double points[PTM_MAX_POINTS][3];
	subtract_barycentre(num, input_points, points);
	for (int i=0;i<num;i++)
		for (int j=0;j<3;j++)
			points[i][j] *= scale;

	double acc = 0;
	for (int i=0;i<num;i++)
	{
		double x0 = points[i][0];
		double y0 = points[i][1];
		double z0 = points[i][2];

		double min_dist = INFINITY;
		for (int j=0;j<num;j++)
		{
			double x1 = transformed_template[j][0];
			double y1 = transformed_template[j][1];
			double z1 = transformed_template[j][2];

			double dx = x1 - x0;
			double dy = y1 - y0;
			double dz = z1 - z0;
			double dist = dx*dx + dy*dy + dz*dz;
			min_dist = std::min(min_dist, dist);
		}

		acc += min_dist;
	}

	return sqrt(fabs(acc / num));
}

static double mapped_neighbour_rmsd(int num, double scale, double* A, double (*input_points)[3], const double (*template_points)[3], int8_t* mapping)
{
	//transform template
	double transformed_template[PTM_MAX_POINTS][3];
	for (int i=0;i<num;i++)
	{
		double row[3] = {0, 0, 0};
		matvec(A, (double*)template_points[i], row);
		memcpy(transformed_template[i], row, 3 * sizeof(double));
	}

	//translate and scale input points
	double points[PTM_MAX_POINTS][3];
	subtract_barycentre(num, input_points, points);
	for (int i=0;i<num;i++)
		for (int j=0;j<3;j++)
			points[i][j] *= scale;

	double acc = 0;
	for (int i=0;i<num;i++)
	{
		double x0 = points[mapping[i]][0];
		double y0 = points[mapping[i]][1];
		double z0 = points[mapping[i]][2];

		double x1 = transformed_template[i][0];
		double y1 = transformed_template[i][1];
		double z1 = transformed_template[i][2];

		double dx = x1 - x0;
		double dy = y1 - y0;
		double dz = z1 - z0;
		double dist = dx*dx + dy*dy + dz*dz;
		acc += dist;
	}

	return sqrt(fabs(acc / num));
}

typedef double points_t[3];


typedef struct
{
	int num_points;
	double (*positions)[3];
	int32_t* numbers;

} unittest_nbrdata_t;

typedef struct
{
	int index;
	double dist;
	double offset[3];
	int32_t number;
} sorthelper_t;

static bool sorthelper_compare(sorthelper_t const& a, sorthelper_t const& b)
{
	return a.dist < b.dist;
}

static int get_neighbours(void* vdata, size_t central_index, size_t atom_index, int num, int* ordering, size_t* output_indices, int32_t* output_numbers, double (*output_pos)[3])
{
	unittest_nbrdata_t* nbrdata = (unittest_nbrdata_t*)vdata;

	int num_points = nbrdata->num_points;
	double (*positions)[3] = nbrdata->positions;
	int32_t* numbers = nbrdata->numbers;

	sorthelper_t data[PTM_MAX_POINTS];
	for (int i=0;i<num_points;i++)
	{
		double x0 = positions[atom_index][0];
		double y0 = positions[atom_index][1];
		double z0 = positions[atom_index][2];

		double x1 = positions[i][0];
		double y1 = positions[i][1];
		double z1 = positions[i][2];

		double dx = x1 - x0;
		double dy = y1 - y0;
		double dz = z1 - z0;
		double dist = dx*dx + dy*dy + dz*dz;

		data[i].dist = dist;
		data[i].index = i;
		data[i].number = numbers == NULL ? -1 : numbers[i];

		for (int j=0;j<3;j++)
			data[i].offset[j] = positions[i][j] - positions[atom_index][j];
	}

	std::sort(data, data + num_points, &sorthelper_compare);

	int n = std::min(num_points, num);
	for (int i=0;i<n;i++)
	{
		output_indices[i] = data[i].index;
		output_numbers[i] = data[i].number;
		ordering[i] = data[i].index;

		for (int j=0;j<3;j++)
			output_pos[i][j] = data[i].offset[j];
	}

	return n;
}

uint64_t run_tests()
{
	int ret = 0;
	const double tolerance = 1E-5;
	double identity_matrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
	int num_structures = sizeof(structdata) / sizeof(structdata_t);

	int num_alloy_tests[] = {	sizeof(fcc_alloy_tests) / sizeof(alloytest_t),
					sizeof(hcp_alloy_tests) / sizeof(alloytest_t),
					sizeof(bcc_alloy_tests) / sizeof(alloytest_t),
					sizeof(ico_alloy_tests) / sizeof(alloytest_t),
					sizeof(sc_alloy_tests) / sizeof(alloytest_t),
					sizeof(dcub_alloy_tests) / sizeof(alloytest_t),
					sizeof(dhex_alloy_tests) / sizeof(alloytest_t),
					sizeof(graphene_alloy_tests) / sizeof(alloytest_t)	};

	alloytest_t* alloy_test[] = {	fcc_alloy_tests,
					hcp_alloy_tests,
					bcc_alloy_tests,
					ico_alloy_tests,
					sc_alloy_tests,
					dcub_alloy_tests,
					dhex_alloy_tests,
					graphene_alloy_tests  };

	int num_quat_tests[] = {	sizeof(cubic_qtest) / sizeof(quattest_t),
					sizeof(hcp_qtest) / sizeof(quattest_t),
					sizeof(cubic_qtest) / sizeof(quattest_t),
					sizeof(ico_qtest) / sizeof(quattest_t),
					sizeof(cubic_qtest) / sizeof(quattest_t),
					1,
					1,
					1	};

	quattest_t* quat_test[] = {	cubic_qtest,
					hcp_qtest,
					cubic_qtest,
					ico_qtest,
					cubic_qtest,
					cubic_qtest,
					hcp_qtest,
					hcp_qtest };
	int num_tests = 0;
	ptm_local_handle_t local_handle = ptm_initialize_local();

	//rotation matrix => quaternion
	{
		double U[9] = {	 0,  0,  1,
				 0,  1,  0,
				-1,  0,  0  };
		double q[4];
		rotation_matrix_to_quaternion(U, q);
		double ans[4] = {1 / sqrt(2), 0, 1 / sqrt(2), 0};
		for (int i = 0;i<4;i++)
			if (fabs(q[i] - ans[i]) >= tolerance)
				CLEANUP("failed on rotation matrix => quaternion conversion", -1)
	}

	//quaternion => rotation matrix
	{
		double q[4] = {1 / sqrt(2), 1 / sqrt(2), 0, 0};
		double U[9];
		quaternion_to_rotation_matrix(q, U);

		double ans[9] = { 1,  0,  0,
				  0,  0, -1,
				  0,  1,  0	};

		for (int i = 0;i<9;i++)
			if (fabs(U[i] - ans[i]) >= tolerance)
				CLEANUP("failed on quaternion => rotation matrix conversion", -1)
	}

/*
	//for (int it = 0;it<num_structures;it++)
int it = 2;
	{
		structdata_t* s = &structdata[it];
		double points[PTM_MAX_POINTS][3];
		double rotated_points[PTM_MAX_POINTS][3];

		memcpy(points, s->points, 3 * sizeof(double) * s->num_points);

		//for (int k=0;k<24;k++)
		for (int k=0;k<6;k++)
		{
			double U[9];
			//extern double generator_cubic[24][4];
			//extern double generator_icosahedral[60][4];
			extern double generator_hcp[6][4];
			double qtemp[4];
			//memcpy(qtemp, generator_cubic[k], 4 * sizeof(double));
			//memcpy(qtemp, generator_icosahedral[k], 4 * sizeof(double));
			memcpy(qtemp, generator_hcp[k], 4 * sizeof(double));
			qtemp[1] = -qtemp[1];
			qtemp[2] = -qtemp[2];
			qtemp[3] = -qtemp[3];
			quaternion_to_rotation_matrix(qtemp, U);

			for (int i=0;i<s->num_points;i++)
			{
				double row[3] = {0, 0, 0};
				matvec(U, (double*)points[i], row);
				memcpy(rotated_points[i], row, 3 * sizeof(double));
			}

			for (int i=0;i<s->num_points;i++)
			{
				double x0 = rotated_points[i][0];
				double y0 = rotated_points[i][1];
				double z0 = rotated_points[i][2];

				int bi = -1;
				double min_dist = INFINITY;
				for (int j=0;j<s->num_points;j++)
				{
					double x1 = points[j][0];
					double y1 = points[j][1];
					double z1 = points[j][2];

					double dx = x1 - x0;
					double dy = y1 - y0;
					double dz = z1 - z0;
					double dist = dx*dx + dy*dy + dz*dz;
					if (dist < min_dist)
						bi = j;
					min_dist = MIN(min_dist, dist);
				}

				printf("%d ", bi);
			}
			printf("\n");
		}
	}
exit(3);*/


	for (int it = 0;it<num_structures;it++)
	{
		for (int iq=0;iq<num_quat_tests[it];iq++)
		{
			quattest_t* qtest = quat_test[it];

			double qpre[4], qpost[4], rot[9];
			memcpy(qpre, qtest[iq].pre, 4 * sizeof(double));
			memcpy(qpost, qtest[iq].post, 4 * sizeof(double));

			if (qtest[iq].strain)
			{
				memcpy(rot, qtest[iq].pre, 9 * sizeof(double));
			}
			else
			{
				normalize_quaternion(qpre);
				normalize_quaternion(qpost);
				quaternion_to_rotation_matrix(qpre, rot);
			}


			structdata_t* s = &structdata[it];
			double scaled_only[PTM_MAX_POINTS][3] = {0};
			double points[PTM_MAX_POINTS][3];
			memcpy(points, s->points, 3 * sizeof(double) * s->num_points);

			for (int i=0;i<s->num_points;i++)
			{
				double row[3] = {0, 0, 0};
				matvec(rot, points[i], row);
				memcpy(points[i], row, 3 * sizeof(double));
			}

			double _x = s->points[1][0];
			double _y = s->points[1][1];
			double _z = s->points[1][2];
			double norm = sqrt(_x*_x + _y*_y + _z*_z) / 2;
			double rescale = 1 / norm;
			double offset[3] = {12.0, 45.6, 789.10};
			for (int i = 0;i<s->num_points;i++)
				for (int j = 0;j<3;j++)
					scaled_only[i][j] = points[i][j] * rescale;

			for (int i = 0;i<s->num_points;i++)
				for (int j = 0;j<3;j++)
					points[i][j] = points[i][j] * rescale + offset[j];


			int tocheck = 0;
			for (int i = 0;i<it+1;i++)
				if (structdata[i].num_points <= structdata[it].num_points)
					tocheck |= structdata[i].check;

			for (int ia=0;ia<num_alloy_tests[it];ia++)
			{
				int32_t* numbers = alloy_test[it][ia].numbers;
				if (numbers[0] == -1)
					numbers = NULL;

				unittest_nbrdata_t nbrlist = {s->num_points, points, numbers};

				int8_t output_indices[PTM_MAX_POINTS];
				int32_t type, alloy_type;
				double scale, rmsd, interatomic_distance, lattice_constant;
				double q[4], F[9], F_res[3], U[9], P[9];
				ret = ptm_index(local_handle, 0, get_neighbours, (void*)&nbrlist, tocheck, false,
						&type, &alloy_type, &scale, &rmsd, q, F, F_res, U, P, &interatomic_distance, &lattice_constant, NULL, NULL, output_indices);

				if (ret != PTM_NO_ERROR)
					CLEANUP("indexing failed", ret);

				num_tests++;

#ifdef DEBUG
				printf("type:\t\t%d\t(should be: %d)\n", type, s->type);
				printf("alloy type:\t%d\n", alloy_type);
				printf("scale:\t\t%f\n", scale);
				printf("rmsd:\t\t%f\n", rmsd);
				printf("quat: \t\t%.6f %.6f %.6f %.6f\n", q[0], q[1], q[2], q[3]);
				printf("qpre:\t\t%.6f %.6f %.6f %.6f\n", qpre[0], qpre[1], qpre[2], qpre[3]);
				printf("qpost:\t\t%.6f %.6f %.6f %.6f\n", qpost[0], qpost[1], qpost[2], qpost[3]);
				printf("rot: %f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", rot[0], rot[1], rot[2], rot[3], rot[4], rot[5], rot[6], rot[7], rot[8]);
				printf("U:   %f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7], U[8]);
				printf("P:   %f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8]);
				printf("F:   %f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8]);
				//printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", rot[0] - U[0], rot[1] - U[1], rot[2] - U[2], rot[3] - U[3], rot[4] - U[4], rot[5] - U[5], rot[6] - U[6], rot[7] - U[7], rot[8] - U[8]);
				printf("interatomic distance:\t\t%f\n", interatomic_distance);
#endif

				//check type
				if (type != s->type)
					CLEANUP("failed on type", -1);

				//check alloy type
				if (alloy_type != alloy_test[it][ia].type)
					CLEANUP("failed on alloy type", -1);

				//check U-matrix is right handed
				if (matrix_determinant(U) <= 0)
					CLEANUP("failed on U-matrix right-handedness test", -1);

				//check strain tensor is symmetric
				if (fabs(P[1] - P[3]) > tolerance)	CLEANUP("failed on strain tensor symmetry test", -1);
				if (fabs(P[2] - P[6]) > tolerance)	CLEANUP("failed on strain tensor symmetry test", -1);
				if (fabs(P[5] - P[7]) > tolerance)	CLEANUP("failed on strain tensor symmetry test", -1);

				//check polar decomposition
				double _F[9];
				matmul(P, U, _F);
				if (!check_matrix_equality(_F, F, tolerance))
					CLEANUP("failed on polar decomposition check", -1);

				double A[9];
				if (!qtest[iq].strain)
				{
					//check rmsd
					if (rmsd > tolerance)
						CLEANUP("failed on rmsd", -1);

					//check scale
					if (fabs(scale - 1 / rescale) > tolerance)
						CLEANUP("failed on scale", -1);

					//check deformation gradient equal to polar decomposition rotation
					if (!check_matrix_equality(F, U, tolerance))
						CLEANUP("failed on deformation gradient check", -1);

					//check strain tensor is identity
					if (!check_matrix_equality(P, identity_matrix, tolerance))
						CLEANUP("failed on P identity matrix check", -1);

					//check rotation
					if (quat_misorientation(q, qpost) > tolerance)
						CLEANUP("failed on disorientation", -1);

					//check deformation gradient disorientation
					double qu[4];
					rotation_matrix_to_quaternion(U, qu);
					if (quat_misorientation(qu, qpost) > tolerance)
						CLEANUP("failed on deformation gradient disorientation", -1);

					quaternion_to_rotation_matrix(q, A);

					double x = scaled_only[1][0];
					double y = scaled_only[1][1];
					double z = scaled_only[1][2];
					double iad = sqrt(x*x + y*y + z*z);
					if (fabs(iad - interatomic_distance) > tolerance)
						CLEANUP("failed on interatomic distance", -1);
				}
				else
				{
					memcpy(A, F, 9 * sizeof(double));
				}

				//check nearest neighbour rmsd
				double rmsd_approx = nearest_neighbour_rmsd(s->num_points, scale, A, points, s->points);
				if (fabs(rmsd_approx) > tolerance)
					CLEANUP("failed on rmsd nearest neighbour", -1);

				//check mapped neighbour rmsd
				double rmsd_mapped = mapped_neighbour_rmsd(s->num_points, scale, A, points, s->points, output_indices);
				if (fabs(rmsd_mapped) > tolerance)
					CLEANUP("failed on rmsd mapped neighbour", -1);
			}
		}
	}


	{
		double lc_points_sc[7][3] = {{0,0,0},{2,0,0},{-2,0,0},{0,2,0},{0,-2,0},{0,0,2},{0,0,-2}};
		double lc_points_fcc[13][3] = {{0,0,0},{0,1,1},{0,-1,-1},{0,1,-1},{0,-1,1},{1,0,1},{-1,0,-1},{1,0,-1},{-1,0,1},{1,1,0},{-1,-1,0},{1,-1,0},{-1,1,0}};
		double lc_points_bcc[15][3] = {{0,0,0},{1,1,1},{1,1,-1},{1,-1,1},{1,-1,-1},{-1,1,1},{-1,1,-1},{-1,-1,1},{-1,-1,-1},{2,0,0},{-2,0,0},{0,2,0},{0,-2,0},{0,0,2},{0,0,-2}};
		double lc_points_dcub[17][3] = {{0,0,0},{-0.5,0.5,0.5},{-0.5,-0.5,-0.5},{0.5,-0.5,0.5},{0.5,0.5,-0.5},{-1,0,1},{-1,1,0},{0,1,1},{-1,-1,0},{-1,0,-1},{0,-1,-1},{0,-1,1},{1,-1,0},{1,0,1},{0,1,-1},{1,0,-1},{1,1,0}};
		points_t* pdata[4] = {lc_points_fcc, lc_points_bcc, lc_points_sc, lc_points_dcub};

		//double* (*pdata)[3] = {lc_points_fcc, lc_points_bcc, lc_points_sc};

		int lcdat[4] = {0, 2, 4, 5};
		for (int i=0;i<4;i++)
		{
			structdata_t* s = &structdata[lcdat[i]];

			int32_t type;
			double scale, rmsd, interatomic_distance, lattice_constant, q[4];

			unittest_nbrdata_t nbrlist = {s->num_points, pdata[i], NULL};
			ret = ptm_index(local_handle, 0, get_neighbours, (void*)&nbrlist, s->check, false,
					&type, NULL, &scale, &rmsd, q, NULL, NULL, NULL, NULL, &interatomic_distance, &lattice_constant, NULL, NULL, NULL);
			if (ret != PTM_NO_ERROR)
				CLEANUP("indexing failed", ret);

			if (type != s->type)
				CLEANUP("failed on type", -1);

			if (fabs(lattice_constant - 2) > tolerance)
				CLEANUP("failed on lattice constant", -1);

			double x = pdata[i][1][0];
			double y = pdata[i][1][1];
			double z = pdata[i][1][2];
			double iad = sqrt(x*x + y*y + z*z);
			if (fabs(iad - interatomic_distance) > tolerance)
				CLEANUP("failed on interatomic distance", -1);

			num_tests++;
		}
	}

	//conventional deformation gradients
	for (int i=0;i<8;i++)
	{
		structdata_t* s = &structdata[i];

		int32_t type;
		double scale, rmsd, interatomic_distance, lattice_constant, q[4], F[9], F_res[3];

		double points[PTM_MAX_POINTS][3];
		memcpy(points, s->points, 3 * sizeof(double) * s->num_points);
		unittest_nbrdata_t nbrlist = {s->num_points, points, NULL};

		ret = ptm_index(local_handle, 0, get_neighbours, (void*)&nbrlist, s->check, true,
				&type, NULL, &scale, &rmsd, q, F, F_res, NULL, NULL, &interatomic_distance, &lattice_constant, NULL, NULL, NULL);
		if (ret != PTM_NO_ERROR)
			CLEANUP("indexing failed", ret);

		//check type
		if (type != s->type)
			CLEANUP("failed on type", -1);

		//check rmsd
		if (rmsd > tolerance)
			CLEANUP("failed on rmsd", -1);

		//check deformation gradient residual
		double d = sqrt(F_res[0] * F_res[0] + F_res[1] * F_res[1] + F_res[2] * F_res[2]);
		if (d > tolerance)
			CLEANUP("failed on deformation gradient residual", -1)

		//check deformation gradient is identity
		if (!check_matrix_equality(F, identity_matrix, tolerance))
			CLEANUP("failed on F identity matrix check", -1);
	}

	{
		const void* alt_templates[6] = { ptm_template_hcp_alt1, ptm_template_dcub_alt1, ptm_template_dhex_alt1, ptm_template_dhex_alt2, ptm_template_dhex_alt3, ptm_template_graphene_alt1};
		int num_points[6] = {13, 17, 17, 17, 17, 10};
		int32_t checks[6] = {PTM_CHECK_HCP, PTM_CHECK_DCUB, PTM_CHECK_DHEX, PTM_CHECK_DHEX, PTM_CHECK_DHEX, PTM_CHECK_GRAPHENE};
		int32_t types[6] = {PTM_MATCH_HCP, PTM_MATCH_DCUB, PTM_MATCH_DHEX, PTM_MATCH_DHEX, PTM_MATCH_DHEX, PTM_MATCH_GRAPHENE};

		for (int i=0;i<6;i++)
		{
			int32_t type;
			double scale, rmsd, interatomic_distance, lattice_constant, q[4], F[9], F_res[3];

			double points[PTM_MAX_POINTS][3];
			memcpy(points, alt_templates[i], 3 * sizeof(double) * num_points[i]);
			unittest_nbrdata_t nbrlist = {num_points[i], points, NULL};

			ret = ptm_index(local_handle, 0, get_neighbours, (void*)&nbrlist, checks[i], true,
					&type, NULL, &scale, &rmsd, q, F, F_res, NULL, NULL, &interatomic_distance, &lattice_constant, NULL, NULL, NULL);
			if (ret != PTM_NO_ERROR)
				CLEANUP("indexing failed", ret);

			//check type
			if (type != types[i])
				CLEANUP("failed on type", -1);

			//check rmsd
			if (rmsd > tolerance)
				CLEANUP("failed on rmsd", -1);

			//check deformation gradient residual
			double d = sqrt(F_res[0] * F_res[0] + F_res[1] * F_res[1] + F_res[2] * F_res[2]);
			if (d > tolerance)
				CLEANUP("failed on deformation gradient residual", -1)

			//check deformation gradient is identity
			if (!check_matrix_equality(F, identity_matrix, tolerance))
				CLEANUP("failed on F identity matrix check", -1);
		}
	}

cleanup:
	printf("num tests completed: %d\n", num_tests);
	ptm_uninitialize_local(local_handle);
	return ret;
}

}

