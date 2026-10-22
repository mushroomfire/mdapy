/*Copyright (c) 2021 PM Larsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef PTM_SCALED_TEMPLATES_H
#define PTM_SCALED_TEMPLATES_H

const int8_t ptm_scaled_template_bcc[PTM_NUM_POINTS_BCC][3] = {
           {   0,    0,    0},
           {   1,    1,    1},
           {  -1,    1,    1},
           {   1,    1,   -1},
           {  -1,   -1,    1},
           {   1,   -1,    1},
           {  -1,    1,   -1},
           {  -1,   -1,   -1},
           {   1,   -1,   -1},
           {   2,    0,    0},
           {  -2,    0,    0},
           {   0,    2,    0},
           {   0,   -2,    0},
           {   0,    0,    2},
           {   0,    0,   -2},};

const int8_t ptm_scaled_template_dcub[PTM_NUM_POINTS_DCUB][3] = {
           {   0,    0,    0},
           {   1,    1,    1},
           {   1,   -1,   -1},
           {  -1,   -1,    1},
           {  -1,    1,   -1},
           {   2,    2,    0},
           {   0,    2,    2},
           {   2,    0,    2},
           {   0,   -2,   -2},
           {   2,   -2,    0},
           {   2,    0,   -2},
           {  -2,   -2,    0},
           {   0,   -2,    2},
           {  -2,    0,    2},
           {  -2,    0,   -2},
           {  -2,    2,    0},
           {   0,    2,   -2},};

const int8_t ptm_scaled_template_dcub_alt1[PTM_NUM_POINTS_DCUB][3] = {
           {   0,    0,    0},
           {   1,   -1,    1},
           {   1,    1,   -1},
           {  -1,   -1,   -1},
           {  -1,    1,    1},
           {   2,    0,    2},
           {   0,   -2,    2},
           {   2,   -2,    0},
           {   0,    2,   -2},
           {   2,    0,   -2},
           {   2,    2,    0},
           {  -2,    0,   -2},
           {   0,   -2,   -2},
           {  -2,   -2,    0},
           {  -2,    2,    0},
           {  -2,    0,    2},
           {   0,    2,    2},};

const int8_t ptm_scaled_template_dhex[PTM_NUM_POINTS_DHEX][3] = {
           {   0,    0,    0},
           {  -8,    8,   -3},
           {  -8,  -16,   -3},
           {  16,    8,   -3},
           {   0,    0,    9},
           { -24,    0,    0},
           {  -8,    8,  -12},
           {   0,   24,    0},
           {   0,  -24,    0},
           {  -8,  -16,  -12},
           { -24,  -24,    0},
           {  16,    8,  -12},
           {  24,   24,    0},
           {  24,    0,    0},
           {  -8,  -16,   12},
           {  16,    8,   12},
           {  -8,    8,   12},};

const int8_t ptm_scaled_template_dhex_alt1[PTM_NUM_POINTS_DHEX][3] = {
           {   0,    0,    0},
           { -16,   -8,   -3},
           {   8,   -8,   -3},
           {   8,   16,   -3},
           {   0,    0,    9},
           { -24,  -24,    0},
           { -16,   -8,  -12},
           { -24,    0,    0},
           {  24,    0,    0},
           {   8,   -8,  -12},
           {   0,  -24,    0},
           {   8,   16,  -12},
           {   0,   24,    0},
           {  24,   24,    0},
           {   8,   -8,   12},
           {   8,   16,   12},
           { -16,   -8,   12},};

const int8_t ptm_scaled_template_dhex_alt2[PTM_NUM_POINTS_DHEX][3] = {
           {   0,    0,    0},
           {  -8,  -16,    3},
           {  -8,    8,    3},
           {  16,    8,    3},
           {   0,    0,   -9},
           { -24,  -24,    0},
           {  -8,  -16,   12},
           {   0,  -24,    0},
           {   0,   24,    0},
           {  -8,    8,   12},
           { -24,    0,    0},
           {  16,    8,   12},
           {  24,    0,    0},
           {  24,   24,    0},
           {  -8,    8,  -12},
           {  16,    8,  -12},
           {  -8,  -16,  -12},};

const int8_t ptm_scaled_template_dhex_alt3[PTM_NUM_POINTS_DHEX][3] = {
           {   0,    0,    0},
           {   8,   -8,    3},
           { -16,   -8,    3},
           {   8,   16,    3},
           {   0,    0,   -9},
           {   0,  -24,    0},
           {   8,   -8,   12},
           {  24,    0,    0},
           { -24,    0,    0},
           { -16,   -8,   12},
           { -24,  -24,    0},
           {   8,   16,   12},
           {  24,   24,    0},
           {   0,   24,    0},
           { -16,   -8,  -12},
           {   8,   16,  -12},
           {   8,   -8,  -12},};

const int8_t ptm_scaled_template_fcc[PTM_NUM_POINTS_FCC][3] = {
           {   0,    0,    0},
           {   1,    1,    0},
           {   0,    1,    1},
           {   1,    0,    1},
           {  -1,   -1,    0},
           {   0,   -1,   -1},
           {  -1,    0,   -1},
           {  -1,    1,    0},
           {   0,   -1,    1},
           {  -1,    0,    1},
           {   1,   -1,    0},
           {   0,    1,   -1},
           {   1,    0,   -1},};

const int8_t ptm_scaled_template_graphene[PTM_NUM_POINTS_GRAPHENE][3] = {
           {   0,    0,    0},
           {   1,    2,    0},
           {   1,   -1,    0},
           {  -2,   -1,    0},
           {   0,    3,    0},
           {   3,    3,    0},
           {   3,    0,    0},
           {   0,   -3,    0},
           {  -3,   -3,    0},
           {  -3,    0,    0},};

const int8_t ptm_scaled_template_graphene_alt1[PTM_NUM_POINTS_GRAPHENE][3] = {
           {   0,    0,    0},
           {  -1,    1,    0},
           {   2,    1,    0},
           {  -1,   -2,    0},
           {  -3,    0,    0},
           {   0,    3,    0},
           {   3,    3,    0},
           {   3,    0,    0},
           {   0,   -3,    0},
           {  -3,   -3,    0},};

const int8_t ptm_scaled_template_hcp[PTM_NUM_POINTS_HCP][3] = {
           {   0,    0,    0},
           {   0,   -6,    0},
           {  -6,    0,    0},
           {  -2,    2,   -3},
           {   4,    2,   -3},
           {  -2,   -4,   -3},
           {   0,    6,    0},
           {   6,    6,    0},
           {   6,    0,    0},
           {  -6,   -6,    0},
           {  -2,   -4,    3},
           {   4,    2,    3},
           {  -2,    2,    3},};

const int8_t ptm_scaled_template_hcp_alt1[PTM_NUM_POINTS_HCP][3] = {
           {   0,    0,    0},
           {   6,    0,    0},
           {  -6,   -6,    0},
           {  -4,   -2,   -3},
           {   2,    4,   -3},
           {   2,   -2,   -3},
           {  -6,    0,    0},
           {   0,    6,    0},
           {   6,    6,    0},
           {   0,   -6,    0},
           {   2,   -2,    3},
           {   2,    4,    3},
           {  -4,   -2,    3},};

const int8_t ptm_scaled_template_sc[PTM_NUM_POINTS_SC][3] = {
           {   0,    0,    0},
           {   0,    0,   -1},
           {   0,    0,    1},
           {   0,   -1,    0},
           {   0,    1,    0},
           {  -1,    0,    0},
           {   1,    0,    0},};

#endif
