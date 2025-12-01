/*Copyright (c) 2022 PM Larsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef PTM_QUAT_H
#define PTM_QUAT_H

namespace ptm {

const double generator_cubic[24][4] = {
        {          1,          0,          0,          0 },
        {  sqrt(2)/2,  sqrt(2)/2,          0,          0 },
        {  sqrt(2)/2,          0,  sqrt(2)/2,          0 },
        {  sqrt(2)/2,          0,          0,  sqrt(2)/2 },
        {  sqrt(2)/2,          0,          0, -sqrt(2)/2 },
        {  sqrt(2)/2,          0, -sqrt(2)/2,          0 },
        {  sqrt(2)/2, -sqrt(2)/2,         -0,         -0 },
        {        0.5,        0.5,        0.5,        0.5 },
        {        0.5,        0.5,        0.5,       -0.5 },
        {        0.5,        0.5,       -0.5,        0.5 },
        {        0.5,        0.5,       -0.5,       -0.5 },
        {        0.5,       -0.5,        0.5,        0.5 },
        {        0.5,       -0.5,        0.5,       -0.5 },
        {        0.5,       -0.5,       -0.5,        0.5 },
        {        0.5,       -0.5,       -0.5,       -0.5 },
        {          0,          1,          0,          0 },
        {          0,  sqrt(2)/2,  sqrt(2)/2,          0 },
        {          0,  sqrt(2)/2,          0,  sqrt(2)/2 },
        {          0,  sqrt(2)/2,          0, -sqrt(2)/2 },
        {          0,  sqrt(2)/2, -sqrt(2)/2,          0 },
        {          0,          0,          1,          0 },
        {          0,          0,  sqrt(2)/2,  sqrt(2)/2 },
        {          0,          0,  sqrt(2)/2, -sqrt(2)/2 },
        {          0,          0,          0,          1 },
};

const double generator_diamond_cubic[12][4] = {
        {    1,    0,    0,    0 },
        {  0.5,  0.5,  0.5,  0.5 },
        {  0.5,  0.5,  0.5, -0.5 },
        {  0.5,  0.5, -0.5,  0.5 },
        {  0.5,  0.5, -0.5, -0.5 },
        {  0.5, -0.5,  0.5,  0.5 },
        {  0.5, -0.5,  0.5, -0.5 },
        {  0.5, -0.5, -0.5,  0.5 },
        {  0.5, -0.5, -0.5, -0.5 },
        {    0,    1,    0,    0 },
        {    0,    0,    1,    0 },
        {    0,    0,    0,    1 },
};

const double generator_hcp[6][4] = {
        {          1,          0,          0,          0 },
        {        0.5,          0,          0,  sqrt(3)/2 },
        {        0.5,          0,          0, -sqrt(3)/2 },
        {          0,  sqrt(3)/2,        0.5,          0 },
        {          0,  sqrt(3)/2,       -0.5,          0 },
        {          0,          0,          1,          0 },
};


const double generator_hcp_conventional[12][4] = {
        {          1,          0,          0,          0 },
        {  sqrt(3)/2,          0,          0,        0.5 },
        {  sqrt(3)/2,          0,          0,       -0.5 },
        {        0.5,          0,          0,  sqrt(3)/2 },
        {        0.5,          0,          0, -sqrt(3)/2 },
        {          0,          1,          0,          0 },
        {          0,  sqrt(3)/2,        0.5,          0 },
        {          0,  sqrt(3)/2,       -0.5,          0 },
        {          0,        0.5,  sqrt(3)/2,          0 },
        {          0,        0.5, -sqrt(3)/2,          0 },
        {          0,          0,          1,          0 },
        {          0,          0,          0,          1 },
};

const double generator_diamond_hexagonal[3][4] = {
        {          1,          0,          0,          0 },
        {        0.5,          0,          0,  sqrt(3)/2 },
        {        0.5,          0,          0, -sqrt(3)/2 },
};

const double generator_icosahedral[60][4] = {
        {                        1,                        0,                        0,                        0 },
        {            (1+sqrt(5))/4,                      0.5,   sqrt(25-10*sqrt(5))/10,   sqrt(50-10*sqrt(5))/20 },
        {            (1+sqrt(5))/4,                      0.5,  -sqrt(25-10*sqrt(5))/10,  -sqrt(50-10*sqrt(5))/20 },
        {            (1+sqrt(5))/4,            1/(1+sqrt(5)),   sqrt(10*sqrt(5)+50)/20,  -sqrt(50-10*sqrt(5))/20 },
        {            (1+sqrt(5))/4,            1/(1+sqrt(5)),  -sqrt(10*sqrt(5)+50)/20,   sqrt(50-10*sqrt(5))/20 },
        {            (1+sqrt(5))/4,                        0,   sqrt(50-10*sqrt(5))/10,   sqrt(50-10*sqrt(5))/20 },
        {            (1+sqrt(5))/4,                        0,                        0,     sqrt(5./8-sqrt(5)/8) },
        {            (1+sqrt(5))/4,                        0,                        0,    -sqrt(5./8-sqrt(5)/8) },
        {            (1+sqrt(5))/4,                        0,  -sqrt(50-10*sqrt(5))/10,  -sqrt(50-10*sqrt(5))/20 },
        {            (1+sqrt(5))/4,           -1/(1+sqrt(5)),   sqrt(10*sqrt(5)+50)/20,  -sqrt(50-10*sqrt(5))/20 },
        {            (1+sqrt(5))/4,           -1/(1+sqrt(5)),  -sqrt(10*sqrt(5)+50)/20,   sqrt(50-10*sqrt(5))/20 },
        {            (1+sqrt(5))/4,                     -0.5,   sqrt(25-10*sqrt(5))/10,   sqrt(50-10*sqrt(5))/20 },
        {            (1+sqrt(5))/4,                     -0.5,  -sqrt(25-10*sqrt(5))/10,  -sqrt(50-10*sqrt(5))/20 },
        {                      0.5,            (1+sqrt(5))/4,   sqrt(50-10*sqrt(5))/20,  -sqrt(25-10*sqrt(5))/10 },
        {                      0.5,            (1+sqrt(5))/4,  -sqrt(50-10*sqrt(5))/20,   sqrt(25-10*sqrt(5))/10 },
        {                      0.5,                      0.5,   sqrt((5+2*sqrt(5))/20),   sqrt(25-10*sqrt(5))/10 },
        {                      0.5,                      0.5,   sqrt(25-10*sqrt(5))/10,  -sqrt((5+2*sqrt(5))/20) },
        {                      0.5,                      0.5,  -sqrt(25-10*sqrt(5))/10,   sqrt((5+2*sqrt(5))/20) },
        {                      0.5,                      0.5,  -sqrt((5+2*sqrt(5))/20),  -sqrt(25-10*sqrt(5))/10 },
        {                      0.5,            1/(1+sqrt(5)),   sqrt(10*sqrt(5)+50)/20,   sqrt((5+2*sqrt(5))/20) },
        {                      0.5,            1/(1+sqrt(5)),  -sqrt(10*sqrt(5)+50)/20,  -sqrt((5+2*sqrt(5))/20) },
        {                      0.5,                        0,     sqrt((5+sqrt(5))/10),  -sqrt(25-10*sqrt(5))/10 },
        {                      0.5,                        0,   sqrt(50-10*sqrt(5))/10,  -sqrt((5+2*sqrt(5))/20) },
        {                      0.5,                        0,  -sqrt(50-10*sqrt(5))/10,   sqrt((5+2*sqrt(5))/20) },
        {                      0.5,                        0,    -sqrt((5+sqrt(5))/10),   sqrt(25-10*sqrt(5))/10 },
        {                      0.5,           -1/(1+sqrt(5)),   sqrt(10*sqrt(5)+50)/20,   sqrt((5+2*sqrt(5))/20) },
        {                      0.5,           -1/(1+sqrt(5)),  -sqrt(10*sqrt(5)+50)/20,  -sqrt((5+2*sqrt(5))/20) },
        {                      0.5,                     -0.5,   sqrt((5+2*sqrt(5))/20),   sqrt(25-10*sqrt(5))/10 },
        {                      0.5,                     -0.5,   sqrt(25-10*sqrt(5))/10,  -sqrt((5+2*sqrt(5))/20) },
        {                      0.5,                     -0.5,  -sqrt(25-10*sqrt(5))/10,   sqrt((5+2*sqrt(5))/20) },
        {                      0.5,                     -0.5,  -sqrt((5+2*sqrt(5))/20),  -sqrt(25-10*sqrt(5))/10 },
        {                      0.5,           -(1+sqrt(5))/4,   sqrt(50-10*sqrt(5))/20,  -sqrt(25-10*sqrt(5))/10 },
        {                      0.5,           -(1+sqrt(5))/4,  -sqrt(50-10*sqrt(5))/20,   sqrt(25-10*sqrt(5))/10 },
        {            1/(1+sqrt(5)),            (1+sqrt(5))/4,   sqrt(50-10*sqrt(5))/20,   sqrt(10*sqrt(5)+50)/20 },
        {            1/(1+sqrt(5)),            (1+sqrt(5))/4,  -sqrt(50-10*sqrt(5))/20,  -sqrt(10*sqrt(5)+50)/20 },
        {            1/(1+sqrt(5)),                      0.5,   sqrt((5+2*sqrt(5))/20),  -sqrt(10*sqrt(5)+50)/20 },
        {            1/(1+sqrt(5)),                      0.5,  -sqrt((5+2*sqrt(5))/20),   sqrt(10*sqrt(5)+50)/20 },
        {            1/(1+sqrt(5)),                        0,     sqrt((5+sqrt(5))/10),   sqrt(10*sqrt(5)+50)/20 },
        {            1/(1+sqrt(5)),                        0,                        0,  sqrt(1-1/(2*sqrt(5)+6)) },
        {            1/(1+sqrt(5)),                        0,                        0, -sqrt(1-1/(2*sqrt(5)+6)) },
        {            1/(1+sqrt(5)),                        0,    -sqrt((5+sqrt(5))/10),  -sqrt(10*sqrt(5)+50)/20 },
        {            1/(1+sqrt(5)),                     -0.5,   sqrt((5+2*sqrt(5))/20),  -sqrt(10*sqrt(5)+50)/20 },
        {            1/(1+sqrt(5)),                     -0.5,  -sqrt((5+2*sqrt(5))/20),   sqrt(10*sqrt(5)+50)/20 },
        {            1/(1+sqrt(5)),           -(1+sqrt(5))/4,   sqrt(50-10*sqrt(5))/20,   sqrt(10*sqrt(5)+50)/20 },
        {            1/(1+sqrt(5)),           -(1+sqrt(5))/4,  -sqrt(50-10*sqrt(5))/20,  -sqrt(10*sqrt(5)+50)/20 },
        {                        0,                        1,                        0,                        0 },
        {                        0,            (1+sqrt(5))/4,     sqrt(5./8-sqrt(5)/8),                        0 },
        {                        0,            (1+sqrt(5))/4,   sqrt(50-10*sqrt(5))/20,  -sqrt(50-10*sqrt(5))/10 },
        {                        0,            (1+sqrt(5))/4,  -sqrt(50-10*sqrt(5))/20,   sqrt(50-10*sqrt(5))/10 },
        {                        0,            (1+sqrt(5))/4,    -sqrt(5./8-sqrt(5)/8),                        0 },
        {                        0,                      0.5,   sqrt((5+2*sqrt(5))/20),   sqrt(50-10*sqrt(5))/10 },
        {                        0,                      0.5,   sqrt(25-10*sqrt(5))/10,     sqrt((5+sqrt(5))/10) },
        {                        0,                      0.5,  -sqrt(25-10*sqrt(5))/10,    -sqrt((5+sqrt(5))/10) },
        {                        0,                      0.5,  -sqrt((5+2*sqrt(5))/20),  -sqrt(50-10*sqrt(5))/10 },
        {                        0,            1/(1+sqrt(5)),  sqrt(1-1/(2*sqrt(5)+6)),                        0 },
        {                        0,            1/(1+sqrt(5)),   sqrt(10*sqrt(5)+50)/20,    -sqrt((5+sqrt(5))/10) },
        {                        0,            1/(1+sqrt(5)),  -sqrt(10*sqrt(5)+50)/20,     sqrt((5+sqrt(5))/10) },
        {                        0,            1/(1+sqrt(5)), -sqrt(1-1/(2*sqrt(5)+6)),                        0 },
        {                        0,                        0,     sqrt((5+sqrt(5))/10),  -sqrt(50-10*sqrt(5))/10 },
        {                        0,                        0,   sqrt(50-10*sqrt(5))/10,     sqrt((5+sqrt(5))/10) },
};


int rotate_quaternion_into_cubic_fundamental_zone(double* q);
int rotate_quaternion_into_diamond_cubic_fundamental_zone(double* q);
int rotate_quaternion_into_icosahedral_fundamental_zone(double* q);
int rotate_quaternion_into_hcp_fundamental_zone(double* q);
int rotate_quaternion_into_hcp_conventional_fundamental_zone(double* q);
int rotate_quaternion_into_diamond_hexagonal_fundamental_zone(double* q);

void quat_rot(double* r, double* a, double* b);
void normalize_quaternion(double* q);
void quaternion_to_rotation_matrix(double* q, double* U);
void rotation_matrix_to_quaternion(double* u, double* q);
double quat_dot(double* a, double* b);
double quat_misorientation(double* q1, double* q2);

int map_quaternion_cubic(double* q, int i);
int map_quaternion_diamond_cubic(double* q, int i);
int map_quaternion_icosahedral(double* q, int i);
int map_quaternion_hcp(double* q, int i);
int map_quaternion_hcp_conventional(double* q, int i);
int map_quaternion_diamond_hexagonal(double* q, int i);

double quat_disorientation_cubic(double* q0, double* q1);
double quat_disorientation_hcp_conventional(double* q0, double* q1);

double quat_disorientation_hexagonal_to_cubic(double* qfcc, double* qhcp);
double quat_disorientation_cubic_to_hexagonal(double* qhcp, double* qfcc);
}

#endif

