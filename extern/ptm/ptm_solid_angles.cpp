/*Copyright (c) 2022 PM Larsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

// Calculation of solid angles uses method described in:
// A. Van Oosterom and J. Strackee
// "The Solid Angle of a Plane Triangle"
// IEEE Transactions on Biomedical Engineering, BME-30, 2, 1983, 125--126
// https://doi.org/10.1109/TBME.1983.325207


#include <cmath>


namespace ptm {

static double dot_product(double* a, double* b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void cross_product(double* a, double* b, double* c)
{
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

// assumes R1, R2, R3 are unit vectors
double calculate_solid_angle(double* R1, double* R2, double* R3)
{
    double R2R3[3];
    cross_product(R2, R3, R2R3);
    double numerator = dot_product(R1, R2R3);

    double r1r2 = dot_product(R1, R2);
    double r2r3 = dot_product(R2, R3);
    double r3r1 = dot_product(R3, R1);

    double denominator = 1 + r1r2 + r3r1 + r2r3;
    return fabs(2 * atan2(numerator, denominator));
}

}

