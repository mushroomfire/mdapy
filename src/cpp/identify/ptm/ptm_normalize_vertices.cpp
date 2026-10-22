/*Copyright (c) 2022 PM Larsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <cmath>

namespace ptm {

double vector_norm(double* v) {
    double x = v[0];
    double y = v[1];
    double z = v[2];
    return sqrt(x*x + y*y + z*z);
}

void subtract_barycentre(int num, double (*points)[3], double (*normalized)[3])
{
    //calculate barycentre
    double sum[3] = {0, 0, 0};
    for (int i=0;i<num;i++)
    {
        sum[0] += points[i][0];
        sum[1] += points[i][1];
        sum[2] += points[i][2];
    }

    sum[0] /= num;
    sum[1] /= num;
    sum[2] /= num;

    //subtract barycentre
    for (int i=0;i<num;i++)
    {
        normalized[i][0] = points[i][0] - sum[0];
        normalized[i][1] = points[i][1] - sum[1];
        normalized[i][2] = points[i][2] - sum[2];
    }
}

double normalize_vertices(int num, double (*points)[3], double (*normalized)[3])
{
    subtract_barycentre(num, points, normalized);

    //calculate mean length
    double scale = 0.0;
    for (int i=1;i<num;i++) {
        scale += vector_norm(normalized[i]);
    }
    scale /= num;

    //scale vertices such that mean length is 1
    for (int i=0;i<num;i++)
    {
        normalized[i][0] /= scale;
        normalized[i][1] /= scale;
        normalized[i][2] /= scale;
    }

    return scale;
}

}

