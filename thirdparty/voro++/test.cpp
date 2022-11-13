#include "voro++.hh"
#include <iostream>
using namespace voro;
using namespace std;
const int particles=20;

double rnd() {return double(rand())/RAND_MAX;}

int main(){

         int i;
         int id,nx,ny,nz;
         double x,y,z,r,rx,ry,rz;
         vector<int> neigh,f_vert;
         vector<double> face;
         voronoicell_neighbor c;
         // Create a container with the geometry given above, and make it
         // non-periodic in each of the three coordinates. Allocate space for
         // eight particles within each computational block
         pre_container pcon(0,1,0,1,0,1,true,true,true,8);
         container con(0,1,0,1,0,1,5,5,5,true,true,true,8);
 
         // Randomly add particles into the container
         for(i=0;i<particles;i++) {
                 x=rnd();
                 y=rnd();
                 z=rnd();
                 con.put(i+100,x,y,z);
         }

         //con.compute_all_cells(); 
         c_loop_all cl(con);
         if (cl.start()) do if(con.compute_cell(c,cl)) {
                cl.pos(x,y,z);id=cl.pid();
                int a=1;
                c.face_areas(face);
                c.neighbors(neigh);
                cout<<face[0]<<endl;
                cout<<neigh[0]<<endl;
         } while (cl.inc());

}