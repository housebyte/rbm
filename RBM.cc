#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <math.h>

using namespace std;

struct layer{

double **wgt;  //wgt matrix for layer
double *act;  //activation of each neuron above
double *preact;
double *bias;
};

class rbm{
public:
layer *net;	//layers for net
int size,nlayers;
double *h_array,*v_array;
double K;
double *input;

void init(int nlayers_, int size_){

K =0.95;		//Learning rate
nlayers = nlayers_;
size = size_;

net = new layer[nlayers]; // 2 layers

h_array = new double[size];
v_array = new double[size];
input = new double[size];

for(int i=0;i<nlayers;i++){
net[i].act = new double[size];
net[i].preact = new double[size];
net[i].bias = new double[size];
net[i].wgt = new double*[size];
for(int j=0;j<size;j++){
net[i].wgt[j] = new double[size];
for(int k=0;k<size;k++){
net[i].wgt[j][k] = (double)(rand()/(RAND_MAX + 1.0));
}
net[i].act[j] = 0;//rand()%2;
net[i].preact[j] = 0;//1/((double)(rand()%10+1));			
net[i].bias[j] = 0;//1/((double)(rand()%10+1));
			}
			
			

			   }

				}
				
				
void propup(double *input_){	//Propup to get h from v

for(int i=0;i<size;i++){	//Clamp inputs
net[0].act[i] = input_[i];
			}


for(int i=1;i<nlayers;i++){

for(int j=0;j<size;j++){
double sum=0;				//Which neuron top
for(int k=0;k<size;k++){		//Which neuron bottom
sum += net[i-1].wgt[j][k]*net[i-1].act[k];
}
net[i].act[j] = (sigmoid(sum+net[i].bias[j])>((double)(rand()/(RAND_MAX + 1.0)))) ? 1 : 0;
net[i].preact[j] = sigmoid(sum+net[i].bias[j]);
//net[i].preact[j] = sum;
}


}
h_array = net[nlayers-1].act;			
			  }

void propdown(double *input_){	//Propdown to get v from h

for(int i=0;i<size;i++){	//Clamp hidden
net[nlayers-1].act[i] = input_[i];
			}

for(int i=nlayers-1;i>0;i--){

for(int j=0;j<size;j++){		//Which neuron bottom
double sum=0;
for(int k=0;k<size;k++){		//Which neuron top
sum += net[i-1].wgt[k][j]*net[i].act[k];
}
net[i-1].act[j] = (sigmoid(sum+net[i-1].bias[j])>((double)(rand()/(RAND_MAX + 1.0)))) ? 1 : 0;
net[i-1].preact[j] = sigmoid(sum+net[i-1].bias[j]);
//net[i-1].preact[j] = sum;
}


}
v_array = net[0].act;
			
			  }
void gibbs_hvh(double *h_array_){

propdown(h_array_);
v_array = net[0].act;
propup(v_array);
h_array = net[nlayers-1].act;
}

void gibbs_vhv(double *v_array_){

propup(v_array_);
h_array = net[nlayers-1].act;
propdown(h_array);
v_array = net[0].act;
}

void test(){

for(int i=0;i<size;i++)
cout<<input[i]<<",";

cout<<"--->"; 

propup(input);
propdown(h_array);


int count=200;
while(count<100){

gibbs_vhv(v_array);

count++;
}


for(int i=0;i<size;i++){
cout<<net[0].act[i]<<",";  
			}
cout<<"Varray:";
for(int i=0;i<size;i++){
cout<<net[0].preact[i]<<",";
}
cout<<"\n";

}

void chain(int length){

double **h_v,**h2_v2;
h_v = new double*[size];
h2_v2 = new double*[size];

for(int i=0;i<size;i++){
h_v[i] = new double[size];
h2_v2[i] = new double[size];
}

double *h_array_;
h_array_ = new double[size];



gibbs_vhv(input);   //gives h_v

for(int i=0;i<size;i++)
h_array_[i] = h_array[i];

for(int i=0;i<size;i++){
for(int j=0;j<size;j++){
//if(net[0].act[i]==1&&net[nlayers-1].act[j]==1)
h_v[i][j] = h_array_[j] * input[i];	/* doesnt work for input needs v_array*/
}}

int count=0;

while(count<length){

if(count==0){
gibbs_hvh(h_array_); //gives v' = v_array & gives h' = h_array 
}else{
gibbs_hvh(h_array);
}
count++;

}

for(int i=0;i<size;i++){
for(int j=0;j<size;j++){
//if(net[0].act[i]==1&&net[nlayers-1].act[j]==1)
h2_v2[i][j] = h_array[j] * v_array[i];
}}

//change weights

for(int i=0;i<size;i++){
for(int j=0;j<size;j++){

//if(net[0].act[i]==1&&net[nlayers-1].act[j]==1)
net[0].wgt[i][j] += K*(h_v[i][j] - h2_v2[i][j]);

			}
net[0].bias[i] += K*(input[i] - v_array[i]);			
net[nlayers-1].bias[i] += K*(h_array_[i] - h_array[i]);			
			}


					}

void sampledata(){

double xord[4][4];

xord[0][0]=0; xord[0][1]=0; xord[0][2]=0;	//-0.5;
xord[1][0]=0; xord[1][1]=1; xord[1][2]=1;	//0.5;
xord[2][0]=1; xord[2][1]=0; xord[2][2]=1;	//0.5;
xord[3][0]=1; xord[3][1]=1; xord[3][2]=0;	//-0.5;

int select = rand()%4;

for(int i=0;i<3;i++)
input[i] = xord[select][i];

}


double sigmoid(double sum){

return 1/(1+exp(-sum));

}

};

int main(){

rbm *mynet;

mynet = new rbm;

mynet->init(2,3);

mynet->K = 0.95;

int count=0;

while(count<1000){

mynet->sampledata();

mynet->chain(1);

count++;
}
count=0;


while(count<20){

mynet->sampledata();

mynet->test();

count++;
}


}

