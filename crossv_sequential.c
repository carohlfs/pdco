#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

// GPU-specific version of quick-sort
// to be implemented by a single thread.
// note that B is to be sorted in increasing
// order of A.
void quicksort(float *A, float *B, int N) {

  int *beg = (int*)malloc(N*sizeof(int));
  int *end = (int*)malloc(N*sizeof(int));

  float pivot_A, pivot_B;
  int i=0, L, R, swap;

  beg[0]=0; end[0]=N;
  while (i>=0) {
    L=beg[i]; R=end[i]-1;

    if (L<R) {
      pivot_A=A[L];
      pivot_B=B[L];
      
      while (L<R) {
        while (A[R]>=pivot_A && L<R) R--;
        if (L<R) {
          A[L]=A[R];
          B[L]=B[R];
          L++;
        }
        while (A[L]<=pivot_A && L<R) L++;
        if (L<R) {
          A[R]=A[L];
          B[R]=B[L];
          R--;
        }
      }
      
      A[L]=pivot_A;
      B[L]=pivot_B;

      beg[i+1]=L+1;
      end[i+1]=end[i];
      end[i++]=L;
      
      if (end[i]-beg[i]>end[i-1]-beg[i-1]) {
        swap=beg[i];
        beg[i]=beg[i-1];
        beg[i-1]=swap;
        swap=end[i];
        end[i]=end[i-1];
        end[i-1]=swap;
      }
    } else {
      i--;
    }
  }
}

// Here, we compute the Epanechnikov sum for a single
// xj value for an array of different bandwidths. The
// operations are performed simultaneously for multiple
// bandwidths to minimize the extent to which operations
// are duplicated.
void epanXYSum(int j, float *yVec, float *xVec, float *bw, float *sumx, float *sumy, float *crossV, int N, int B){

	float *xjxVec = (float*)malloc(N*sizeof(float));
	float *yCopy = (float*)malloc(N*sizeof(float));

	int i,b;

	for(i=0;i<N;i++){
		xjxVec[i] = fabsf(xVec[j]-xVec[i]);
		yCopy[i] = yVec[i];
	}

	// sort the data so that the closest to xj
	// appear first. Then, if an observation i
	// is within some bandwidth b, then so is
	// every observation less than i. This
	// function also ensures that yCopy is sorted
	// in order of xjxVec.
	quicksort(xjxVec, yCopy, N);

	i=0;
	sumx[B*j] = 0.0;
	sumy[B*j] = 0.0;
	while(xjxVec[i]<=bw[0] && i<N){
		sumx[B*j]-=xjxVec[i]*xjxVec[i];
		sumy[B*j]-=yCopy[i]*xjxVec[i]*xjxVec[i];		
		i++;
	}

	for(b=1;b<B;b++){
		sumx[B*j+b]=sumx[B*j+b-1];
		sumy[B*j+b]=sumy[B*j+b-1];
		while(xjxVec[i]<=bw[b] && i<N){
			sumx[B*j+b]-=xjxVec[i]*xjxVec[i];
			sumy[B*j+b]-=yCopy[i]*xjxVec[i]*xjxVec[i];
			i++;
		}
	}

	free(xjxVec);
	free(yCopy);

	for(b=0;b<B;b++){
		sumx[B*j+b]/=(bw[b]*bw[b]);
		sumx[B*j+b]=-0.75*sumx[B*j+b]; // note that the 0.75*1 is excluded due to leave-one-out.

		sumy[B*j+b]/=(bw[b]*bw[b]);
		sumy[B*j+b]=-0.75*sumy[B*j+b]; // similarly, we do not add 0.75*yVec[i] here.
	
		if(sumx[B*j+b]!=0.0){
			crossV[B*j+b]=(yVec[j] -sumy[B*j+b]/sumx[B*j+b])*(yVec[j] -sumy[B*j+b]/sumx[B*j+b])/N;
		} else {
			crossV[B*j+b]=FLT_MAX;
		}
	}

}

int main(int argc, char *argv[]) {

	srand(4251978);

	// constants used in calculations.
    // we'll allow user inputs but
    // have default values.
	int N=1024; // number of observations in vector
	int B = 50; // number of bandwidths in grid.

	if(argc>1){
		N = atoi(argv[1]);
	}

	int i;
	float *X = (float*)malloc(N*sizeof(float));
	float *Y = (float*)malloc(N*sizeof(float));

	for(i=0;i<N;i++){
		X[i] = (float)rand()/(float)RAND_MAX;
		Y[i] = 0.5*X[i] + 10*X[i]*X[i] + 0.5*(float)rand()/(float)RAND_MAX;
	}

	float minx=X[0], maxx=X[0];
	for(i=1;i<N;i++){
		if(X[i]<minx){
			minx = X[i];
		} else if(X[i]>maxx){
			maxx = X[i];
		}
	}

	float range = maxx -minx;

	if(argc>2){
		B = atoi(argv[2]);
	}
	if(argc>3){
		range = atof(argv[3]);
	}

	float increment = range/B;
	float start = increment;

	if(argc>4){
		start = atof(argv[4]);
		increment = range/(B-1);
	}

	int b;
	float *bw = (float*)malloc(B*sizeof(float));
	
	bw[0] = start;
	for(b=1;b<B;b++){
		bw[b] = bw[b-1] + increment;
	}

	float *sumx = (float*)malloc(B*N*sizeof(float));
	float *sumy = (float*)malloc(B*N*sizeof(float));
	float *crossV_elem = (float*)malloc(B*N*sizeof(float));
	float *crossV_sum = (float*)calloc(B,sizeof(float));

	for(i=0;i<N;i++){
		epanXYSum(i,Y,X,bw,sumx,sumy,crossV_elem,N,B);
		for(b=0;b<B;b++){
			crossV_sum[b]+=crossV_elem[B*i+b];
		}
	}

	free(sumx);
	free(sumy);
	free(crossV_elem);
	free(Y);
	free(X);

	float crossV_optimal = crossV_sum[0];
	float bw_optimal = bw[0];

	for(b=1;b<B;b++){
		if(crossV_sum[b]<crossV_optimal){
			crossV_optimal=crossV_sum[b];
			bw_optimal = bw[b];
		}
	}

	printf("Optimal bandwidth is: %f\n",bw_optimal);
	printf("Min bandwidth is: %f\n",bw[0]);
	printf("Max bandwidth is: %f\n",bw[B-1]);
	free(crossV_sum);
	free(bw);

    return(0);
}