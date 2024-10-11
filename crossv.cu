#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda.h>

// put the bandwidths in constant memory.
// note that max number cannot exceed 8*1024,
// given the 64kb limit for constant memory
__constant__ float d_bw[1000];

// GPU-specific implementation of quick-sort
// to be implemented by a single thread.
// Note that both A and B are sorted in
// increasing order of A.
// CUDA had problems with my recursive quick-sort,
// so I took an interative version from the
// following website.

// iterative version of quicksort taken
// from http://alienryderflex.com/quicksort/
// and then adapted to sort two arrays
// and to malloc the beg and end arrays.
__device__ void d_quicksort(float *A, float *B, int N) {

  #define  MAX_DEPTH  1000

  float pivot_A, pivot_B;
  int beg[MAX_DEPTH], end[MAX_DEPTH], i=0, L, R, swap;

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
__global__ void epanXYSum(float *yVec, float *xVec, float *xjxVec, float *yCopy, float *sumx, float *sumy, float *crossV, int N, int B){

	int j = blockDim.x*blockIdx.x + threadIdx.x;

	if(j<N){

		int i,b;
		for(i=0;i<N;i++){
			xjxVec[N*j + i] = fabsf(xVec[j]-xVec[i]);
			yCopy[N*j + i] = yVec[i];
		}

		// sort the data so that the closest to xj
		// appear first. Then, if an observation i
		// is within some bandwidth b, then so is
		// every observation less than i. This
		// function also ensures that yCopy is sorted
		// in order of xjxVec.
		d_quicksort(&xjxVec[N*j], &yCopy[N*j], N);

		i=0;
		sumx[B*j] = 0.0;
		sumy[B*j] = 0.0;
		while(xjxVec[N*j+i]<=d_bw[0] && i<N){
			sumx[B*j]-=xjxVec[N*j+i]*xjxVec[N*j+i];
			sumy[B*j]-=yCopy[N*j+i]*xjxVec[N*j+i]*xjxVec[N*j+i];		
			i++;
		}

		for(b=1;b<B;b++){
			sumx[B*j+b]=sumx[B*j+b-1];
			sumy[B*j+b]=sumy[B*j+b-1];
			while(xjxVec[N*j+i]<=d_bw[b] && i<N){
				sumx[B*j+b]-=xjxVec[N*j+i]*xjxVec[N*j+i];
				sumy[B*j+b]-=yCopy[N*j+i]*xjxVec[N*j+i]*xjxVec[N*j+i];
				i++;
			}
		}

		for(b=0;b<B;b++){
			sumx[B*j+b]/=(d_bw[b]*d_bw[b]);
			sumx[B*j+b]=-0.75*sumx[B*j+b]; // note that the 0.75*1 is excluded due to leave-one-out.

			sumy[B*j+b]/=(d_bw[b]*d_bw[b]);
			sumy[B*j+b]=-0.75*sumy[B*j+b]; // similarly, we do not add 0.75*yVec[i] here.
		
			// switch the ordering of indices to facilitate the reduction.
			if(sumx[B*j+b]!=0.0){
				crossV[N*b+j]=(yVec[j] -sumy[B*j+b]/sumx[B*j+b])*(yVec[j] -sumy[B*j+b]/sumx[B*j+b])/N;
			} else {
				crossV[N*b+j]=FLT_MAX;
			}
		}
	}
}

// This function is taken from:
// http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <unsigned int blockSize>
__device__ void warpReduce_sum(volatile float *sdata, unsigned int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

// This function is a modified version of the reduce6 from:
// http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <unsigned int blockSize>
__global__ void reduce_sum(float *g_idata, float *g_odata, int n) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = tid;
	sdata[tid] = 0.0;

	while (i < n) { sdata[tid] += g_idata[i]; i += blockSize; }
	__syncthreads();
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) warpReduce_sum<blockSize>(sdata, tid);
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// This function is modified from:
// http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <unsigned int blockSize>
__device__ void warpReduce_min(volatile float *sdata, unsigned int tid) {
	if (blockSize >= 64) {
		if(sdata[tid+32]<sdata[tid]){
			sdata[tid]=sdata[tid+32];
			sdata[tid+blockSize]=sdata[tid+32+blockSize];
		}
	};
	if (blockSize >= 32) {
		if(sdata[tid+16]<sdata[tid]){
			sdata[tid]=sdata[tid+16];
			sdata[tid+blockSize]=sdata[tid+16+blockSize];
		}
	}
	if (blockSize >= 16) {
		if(sdata[tid+8]<sdata[tid]){
			sdata[tid]=sdata[tid+8];
			sdata[tid+blockSize]=sdata[tid+8+blockSize];			
		}
	}
	if (blockSize >= 8) {
		if(sdata[tid+4]<sdata[tid]){
			sdata[tid]=sdata[tid+4];
			sdata[tid+blockSize]=sdata[tid+4+blockSize];			
		}	
	}
	if (blockSize >= 4) {
		if(sdata[tid+2]<sdata[tid]){
			sdata[tid]=sdata[tid+2];
			sdata[tid+blockSize]=sdata[tid+2+blockSize];			
		}
	}
	if (blockSize >= 2){
		if(sdata[tid+1]<sdata[tid]){
			sdata[tid]=sdata[tid+1];
			sdata[tid+blockSize]=sdata[tid+1+blockSize];			
		}
	}
}

// This function is a modified version of the reduce6 from:
// http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <unsigned int blockSize>
__global__ void reduce_min(float *g_idata, float *g_odata, int n) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = tid;
	sdata[tid] = FLT_MAX;

	while (i < n) {
		if(g_idata[i]<sdata[tid]){
			sdata[tid] = g_idata[i];
			sdata[tid+blockSize]=d_bw[i];
		}
		i+=blockSize;
		__syncthreads();
	}

	if (blockSize >= 1024) {
		if (tid < 512) {
			if(sdata[tid+512]<sdata[tid]){
				sdata[tid]=sdata[tid+512];
				sdata[tid+blockSize]=sdata[tid+512+blockSize];
			}
		}
		__syncthreads();
	} 
	if (blockSize >= 512) {
		if (tid < 256) {
			if(sdata[tid+256]<sdata[tid]){
				sdata[tid]=sdata[tid+256];
				sdata[tid+blockSize]=sdata[tid+256+blockSize];
			}
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			if(sdata[tid+128]<sdata[tid]){
				sdata[tid]=sdata[tid+128];
				sdata[tid+blockSize]=sdata[tid+128+blockSize];
			}
			
		}
		__syncthreads();
	} 
	if (blockSize >= 128) {
		if (tid < 64) {
			if(sdata[tid+64]<sdata[tid]){
				sdata[tid]=sdata[tid+64];
				sdata[tid+blockSize]=sdata[tid+64+blockSize];
			}
		}
		__syncthreads();
	}
	if (tid < 32) warpReduce_min<blockSize>(sdata, tid);

	// only return the optimal bandwidth
	if (tid == 0) *g_odata = sdata[blockSize];
}

int main(int argc, char *argv[]) {

	srand(4251978);

	// constants used in calculations.
    // we'll allow user inputs but
    // have default values.
	int N=1024; // number of observations in vector
	int B = 50; // number of bandwidths in grid.
	const int T=512;

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

	float *d_X, *d_Y, *d_xjxVec, *d_yCopy, *d_sumx, *d_sumy, *d_crossV_elem;
	cudaMalloc((void **)&d_X, N*sizeof(float));
	cudaMalloc((void **)&d_Y, N*sizeof(float));
	cudaMalloc((void **)&d_xjxVec, N*N*sizeof(float));
	cudaMalloc((void **)&d_yCopy, N*N*sizeof(float));
	cudaMalloc((void **)&d_sumx, B*N*sizeof(float));
	cudaMalloc((void **)&d_sumy, B*N*sizeof(float));
	cudaMalloc((void **)&d_crossV_elem, B*N*sizeof(float));

	cudaMemcpy(d_X, X, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, Y, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_bw, bw, B*sizeof(float));
	free(X);
	free(Y);

	cudaError_t code = cudaGetLastError();
	if(code!=cudaSuccess){
        printf("Cuda error -- %s\n", cudaGetErrorString(code));
    } 

	int numBlocks = N/T +1;
	dim3 threadsPerBlock(T);

	epanXYSum<<<numBlocks,threadsPerBlock>>>(d_Y,d_X,d_xjxVec,d_yCopy,d_sumx,d_sumy,d_crossV_elem,N,B);

	cudaFree(d_sumx);
	cudaFree(d_sumy);
	cudaFree(d_Y);
	cudaFree(d_X);

	// perform B separate reduction sums to obtain the
	// cross-validation statistic for each bandwidth.
	float *d_crossV_sum;
	cudaMalloc((void **)&d_crossV_sum, B*sizeof(float));
	for(b=0;b<B;b++){
		reduce_sum<T><<< 1,threadsPerBlock,T*sizeof(float) >>>(&d_crossV_elem[N*b], &d_crossV_sum[b], N);
	}
	cudaFree(d_crossV_elem);

	float *d_bw_optimal;
	cudaMalloc((void **)&d_bw_optimal, sizeof(float));
	reduce_min<T><<< 1,threadsPerBlock,2*T*sizeof(float) >>>(d_crossV_sum, d_bw_optimal, B);
	cudaFree(d_crossV_sum);

	float *bw_optimal = (float*)malloc(sizeof(float));
	cudaMemcpy(bw_optimal, d_bw_optimal, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_bw_optimal);

	printf("Optimal bandwidth is: %f\n",*bw_optimal);
	printf("Min bandwidth is: %f\n",bw[0]);
	printf("Max bandwidth is: %f\n",bw[B-1]);
	free(bw);
	free(bw_optimal);

    return(0);
}