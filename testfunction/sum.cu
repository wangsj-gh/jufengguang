__global__ void parallel_reduce_kernel(float *d_out, float* d_in){
    int myID = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    //divide threads into two parts according to threadID, and add the right part to the left one, lead to reducing half elements, called an iteration; iterate until left only one element
    for(unsigned int s = blockDim.x / 2 ; s>0; s>>=1){
        if(tid<s){
            d_in[myID] += d_in[myID + s];
        }
        __syncthreads(); //ensure all adds at one iteration are done
    }
    if (tid == 0){
        d_out[blockIdx.x] = d_in[myId];
    }
}

__global__ void parallel_shared_reduce_kernel(float *d_out, float* d_in){
    int myID = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    extern __shared__ float sdata[];
    sdata[tid] = d_in[myID];
    __syncthreads();

    //divide threads into two parts according to threadID, and add the right part to the left one, lead to reducing half elements, called an iteration; iterate until left only one element
    for(unsigned int s = blockDim.x / 2 ; s>0; s>>=1){
        if(tid<s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); //ensure all adds at one iteration are done
    }
    if (tid == 0){
        d_out[blockIdx.x] = sdata[myId];
    }
}