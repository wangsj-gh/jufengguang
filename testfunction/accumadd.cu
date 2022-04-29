#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
using namespace std;

__global__ void accumdd(const float *GPP, float *sum)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    sum[i] = sum[i] + GPP[i];
};

__global__ void GetMean(const float *TotalGpp, float *MeanGpp)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    MeanGpp[i] = TotalGpp[i] / 8;
}

int main()
{
    int N = 1 << 4;

    int nBytes = N * sizeof(float);
    // 申请host内存
    float *x, *sum, *mean;
    x = (float *)malloc(nBytes);
    sum = (float *)malloc(nBytes);
    mean = (float *)malloc(nBytes);
    // 初始化数据
    for (int i = 0; i < N; ++i)
    {
        x[i] = 4.0;
    }

    for (int i = 0; i < N; ++i)
    {
        sum[i] = 0.0;
    }

    // 申请device内存
    float *d_x, *d_sum, *d_mean;
    cudaMalloc((void **)&d_x, nBytes);
    cudaMalloc((void **)&d_sum, nBytes);
    cudaMalloc((void **)&d_mean, nBytes);
    // 将host数据拷贝到device
    cudaMemcpy((void *)d_x, (void *)x, nBytes, cudaMemcpyHostToDevice);
    // cudaMemcpy((void *)d_sum, (void *)sum, nBytes, cudaMemcpyHostToDevice);

    // 定义kernel的执行配置
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    // 执行kernel
    for (int j = 0; j < 8 * 8; j++)
    {

        memset(sum, 0, nBytes);
        // memset(h_MeanGpp, 0, size);
        cudaMemcpy((void *)d_sum, (void *)sum, nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)d_mean, (void *)sum, nBytes, cudaMemcpyHostToDevice);
        for (int i = j; i < j + 8; i++)
        {
            // std::cout << "i:" << i << std::endl;
            accumdd<<<gridSize, blockSize>>>(d_x, d_sum);
        }
        GetMean<<<gridSize, blockSize>>>(d_sum, d_mean);

        // 将device得到的结果拷贝到host
        cudaMemcpy((void *)sum, (void *)d_sum, nBytes, cudaMemcpyDeviceToHost);
        cudaMemcpy((void *)mean, (void *)d_mean, nBytes, cudaMemcpyDeviceToHost);

        std::cout << "sum: " << sum[0] << std::endl;
        std::cout << "mean: " << mean[0] << std::endl;
    }

    // 释放device内存
    cudaFree(d_x);
    cudaFree(d_sum);
    cudaFree(d_mean);
    // 释放host内存
    free(x);
    free(sum);
    free(mean);

    return 0;
}