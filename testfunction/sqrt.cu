#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
using namespace std;

cudaError_t err;

__global__ void add(float *x, float *y)
{
    // 获取全局索引
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    y[i] = x[i];
    // y[i] = 6 * sqrt(x[i]) + 2;

    // y[index] = x[index] / 0.0;
    // if (x[i] == 0)
    // {
    //     y[i] = NAN;
    // }

    // y[index] = min(min(x[index], rr), bb);
}
int main()
{
    int N = 1 << 4;

    int nBytes = N * sizeof(float);
    // 申请host内存
    float *x, *y;
    x = (float *)malloc(nBytes);
    y = (float *)malloc(nBytes);

    // 初始化数据
    for (int i = 0; i < N; ++i)
    {
        x[i] = 4.0;
    }

    // 申请device内存
    float *d_x, *d_y;
    cudaMalloc((void **)&d_x, nBytes);
    cudaMalloc((void **)&d_y, nBytes);

    // 将host数据拷贝到device
    cudaMemcpy((void *)d_x, (void *)x, nBytes, cudaMemcpyHostToDevice);

    for (int i = 0; i < N; ++i)
    {
        std::cout << x[i] << std::endl;
    }

    // 定义kernel的执行配置
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    // 执行kernel
    add<<<gridSize, blockSize>>>(d_x, d_y);

    // 将device得到的结果拷贝到host
    err = cudaMemcpy((void *)y, (void *)d_y, nBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cout << " " << cudaGetErrorString(err) << std::endl;
    }
    // 检查执行结果
    // float maxError = 0.0;
    // for (int i = 0; i < N; i++)
    //     maxError = fmax(maxError, fabs(z[i] - 30.0));
    // std::cout << "最大误差: " << maxError << std::endl;
    for (int i = 0; i < N; i++)
    {
        std::cout << "y: " << y[i] << std::endl;
    }

    // 释放device内存
    cudaFree(d_x);
    cudaFree(d_y);
    // 释放host内存
    free(x);
    free(y);

    return 0;
}