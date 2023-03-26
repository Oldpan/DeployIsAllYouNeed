// 统计当前显卡的算力

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#define CHECK_CUDA(x, str) \
  if((x) != cudaSuccess) \
  { \
    fprintf(stderr, str); \
    exit(EXIT_FAILURE); \
  }

int cc2cores(int major, int minor)
{
  typedef struct
  {
    int SM;
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] =
  {
    {0x30, 192},
    {0x32, 192},
    {0x35, 192},
    {0x37, 192},
    {0x50, 128},
    {0x52, 128},
    {0x53, 128},
    {0x60,  64},
    {0x61, 128},
    {0x62, 128},
    {0x70,  64},
    {0x72,  64},
    {0x75,  64},
    {0x80,  64},
    {0x86, 128},
    {-1, -1}
  };

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1)
  {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
    {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

bool has_fp16(int major, int minor)
{
  int cc = major * 10 + minor;
  return ((cc == 60) || (cc == 62) || (cc == 70) || (cc == 75));
}
bool has_fp16_hfma2(int major, int minor)
{
  int cc = major * 10 + minor;
  return (cc == 80);
}
bool has_bf16(int major, int minor)
{
  int cc = major * 10 + minor;
  return (cc == 80);
}
bool has_int8(int major, int minor)
{
  int cc = major * 10 + minor;
  return ((cc == 61) || (cc == 70) || (cc == 75) || (cc == 80));
}
bool has_tensor_core_v1(int major, int minor)
{
  int cc = major * 10 + minor;
  return ((cc == 70) || (cc == 72) );
}
bool has_tensor_core_v2(int major, int minor)
{
  int cc = major * 10 + minor;
  return (cc == 75);
}
bool has_tensor_core_v3(int major, int minor)
{
  int cc = major * 10 + minor;
  return (cc == 80) || (cc == 86);
}

int main(int argc, char **argv)
{
  cudaDeviceProp prop;
  int dc;
  CHECK_CUDA(cudaGetDeviceCount(&dc), "cudaGetDeviceCount error!");
  printf("GPU count = %d\n", dc);

  for(int i = 0; i < dc; i++)
  {
    printf("=================GPU #%d=================\n", i);
    CHECK_CUDA(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties error");
    printf("GPU Name = %s\n", prop.name);
    printf("Compute Capability = %d.%d\n", prop.major, prop.minor);
    printf("GPU SMs = %d\n", prop.multiProcessorCount);
    printf("GPU CUDA cores = %d\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount);
    printf("GPU SM clock rate = %.3f GHz\n", prop.clockRate/1e6);
    printf("GPU Mem clock rate = %.3f GHz\n", prop.memoryClockRate/1e6);
    printf("FP32 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2);
    if(has_fp16(prop.major, prop.minor))
    {
      printf("FP16 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 2);
    }
    if(has_fp16_hfma2(prop.major, prop.minor))
    {
      printf("FP16 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 4);
    }
    if(has_bf16(prop.major, prop.minor))
    {
      printf("BF16 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 2);
    }
    if(has_int8(prop.major, prop.minor))
    {
      printf("INT8 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 4);
    }
    if(has_tensor_core_v1(prop.major, prop.minor))
    {
      printf("Tensor Core FP16 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 8);
    }
    if(has_tensor_core_v2(prop.major, prop.minor))
    {
      printf("Tensor Core FP16 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 8);
      printf("Tensor Core INT8 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 16);
    }
    if(has_tensor_core_v3(prop.major, prop.minor))
    {
      printf("Tensor Core TF32 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 8);
      printf("Tensor Core FP16 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 16);
      printf("Tensor Core BF16 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 16);
      printf("Tensor Core INT8 Peak Performance = %.3f GFLOPS\n", cc2cores(prop.major, prop.minor) * prop.multiProcessorCount * (prop.clockRate / 1e6) * 2 * 32);
    }
  }
  return 0;
}