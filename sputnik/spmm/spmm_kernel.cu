#include <cmath>
#include <cstdint>
#include <cudnn.h>
#include <cuda.h>
#include <fstream>
#include <cuda_runtime.h>
#include "sputnik/cuda_utils.h"
#include "sputnik/matrix_utils.h"
#include "sputnik/spmm/cuda_spmm.h"
#include "sputnik/spmm/spmm_config.h"
#include "sputnik/test_utils.h"
using namespace std;
using namespace sputnik;

#define CUDA_SAFE_CALL(x)                                                                          \
    do                                                                                             \
    {                                                                                              \
        cudaError_t result = (x);                                                                  \
        if (result != cudaSuccess)                                                                 \
        {                                                                                          \
            const char* msg = cudaGetErrorString(result);                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #x " failed with error"                                    \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
#define CUDNN_SAFE_CALL(func)                                                                      \
    do                                                                                             \
    {                                                                                              \
        cudnnStatus_t e = (func);                                                                  \
        if (e != CUDNN_STATUS_SUCCESS)                                                             \
        {                                                                                          \
            const char* msg = cudnnGetErrorString(e);                                              \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #func " failed with error"                                 \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
#define CUBLAS_SAFE_CALL(func)                                                                     \
    do                                                                                             \
    {                                                                                              \
        cublasStatus_t e = (func);                                                                 \
        if (e != CUBLAS_STATUS_SUCCESS)                                                            \
        {                                                                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #func " failed with error"                                 \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << e;    \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)

size_t load_from_file(char* ptr, size_t buff_size, string file_path)
{
    std::ifstream fin(file_path, ios::in | ios::binary);
    size_t loaded_size = fin.read(ptr, buff_size).gcount();
    return loaded_size;
}


void SortedRowSwizzle(int rows, int *row_offsets, int *row_indices) {
  // Create our unsorted row indices.
  std::vector<int> swizzle_staging(rows);
  std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);

  // Argsort the row indices based on their length.
  std::sort(swizzle_staging.begin(), swizzle_staging.end(),
            [&row_offsets](int idx_a, int idx_b) {
              int length_a = row_offsets[idx_a + 1] - row_offsets[idx_a];
              int length_b = row_offsets[idx_b + 1] - row_offsets[idx_b];
              return length_a > length_b;
            });

  // Copy the ordered row indices to the output.
  std::memcpy(row_indices, swizzle_staging.data(), sizeof(int) * rows);
}
int main()
{
    string row_f, col_f, value_f;
    int m, k, n;
    cin >> m >> k >> n >> row_f >> col_f >> value_f;

    int * d_row_index, *d_col_index, *d_row_swizzle;
    float *d_values, *d_dense_m, *d_output_m;
    int * row_index, *col_index, *row_swizzle;
    float *values, *dense_m, *output_m;
    size_t enough_memory_size = 4096 * 4096 * sizeof(float);
    row_index = (int *)malloc(enough_memory_size);
    col_index = (int *)malloc(enough_memory_size);
    row_swizzle = (int *)malloc(enough_memory_size);
    values = (float *)malloc(enough_memory_size);
    dense_m = (float*) malloc(enough_memory_size);
    output_m = (float*) malloc(enough_memory_size);

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_row_index, enough_memory_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_col_index, enough_memory_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_values, enough_memory_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_output_m, enough_memory_size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_row_swizzle, enough_memory_size));

    load_from_file((char*)row_index, enough_memory_size, row_f);
    load_from_file((char*)col_index, enough_memory_size, col_f);
    SortedRowSwizzle(m, row_index, row_swizzle);
    CUDA_SAFE_CALL(cudaMemcpy(row_index, d_row_index, enough_memory_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(col_index, d_col_index, enough_memory_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(row_swizzle, d_row_swizzle, enough_memory_size, cudaMemcpyHostToDevice));

    size_t nnz = load_from_file((char*)values, enough_memory_size, value_f)/sizeof(float);

    CUDA_SAFE_CALL(cudaMalloc((void**)&dense_m, sizeof(float)*k*n));
    CUDA_SAFE_CALL(CudaSpmm(m, k, n, nnz, row_swizzle, values, row_index, col_index, dense_m, output_m, 0));


    return 0;
}