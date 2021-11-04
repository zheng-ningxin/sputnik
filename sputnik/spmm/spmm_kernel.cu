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
#include "iostream"
#include "sstream"
#include "cuda.h"
#include "time.h"
#include "memory"
#include "cublas_v2.h"
#include "vector"

using namespace std;
using namespace sputnik;

int32_t * row_idx, *col_idx, *d_row_idx, *d_col_idx, *row_swizzle, *d_row_swizzle;
int32_t row_idx_size, col_idx_size, values_size;
float * values, *d_values;
float * matA, *matB, matC,*d_matA, *d_matB, *d_matC;
int m, k, n;

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

void init(float * ptr, size_t length, float sparsity)
{
    for (int i = 0; i < length; i++)
    {
        float pro = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        if (pro < sparsity)
        {
            ptr[i] = 0.0;
        }
        else
        {
            ptr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
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
void convert_csr(float * ptr, int32_t row, int32_t col, int32_t * &row_idx, int32_t * &col_idx, float * &values)
{
    auto v_row_idx = std::make_shared<vector<int32_t>>();
    auto v_col_idx = std::make_shared<vector<int32_t>>();
    auto v_values = std::make_shared<vector<float>>();

    for (int i = 0; i < row; i++)
    {
        v_row_idx->push_back(v_values->size());
        for (int j = 0; j < col; j++)
        {
            size_t pos = i * col + j;
            if (ptr[pos] < 1e-8)
            {
                // sparsity
                continue;
            }
            else
            {
                v_values->push_back(ptr[pos]);
                v_col_idx->push_back(j);
            }
        }
    }
    v_row_idx->push_back(v_values->size());
    row_idx_size = sizeof(int32_t)*v_row_idx->size();
    col_idx_size = sizeof(int32_t)*v_col_idx->size();
    values_size = sizeof(float)*v_values->size();
    row_idx = (int32_t*) malloc(row_idx_size);
    col_idx = (int32_t*) malloc(col_idx_size);
    values = (float*) malloc(values_size);
    memcpy(row_idx, v_row_idx->data(), row_idx_size);
    memcpy(col_idx, v_col_idx->data(), col_idx_size);
    memcpy(values, v_values->data(), values_size);
}
int main()
{
    string row_f, col_f, value_f;
    int m=768, k=768, n=4096;
    int sparsity = 0.95;
    matA = (float*) malloc(sizeof(float)*m*k);
    matB = (float*) malloc(sizeof(float)*k*n);
    matC = (float*) malloc(sizeof(float)*m*n);
    init(matA, m*k, sparsity);
    init(matB, k*n, 0);
    convert_csr(matA, m, k, row_idx, col_idx, values);

    cublasHandle_t cublas_handle;
    CUSPARSE_SAFE_CALL(cusparseCreate(&cusparse_handle));
    CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle));
    CUDA_SAFE_CALL(cudaMalloc(&d_matA, sizeof(float)*m*k));
    CUDA_SAFE_CALL(cudaMalloc(&d_matB, sizeof(float)*n*k));
    CUDA_SAFE_CALL(cudaMalloc(&d_row_idx, row_idx_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_col_idx, col_idx_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_values, values_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_matC, sizeof(float)*m*n));
    CUDA_SAFE_CALL(cudaMemcpy(d_matA, matA, sizeof(float)*m*k, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_matB, matB, sizeof(float)*n*k, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_row_idx, row_idx, row_idx_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_col_idx, col_idx, col_idx_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_values, values, values_size, cudaMemcpyHostToDevice));
    row_swizzle = (int *) malloc(sizeof(int) * m);
    CUDA_SAFE_CALL(cudaMalloc(&d_row_swizzle, sizeof(int)*m));
    SortedRowSwizzle(m, row_idx, row_swizzle);
    CUDA_SAFE_CALL(cudaMemcpy(row_swizzle, d_row_swizzle, sizeof(int)*m, cudaMemcpyHostToDevice));
    int nnz = values_size / sizeof(float);
    
    CUDA_SAFE_CALL(CudaSpmm(m, k, n, nnz, d_row_swizzle, d_values, d_row_idx, d_col_idx, matB, d_matC, 0));
    CUDA_SAFE_CALL(cudaMemcp(matC, d_matC, sizeof(float)*m*n), cudaMemcpyDeviceToHost);
    for(int i=0;i<20;i++)
        std::cout<<matC[i]<<" ";
    std::cout<<std::endl;
    CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_matA, m, d_matB, k, &beta, d_matC, m));
    CUDA_SAFE_CALL(cudaMemcp(matC, d_matC, sizeof(float)*m*n), cudaMemcpyDeviceToHost);
    for(int i=0;i<20;i++)
        std::cout<<matC[i]<<" ";
    std::cout<<std::endl;
    return 0;
}