// Copyright 2020 The Sputnik Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sputnik/cuda_utils.h"
#include "sputnik/matrix_utils.h"
#include "sputnik/sddmm/cuda_sddmm.h"

#include "absl/random/random.h"
#include "benchmark/benchmark.h"
#include "iostream"
#include "sstream"
#include "cuda.h"
#include "time.h"
#include "memory"
#include "cublas_v2.h"
#include "vector"
#include <fstream>
using namespace std;
using namespace sputnik;

int32_t * row_idx, *col_idx, *d_row_idx, *d_col_idx, *row_swizzle, *d_row_swizzle;
int32_t row_idx_size, col_idx_size, values_size;
float * values, *d_values;
float * matA, *matB, *matC,*d_matA, *d_matB, *d_matC;
int m, k, n;
float alpha=1.0, beta=0.0;
int * mask;

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

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



// void BM_CudaSddmm_GenericFloat(benchmark::State& state) {
//   const int kDimM = state.range(0);
//   const int kDimK = state.range(1);
//   const int kDimN = state.range(2);
//   const int kNonZeros = state.range(3);

//   const int kRowPadding = 0;

//   // Create the sparse matrix on the gpu.
//   absl::BitGen generator;
//   CudaSparseMatrix<float> output_matrix(kDimM, kDimN, kNonZeros, RANDOM_UNIFORM,
//                                         &generator, SORTED, kRowPadding);

//   // Create the dense matrix on the gpu.
//   CudaMatrix<float> lhs_matrix(kDimM, kDimK, &generator);
//   CudaMatrix<float> rhs_matrix(kDimN, kDimK, &generator);

//   int batch_size = 10;
//   while (state.KeepRunningBatch(batch_size)) {
//     for (int i = 0; i < batch_size; ++i) {
//       CUDA_CALL(CudaSddmm(
//           output_matrix.Rows(),
//           lhs_matrix.Columns(),
//           output_matrix.Columns(),
//           output_matrix.NumElementsWithPadding(),
//           output_matrix.RowIndices(),
//           output_matrix.RowOffsets(),
//           output_matrix.ColumnIndices(),
//           lhs_matrix.Values(),
//           rhs_matrix.Values(),
//           output_matrix.Values(), 0));
//     }
//     CUDA_CALL(cudaStreamSynchronize(nullptr));
//   }
//   ReportThroughput(state);
// }


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
  // printf("before memory cpy\n");
  // for(int i=0;i<rows;i++)
  //   printf("%d %d\n", i ,row_offsets[i]);
  std::memcpy(row_indices, swizzle_staging.data(), sizeof(int) * rows);
  // printf("after memory cpy\n");

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
    printf("values_size: %d\n", values_size);
    row_idx = (int32_t*) malloc(row_idx_size);
    col_idx = (int32_t*) malloc(col_idx_size);
    values = (float*) malloc(values_size);
    memcpy(row_idx, v_row_idx->data(), row_idx_size);
    memcpy(col_idx, v_col_idx->data(), col_idx_size);
    memcpy(values, v_values->data(), values_size);
}

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
        //printf("pro: %f\n", pro);
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

void load_mask(float* mat, int buffer_size, string filepath)
{
    mask = (int*) malloc(sizeof(int) * buffer_size);
    memset(mat, 0, sizeof(float)*buffer_size);
    load_from_file((char*)mask, buffer_size*sizeof(int), filepath);
    for(int i=0;i<buffer_size;i++){
        if(mask[i]){
            // printf("%d\n", mask[i]);
            mat[i] = 1.0;
        }
    }
}

int main()
{
    int m=2560, k=64, n=2560;
    srand(1);
    matA = (float*) malloc(sizeof(float)*m*k);
    matB = (float*) malloc(sizeof(float)*k*n);
    matC = (float*) malloc(sizeof(float)*m*n);
    load_mask(matC, m*n, "./mask.bin");
    init(matA, m*k, 0);
    init(matB, k*n, 0);
    convert_csr(matC, m, n, row_idx, col_idx, values);
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
    printf("before sort row swizzle\n");
    SortedRowSwizzle(m, row_idx, row_swizzle);
    printf("after sort row swizzle\n");

    CUDA_SAFE_CALL(cudaMemcpy(d_row_swizzle, row_swizzle, sizeof(int)*m, cudaMemcpyHostToDevice));
    int nnz = values_size / sizeof(float);
    printf("nnz: %d\n", nnz);

    int n_iter = 100;
    cudaEvent_t start, stop;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));
    for(int i = 0; i < n_iter; i += 1){
      CUDA_SAFE_CALL(CudaSddmm(
          m,
          k,
          n,
          nnz,
          d_row_swizzle,
          d_row_idx,
          d_col_idx,
          d_matA,
          d_matB,
          d_values, 0));
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_SAFE_CALL(cudaMemcpy(values, d_values, sizeof(float)*nnz, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    float sum1 = 0, sum2 = 0;
    for(int i=0;i<10;i++){
        printf("%f\n", values[i]);
        sum1 += values[i];
    }
    int count = 10;
    
    // cublasHandle_t cublas_handle;
    // float alpha=1.0, beta=0.0;
    // CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle));
    // CUBLAS_SAFE_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, d_matA, k, d_matB, n, &beta, d_matC, n));
    // CUDA_SAFE_CALL(cudaMemcpy(matC, d_matC, sizeof(float)*m*n, cudaMemcpyDeviceToHost));
    // for(int i=0;i<m*n && count>0; i++){
    //   if(mask[i]>0){
    //     printf("sparse: %f\n", matC[i]);
    //     count--;
    //   }
    //     // sum2+=matC[i];
    // }
    
    // transpose Sparse Matrix B
    // float* matB = (float*) malloc(sizeof(float)*k*n);
    // for(int i=0;i<k;i++){
    //     for(int j=0;j<n;j++){

    //     }
    // }

    printf("%f  %f\n",sum1, sum2);
    // #pragma omp for collapse(2)
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            float sum = 0 ;
            for(int tmp=0;tmp<k;tmp++){
                sum += matA[i*k+tmp]*matB[tmp*n+j];
            }
            matC[i*n+j] = sum;
        }
    }
    count = 10;
    
    for(int i=0;i<m*n && count>0; i++){
        // printf("%d\n",mask[i]);
        if(mask[i]>0){
            printf("dense: %f\n", matC[i]);
            count--;
        }
        // sum2+=matC[i];
    }
    
    printf("Time: %f ms\n", milliseconds / n_iter);
}