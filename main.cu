// #include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <sstream>
// #include <cstring>
#include <chrono>
// #include <cstdlib>
// #include <stdio.h>
// #include <math.h>
// #include <cuda_runtime_api.h>

using namespace std;

#define BLOCK_SIZE 16
#define MAX_VAL 4294967295 
// #define BLOCK_SIZE_4 4
// #define BLOCK_SIZE_8 8

// __global__ void find_zero_tiles(int* matrix, int n, int m, int* result)
// {
//     // __shared__ int tile[BLOCK_SIZE][BLOCK_SIZE];
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int x = bx * BLOCK_SIZE + tx;
//     int y = by * BLOCK_SIZE + ty;


//     // Load tile data into shared memory
//     // tile[ty][tx] = matrix[y * n + x];

//     __syncthreads();

//     // Check if tile is all zeroes
//     int sum = 0;
//     if (tx < m && ty < m)
//     {
//         if(matrix[y * n + x] != 0){
//             atomicAdd(&sum, 1);
//         }
//         __syncthreads();
//     }
//     if(sum == 0){
//         result[bx * (n/m) + by] = 1;
//     }
//     else{
//         result[bx * (n/m) + by] = 0;
//     }
//     __syncthreads();
// }


__global__ void blockwise_matrix_multiply(unsigned int *A, unsigned int *B, unsigned int *C, int *block_indices_A, int *block_indices_B, int num_blocks, int block_size, int *result) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * block_size + ty;
    int col = bx * block_size + tx;

    int mat_size = num_blocks * block_size;

    __shared__ unsigned int shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int shared_B[BLOCK_SIZE][BLOCK_SIZE];
    // __shared__ unsigned int shared_C[BLOCK_SIZE][BLOCK_SIZE];
    // __shared__ unsigned int flag;

    if(tx < block_size && ty < block_size){
        unsigned int sum = 0;
        for (int i = 0; i < num_blocks; i++) {
            int A_block_index = block_indices_A[by * num_blocks + i];
            int B_block_index = block_indices_B[i * num_blocks + bx];
            if(A_block_index >= 0 && B_block_index >= 0){
            
                // int *A_block = &A[A_block_index];
                // int *B_block = &B[B_block_index];
                
                shared_A[ty][tx] = A[A_block_index + ty * block_size + tx];
                // printf("%d %d %d %d %d %d\n", by, bx, ty, tx, i, shared_A[ty][tx]);
                shared_B[ty][tx] = B[B_block_index + ty * block_size + tx];
                // printf("%d %d %d %d %d %d\n", by, bx, ty, tx, i, shared_B[ty][tx]);
                // flag = 0;
                __syncthreads();

                // int sum = 0;
                for (int j = 0; j < block_size; j++) {
                    long long int k = (long long int)shared_A[ty][j] * (long long int)shared_B[j][tx];
                    k = (long long int)sum + k;
                    // if(k < MAX_VAL){
                    //     sum = k;
                    // }
                    // else{
                    //     sum = MAX_VAL;
                    // }
                    k < MAX_VAL ? sum = k : sum = MAX_VAL;
                }
                // for (int j = 0; j < block_size; j++) {
                //     sum += A[A_block_index + ty * block_size + j] * B[B_block_index + j * block_size + tx];
                // }

                // sumt += sum;
                // atomicAdd(&C[row * block_size + col], sum);
                __syncthreads();
            }
        }
        // __syncthreads();
        // shared_C[ty][tx] = sum;
        C[row * mat_size + col] = sum;
        // printf("%d %d %d %d %d \n", by, bx, ty, tx, sum);
        if(sum != 0){
            atomicAdd(&result[by * num_blocks + bx], 1);
            // flag = 1;
        }
        // __syncthreads();
    }  
}

// __global__ void matrixmultiply(int )

int main(int argc, char *argv[])
{
    // if(argc < 3){
    //     cout << "please provide two input files and a output file" << endl;
    // }
    
    string inputfile1 = argv[1];
    string inputfile2 = argv[2];
    string outputfile = argv[3];

    // cout << "problem 0" << endl;

    ifstream file1;
    file1.open(inputfile1, ios::in | ios::binary);
    if (!file1.is_open()){
        cout << "Failed to open file" << endl;
    }
    int n1, m1, k1;
    file1.read((char*) &n1, 4);
    file1.read((char*) &m1, 4);
    file1.read((char*) &k1, 4);

    int size1 = k1 * m1 * m1;

    int num_blocks1 = n1 / m1; 

    // Allocate memory on the host
    unsigned int *h_A = (unsigned int *) calloc(size1, sizeof(int));
    
    int *h_bA = (int *) calloc(num_blocks1 * num_blocks1, sizeof(int));

    for(int i = 0; i < num_blocks1; i++){
        for(int j = 0; j < num_blocks1; j++){
            h_bA[i * num_blocks1 + j] = -1;
        }
    }

    for(int b1 = 0; b1 < k1; b1++){
        int i = 0, j = 0;
        file1.read((char*) &i, 4);
        file1.read((char*) &j, 4);
        h_bA[i * num_blocks1 + j] = b1 * m1 * m1;
        for(int i1 = 0; i1 < m1; i1++){
            for(int j1 = 0; j1 < m1; j1++){
                unsigned int val = 0;
                file1.read((char*) &val, 2);
                h_A[b1 * m1 * m1 + i1 * m1 + j1] = val;
            }
        }
    }

    // for(int i = 0; i < num_blocks1*num_blocks1; i++){
    //     for(int j = 0; j < m1*m1; j++){
    //         cout << h_A[i*m1*m1 + j] << " ";
    //     }
    //     cout << endl;
    // }
    // for(int i = 0; i < num_blocks1*num_blocks1; i++){
    //     cout << h_bA[i] << " ";
    // }

    file1.close();

    ifstream file2;
    file2.open(inputfile2, ios::in | ios::binary);
    if (!file2.is_open()){
        cout << "Failed to open file" << endl;
    }
    int n2, m2, k2;
    file2.read((char*) &n2, 4);
    file2.read((char*) &m2, 4);
    file2.read((char*) &k2, 4);

    int size2 = k2 * m2 * m2;

    int num_blocks2 = n2 / m2; 

    // Allocate memory on the host
    unsigned int *h_B = (unsigned int *) calloc(size2, sizeof(int));
    
    int *h_bB = (int *) calloc(num_blocks2 * num_blocks2, sizeof(int));

    for(int i = 0; i < num_blocks2; i++){
        for(int j = 0; j < num_blocks2; j++){
            h_bB[i * num_blocks2 + j] = -1;
        }
    }

    // cout << "n:"<<n << " " <<"m:"<< m << " " <<"k:"<< k << endl;
    
    for(int b2 = 0; b2 < k2; b2++){
        int i = 0, j = 0;
        file2.read((char*) &i, 4);
        file2.read((char*) &j, 4);
        h_bB[i * num_blocks2 + j] = b2 * m2 * m2;
        for(int i1 = 0; i1 < m2; i1++){
            for(int j1 = 0; j1 < m2; j1++){
                unsigned int val = 0;
                file2.read((char*) &val, 2);
                h_B[b2 * m2 * m2 + i1 * m2 + j1] = val;
            }
        }
    }

    file2.close();

    int n = n1, m = m1, num_blocks = num_blocks1;
    if(n1 != n2 || m1 != m2){
        cout << "mismatch in matrix/block sizes" << endl;
    }

    int size3 = n * n;
    unsigned int *h_C = (unsigned int *) calloc(size3, sizeof(int));

    int *h_bC = (int *) calloc(num_blocks * num_blocks, sizeof(int));

    // Allocate memory on the device
    // int n = 1 << 15;
    // int m = 4;

    // cout << "problem1" << endl;

    // int size1 = n * n, size2 = n * n, size3 = n * n;
    // int num_blocks = n/m;

    // cout << "problem2" << endl;

    // for(int i = 0; i < n * n; i++){
    //     h_A[i] = i + 1;
    //     h_B[i] = n * n - 1;
    //     int k = i/(m * m);
    //     h_bA[k] = k * m * m;
    //     h_bB[k] = k * m * m; 
    // }
    // for(int i = 0; i < num_blocks*num_blocks; i++){
    //     cout<<h_bA[i]<<endl;
    // }

    // cout << "problem3" << endl;

    unsigned int *d_A, *d_B, *d_C;
    int *d_bA, *d_bB, *d_bC;

    cudaMalloc(&d_A, size1 * sizeof(int));
    cudaMalloc(&d_B, size2 * sizeof(int));
    cudaMalloc(&d_C, size3 * sizeof(int));
    cudaMalloc(&d_bA, num_blocks * num_blocks * sizeof(int));
    cudaMalloc(&d_bB, num_blocks * num_blocks * sizeof(int));
    cudaMalloc(&d_bC, num_blocks * num_blocks * sizeof(int));

    // cout << "problem4" << endl;

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bA, h_bA, num_blocks * num_blocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bB, h_bB, num_blocks * num_blocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bC, h_bC, num_blocks * num_blocks * sizeof(int), cudaMemcpyHostToDevice);

    // cout << "problem5" << endl;

    // Define the grid and block dimensions
    dim3 dimGrid(num_blocks, num_blocks);
    dim3 dimBlock(m, m);
    size_t sharedMemBytes = BLOCK_SIZE * BLOCK_SIZE * sizeof(int);

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start); // start timing
    auto start1 = chrono::high_resolution_clock::now();

    blockwise_matrix_multiply<<<dimGrid, dimBlock, sharedMemBytes>>>(d_A, d_B, d_C, d_bA, d_bB, num_blocks, m, d_bC);    // Launch the kernel
    
    // cudaEventRecord(stop); // stop timing
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // cout << "Time taken for kernel execution: " << milliseconds << " ms" << std::endl;

    cudaDeviceSynchronize(); 

    auto end1 = chrono::high_resolution_clock::now();
    // cout << "Time taken for gpu: " << chrono::duration_cast<chrono::milliseconds>(end1 - start1).count() << "ms" << endl;
    // cudaMemcpy(h_C, d_C, size3 * sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < n; i++){
    //     for(int j = 0; j < n; j++){
    //         cout << h_C[i * n + j] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << "problem6" << endl;

    // Copy the result matrix from device to host
    // dim3 grid(num_blocks, num_blocks);
    // dim3 block(m, m);
    // // size_t sharedMemBytes_ = 4 * BLOCK_SIZE * BLOCK_SIZE;

    // find_zero_tiles<<<grid, block>>>(d_C, n, m, d_bC);

    // cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bC, d_bC, num_blocks * num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

    int count = 0;

    // cout << endl;
    // for(int i = 0; i < n*n; i++){
    //     cout << " " << h_C[i] << " ";
    // }
    // cout << endl;
    for(int i = 0; i < num_blocks*num_blocks; i++){
        // cout << h_bC[i] << " ";
        if(h_bC[i] > 0){
            count++;
        }
    }
    // cout << endl;
    
    // cout << "problem7" << endl;

    ofstream outfile;
    outfile.open(outputfile, ios::out | ios::binary);
    if (!outfile.is_open()){
        cout << "Failed to open file" << endl;
    }
    outfile.write((char*) &n, 4);
    outfile.write((char*) &m, 4);
    outfile.write((char*) &count, 4);

    for(int i = 0; i < num_blocks; i++){
        for(int j = 0; j < num_blocks; j++){
            int addr = h_bC[i * num_blocks + j];
            int p = i;
            int q = j;
            if(addr > 0){
                outfile.write((char*) &p, 4);
                outfile.write((char*) &q, 4);
                for(int x = 0; x < m; x++){
                    for(int y = 0; y < m; y++){
                        int val = h_C[i * m * n + j * m + x * n + y];
                        outfile.write((char*) &val, 4);
                        // cout << val << " ";
                    }
                }
            }
            // cout << endl;
        }
    }
    // cout<<endl;
    // cout<<"writing complete"<<endl;
    outfile.close();

    // Print the result matrix
    // for(int i = 0; i < k1 * m * m; i++){
    //     // for(int j = 0; j < m*m; j++){
    //     //     cout << i << " " << j << " " << h_A[i*m*m + j] << " ";
    //     // }
    //     cout << " " << h_A[i] << " ";
    //     // cout << endl;
    // }

    // cout << "\n";
    // for(int i = 0; i < num_blocks*num_blocks; i++){
    //     cout << h_bA[i] << " ";
    // }
    // cout << endl;
    // for(int i = 0; i < num_blocks*num_blocks; i++){
    //     for(int j = 0; j < m*m; j++){
    //         cout << i << " " << j << " " << h_B[i*m*m + j] << " ";
    //     }
    //     cout << endl;
    // }

    // for(int i = 0; i < k2 * m * m; i++){
    //     // for(int j = 0; j < m*m; j++){
    //     //     cout << i << " " << j << " " << h_A[i*m*m + j] << " ";
    //     // }
    //     cout << " " << h_B[i] << " ";
    //     // cout << endl;
    // }

    // cout << "\n";
    // for(int i = 0; i < num_blocks*num_blocks; i++){
    //     cout << h_bB[i] << " ";
    // }
    // cout << endl;
    // for(int i = 0; i < num_blocks*num_blocks; i++){
    //     for(int j = 0; j < m*m; j++){
    //         cout << h_C[i*m*m + j] << " ";
    //     }
    //     cout << endl;
    // }
    // for(int i = 0; i < num_blocks*num_blocks; i++){
    //     cout << h_bC[i] << " ";
    // }
    // cout << endl;
    // cout << endl;
    // cout << "n1:" << n1 << " " << "n2:" << n2 << " " << "m1:" << m1 << " " << "m2:" << m2 << endl;
    
    

    // Free memory on the device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_bA);
    cudaFree(d_bB);
    cudaFree(d_bC);

    // Free memory on the host
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_bA);
    free(h_bB);
    free(h_bC);

    return 0;

}