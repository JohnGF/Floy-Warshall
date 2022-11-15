
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>

#include <cstdio>
#include <cassert>
#include <stdlib.h>

#include <string.h>


#define GRAPH_SIZE 2000
//macro for adding a edge_cost to cell
#define EDGE_COST(graph, graph_size, a, b) graph[a * graph_size + b]
//macro for getting edgE_cost 
#define D(a, b) EDGE_COST(output, graph_size, a, b)
//macro to have type 
#define INF 0x1fffffff

#define THREADS_PER_BLOCK_SIDE 16
#define BLOCKS_PER_GRAPH_SIDE ((GRAPH_SIZE+THREADS_PER_BLOCK_SIDE-1) / THREADS_PER_BLOCK_SIDE)
#define HANDLE_ERROR(x) x



__global__ void run_on_gpu(const int graph_size, int* output, int k) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if ((i < graph_size) && (j < graph_size))
        if (D(i, k) + D(k, j) < D(i, j)) {
            D(i, j) = D(i, k) + D(k, j);
        }
}

void floyd_warshall_gpu(const int* graph, int graph_size, int* output) {
    int* dev_output;

    HANDLE_ERROR(cudaMalloc(&dev_output, sizeof(int) * graph_size * graph_size));

    cudaMemcpy(dev_output, graph, sizeof(int) * graph_size * graph_size, cudaMemcpyHostToDevice);
    dim3 blocks(BLOCKS_PER_GRAPH_SIDE, BLOCKS_PER_GRAPH_SIDE, 1);
    dim3 threadsPerBlock(THREADS_PER_BLOCK_SIDE, THREADS_PER_BLOCK_SIDE, 1);
    int k;
    for (k = 0; k < graph_size; k++) {
        run_on_gpu <<<blocks, threadsPerBlock >>> (graph_size, dev_output, k);
    }
    cudaMemcpy(output, dev_output, sizeof(int) * graph_size * graph_size, cudaMemcpyDeviceToHost);

    cudaFree(dev_output);
}

void generate_random_graph(int* output, int graph_size) {
    int i, j;

    srand(0xdadadada);

    for (i = 0; i < graph_size; i++) {
        for (j = 0; j < graph_size; j++) {
            if (i == j) {
                D(i, j) = 0;
            }
            else {
                int r;
                r = rand() % 1000;
                if (r > 20) {
                    D(i, j) = INF;
                }
                else
                    D(i, j) = r + 10;
            }
        }
    }
}

void floyd_warshall_cpu(const int* graph, int graph_size, int* output) {
    int i, j, k;

    memcpy(output, graph, sizeof(int) * graph_size * graph_size);

    for (k = 0; k < graph_size; k++) {
        for (i = 0; i < graph_size; i++) {
            for (j = 0; j < graph_size; j++) {
                if (D(i, k) + D(k, j) < D(i, j)) {
                    D(i, j) = D(i, k) + D(k, j);
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    clock_t start, end;
    int* graph, * output_cpu, * output_gpu;
    int size;
    size = sizeof(int) * GRAPH_SIZE * GRAPH_SIZE;
    //returns a pointer to memory adress for the graph
    graph = (int*)malloc(size);
    output_cpu = (int*)malloc(size);
    assert(output_cpu);
    memset(output_cpu, 0, size);
    output_gpu = (int*)malloc(size);
    generate_random_graph(graph, GRAPH_SIZE);
    start = clock();
    floyd_warshall_cpu(graph, GRAPH_SIZE, output_cpu);
    end = clock();
    double duration = ((double)end - start) / CLOCKS_PER_SEC;
    printf("Time taken to execute cpu in seconds : %f\n", duration);
    start = clock();
    floyd_warshall_gpu(graph, GRAPH_SIZE, output_gpu);
    end = clock();
    duration = ((double)end - start) / CLOCKS_PER_SEC;
    printf("Time taken to execute gpu in seconds : %f\n", duration);

    if (memcmp(output_cpu, output_gpu, size) != 0) {
        fprintf(stderr, "FAIL!\n");
        int qq = 0;
        //tries to run mismatch in the matrix
        for (int i = 0; i < GRAPH_SIZE * GRAPH_SIZE; i++)
            if (output_cpu[i] != output_gpu[i]) { qq++; printf("i: %d, cpu: %d, gpu: %d\n", i, output_cpu[i], output_gpu[i]); }
        printf("# mismatches: %d\n", qq);
    }
    else {
        fprintf(stderr, "SUCCESS!\n");
        //  for (int i = 0; i < 100; i++)
        //   printf("i: %d, cpu: %d, gpu: %d\n",i, output_cpu[i], output_gpu[i]);
    }
    //free memory
    free(graph);
    free(output_cpu);
    free(output_gpu);
    return 0;
}