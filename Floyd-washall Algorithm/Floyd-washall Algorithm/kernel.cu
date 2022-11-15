
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#define GRAPH_SIZE 2000
#define EDGE_COST(graph, graph_size, a, b) graph[a * graph_size + b]
#define D(a, b) EDGE_COST(output, graph_size, a, b)
#define INF 0x1fffffff

int dis = 0;
int k = 4;

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
                r = rand() % 40;
                if (r > 20) {
                    r = INF;
                }

                D(i, j) = r;
            }
        }
    }
}
__global__
void floyd_warshall_cpu(const int* graph, int graph_size, int* output) {
    int i, j, k,t;

    memcpy(output, graph, sizeof(int) * graph_size * graph_size);

    for (k = 0; k < graph_size; k++) {
        for (i = 0; i < graph_size; i++) {
            for (j = 0; j < graph_size; j++) {

                t = D(i, k) + D(k, j);
                D(i, j) = t * (t < D(i, j)) + D(i, j) * (t >= D(i, j));

            }
        }
    }
}

void floyd_warshall_gpu(const int* graph, int graph_size, int* output) {
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
    int* graph, * output_cpu, * output_gpu;
    int size;

    size = sizeof(int) * GRAPH_SIZE * GRAPH_SIZE;

    graph = (int*)malloc(size);
    assert(graph);

    output_cpu = (int*)malloc(size);
    assert(output_cpu);
    memset(output_cpu, 0, size);

    output_gpu = (int*)malloc(size);
    assert(output_gpu);

    generate_random_graph(graph, GRAPH_SIZE);
    fprintf(stderr, "running on cpu...\n");
    floyd_warshall_cpu(graph, GRAPH_SIZE, output_cpu);
    fprintf(stderr, "running on gpu...\n");
    floyd_warshall_gpu(graph, GRAPH_SIZE, output_gpu);
    bool works = output_cpu == output_gpu;
    //fprintf("%d\n",output_cpu);
    //fprintf("%d\n",output_gpu);
    printf("%d\n", works);

    if (memcmp(output_cpu, output_gpu, size) != 0) {
        fprintf(stderr, "FAIL!\n");
    }
    return 0;
}