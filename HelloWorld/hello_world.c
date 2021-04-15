#include "stdio.h"
#include "omp.h"

void main() {
#pragma omp parallel
    {
        printf("Hello World from Thread %d!\n", omp_get_thread_num());
    }
}
