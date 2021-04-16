#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include "omp.h"

void generate_random(double *input, size_t size) {
    for (size_t i = 0; i < size; i++) {
        input[i] = rand() / (double)(RAND_MAX);
    }
}

void compute_statistics(double *times, size_t n_runs, double *avg, double *std_dev) {
    *avg = 0;
    for (int i = 0; i < n_runs; ++i) {
        *avg += times[i];
    }
    *avg /= n_runs;

    *std_dev = 0;
    for (int i = 0; i < n_runs; ++i) {
        *std_dev += pow(times[i] - *avg, 2);    
    }
    *std_dev /= n_runs;
    *std_dev = sqrt(*std_dev);
}

double serial_sum(double *x, size_t size) {
    double sum_val = 0.0;

    for (size_t i = 0; i < size; i++) {
        sum_val += x[i];
    }

    return sum_val;
}

double omp_sum(double *x, size_t size) {
    double sum_val = 0.0;

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        sum_val += x[i];
    }

    return sum_val;
}

double omp_critical_sum(double *x, size_t size) {
    double sum_val = 0.0;

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        #pragma omp critical
        {
            sum_val += x[i];
        }
    }

    return sum_val;
}

double omp_local_sum(double *x, size_t size, int n_threads) {
    double sum_val;
    double sum[n_threads];

    #pragma omp parallel shared(sum)
    {
        int id = omp_get_thread_num();
        sum[id] = 0;
        #pragma omp for
        for (size_t i = 0; i < size; i++) {
            sum[id] += x[i];
        }
    }

    sum_val = 0;
    for (size_t i = 0; i < n_threads; ++i) {
        sum_val += sum[i];
    }

    return sum_val;
}

typedef struct {
    double val;
    char pad[128];
} tSum;

double opt_local_sum(double *x, size_t size, int n_threads) {
    double sum_val;
    tSum sum[n_threads];

    #pragma omp parallel shared(sum)
    {
        int id = omp_get_thread_num();
        sum[id].val = 0;
        #pragma omp for
        for (size_t i = 0; i < size; i++) {
            sum[id].val += x[i];
        }
    }

    sum_val = 0;
    for (size_t i = 0; i < n_threads; ++i) {
        sum_val += sum[i].val;
    }

    return sum_val;
}

void measure_serial_sum(double *x, size_t size, size_t n_runs) {
    double start_time;
    double times[n_runs];
    double warmup_sum, total_sum;

    // Warmup run
    warmup_sum = serial_sum(x, size);
    
    for (int i = 0; i < n_runs; ++i) {
        start_time = omp_get_wtime();

        // Actual computation
        total_sum = serial_sum(x, size);

        times[i] = omp_get_wtime() - start_time;
        
        if (abs(total_sum - warmup_sum) > 0.001)
            printf("Different results in serial_sum\n");
    }

    // Print average times and standard deviation
    double avg, std_dev;

    compute_statistics(times, n_runs, &avg, &std_dev);
    printf("serial sum - Average: %fs - Std. deviation: %fs - Sum value: %f\n", avg, std_dev, total_sum);
}

void measure_omp_sum(double *x, size_t size, size_t n_runs) {
    double start_time;
    double times[n_runs];
    double warmup_sum, total_sum;

    // Warmup run
    omp_set_num_threads(1);
    warmup_sum = omp_sum(x, size);

    // Vary the number of threads: 1, 2, 4, 8, 16, 20, 24, 28, 32 
    for (int n_threads = 1; n_threads <= 32; n_threads = (n_threads <= 8) ? 2 * n_threads : n_threads + 4) {
        omp_set_num_threads(n_threads);

        for (int i = 0; i < n_runs; ++i) {
            start_time = omp_get_wtime();

            // Actual computation
            total_sum = omp_sum(x, size);

            times[i] = omp_get_wtime() - start_time;
            
            if (abs(total_sum - warmup_sum) > 0.001)
                printf("Different results in omp_sum with %2d threads: %f\n", n_threads, total_sum);
        }

        // Print average times and standard deviation
        double avg, std_dev;

        compute_statistics(times, n_runs, &avg, &std_dev);
        printf("omp sum (%2d threads) - Average: %fs - Std. deviation: %fs - Sum value: %f\n", n_threads, avg, std_dev, total_sum);
    }
}

void measure_omp_critical_sum(double *x, size_t size, size_t n_runs) {
    double start_time;
    double times[n_runs];
    double warmup_sum, total_sum;

    // Warmup run
    omp_set_num_threads(1);
    warmup_sum = omp_critical_sum(x, size);

    // Vary the number of threads: 1, 2, 4, 8, 16, 20, 24, 28, 32 
    for (int n_threads = 1; n_threads <= 32; n_threads = (n_threads <= 8) ? 2 * n_threads : n_threads + 4) {
        omp_set_num_threads(n_threads);

        for (int i = 0; i < n_runs; ++i) {
            start_time = omp_get_wtime();

            // Actual computation
            total_sum = omp_critical_sum(x, size);

            times[i] = omp_get_wtime() - start_time;
            
            if (abs(total_sum - warmup_sum) > 0.001)
                printf("Different results in omp_critical_sum with %2d threads: %f\n", n_threads, total_sum);
        }

        // Print average times and standard deviation
        double avg, std_dev;

        compute_statistics(times, n_runs, &avg, &std_dev);
        printf("omp critical sum (%2d threads) - Average: %fs - Std. deviation: %fs - Sum value: %f\n", n_threads, avg, std_dev, total_sum);
    }
}

void measure_omp_local_sum(double *x, size_t size, size_t n_runs) {
    double start_time;
    double times[n_runs];
    double warmup_sum, total_sum;

    // Warmup run
    omp_set_num_threads(1);
    warmup_sum = omp_local_sum(x, size, 1);

    // Vary the number of threads: 1, 2, 4, 8, 16, 20, 24, 28, 32 
    for (int n_threads = 1; n_threads <= 32; n_threads = (n_threads <= 8) ? 2 * n_threads : n_threads + 4) {
        omp_set_num_threads(n_threads);

        for (int i = 0; i < n_runs; ++i) {
            start_time = omp_get_wtime();

            // Actual computation
            total_sum = omp_local_sum(x, size, n_threads);

            times[i] = omp_get_wtime() - start_time;
            
            if (abs(total_sum - warmup_sum) > 0.001)
                printf("Different results in omp_local_sum with %2d threads: %f\n", n_threads, total_sum);
        }

        // Print average times and standard deviation
        double avg, std_dev;

        compute_statistics(times, n_runs, &avg, &std_dev);
        printf("omp local sum (%2d threads) - Average: %fs - Std. deviation: %fs - Sum value: %f\n", n_threads, avg, std_dev, total_sum);
    }
}

void measure_opt_local_sum(double *x, size_t size, size_t n_runs) {
    double start_time;
    double times[n_runs];
    double warmup_sum, total_sum;

    // Warmup run
    omp_set_num_threads(1);
    warmup_sum = opt_local_sum(x, size, 1);

    // Vary the number of threads: 1, 2, 4, 8, 16, 20, 24, 28, 32 
    for (int n_threads = 1; n_threads <= 32; n_threads = (n_threads <= 8) ? 2 * n_threads : n_threads + 4) {
        omp_set_num_threads(n_threads);

        for (int i = 0; i < n_runs; ++i) {
            start_time = omp_get_wtime();

            // Actual computation
            total_sum = opt_local_sum(x, size, n_threads);

            times[i] = omp_get_wtime() - start_time;
            
            if (abs(total_sum - warmup_sum) > 0.001)
                printf("Different results in opt_local_sum with %2d threads: %f\n", n_threads, total_sum);
        }

        // Print average times and standard deviation
        double avg, std_dev;

        compute_statistics(times, n_runs, &avg, &std_dev);
        printf("opt local sum (%2d threads) - Average: %fs - Std. deviation: %fs - Sum value: %f\n", n_threads, avg, std_dev, total_sum);
    }
}

int main() {
    const size_t SIZE = 10000000;
    const size_t NRUNS = 10;
    double *x;

    srand(time(NULL));
    
    x = malloc(SIZE * sizeof(double));
    generate_random(x, SIZE);
    
    printf("==========\n");
    measure_serial_sum(x, SIZE, NRUNS);
    printf("==========\n");
    measure_omp_sum(x, SIZE, NRUNS);
    printf("==========\n");
    measure_omp_critical_sum(x, SIZE, NRUNS);
    printf("==========\n");
    measure_omp_local_sum(x, SIZE, NRUNS);
    printf("==========\n");
    measure_opt_local_sum(x, SIZE, NRUNS);
    printf("==========\n");

    free(x);

    return 0;
}
