    // the code calculates a DFT of a random complex number input and 
    // then an IDFT. The IDFT result should be the input vector 
    // to compile with gcc
    // gcc -Wall -O2 -fopenmp -o DFTW DFTW.c -lm
    // written by stef

    // exercise


	#include "stdio.h" // printf
	#include "stdlib.h" // malloc and rand for instance. Rand not thread safe!
	#include "time.h"   // time(0) to get random seed
	#include "math.h"  // sine and cosine
	#include "omp.h"   // openmp library like timing

	
	// two pi
	#define PI2 6.28318530718
	// this for the rounding error, increasing N rounding error increases
    // 0.01 precision good for N > 8000
	#define R_ERROR 0.01
	
	// main routine to calculate DFT
	int DFT(int idft, double* xr, double* xi, double* Xr_o, double* Xi_o, int N);
    // Compute mean and standard deviation
    void compute_statistics(double *times, size_t n_runs, double *avg, double *std_dev);
	// set the input array with random number
	int fillInput(double* xr, double* xi, int N);
	// set to zero the input vector
	int setOutputZero(double* Xr_o, double* Xi_o, int N);
	// check if x = IDFT(DFT(x))
	int checkResults(double* xr, double* xi, double* xr_check, double* xi_check, double* Xr_o, double* Xi_r, int N);
	// print the results of the DFT
	int printResults(double* xr, double* xi, int N);


	int main(int argc, char* argv[]){
	  // size of input array
	  int N = 10000; // 8,000 is a good number for testing
      size_t NRUNS = 10;
      int DEBUG = 0;
	  printf("DFTW calculation with N = %d \n",N);
	  
	  // Allocate array for input vector
	  double* xr = (double*) malloc (N *sizeof(double));
	  double* xi = (double*) malloc (N *sizeof(double));

	  // for checking purposes
	  double* xr_check = (double*) malloc (N *sizeof(double));
	  double* xi_check = (double*) malloc (N *sizeof(double));
	  
	  // Allocate array for output vector
	  double* Xr_o = (double*) malloc (N *sizeof(double));
	  double* Xi_o = (double*) malloc (N *sizeof(double));
      
      double times[NRUNS];
      // Vary the number of threads: 1, 2, 4, 8, 16, 20, 24, 28, 32 
      for (int n_threads = 1; n_threads <= 32; n_threads = (n_threads <= 8) ? 2 * n_threads : n_threads + 4) {
          omp_set_num_threads(n_threads);

          for (int i = 0; i < NRUNS; ++i) {
              fillInput(xr,xi,N);
              setOutputZero(xr_check,xi_check,N);
              setOutputZero(Xr_o,Xi_o,N);

              // start timer
              double start_time = omp_get_wtime();
          
              // DFT
              int idft = 1;
              DFT(idft,xr,xi,Xr_o,Xi_o,N);
              // IDFT
              idft = -1;
              DFT(idft,Xr_o,Xi_o,xr_check,xi_check,N);
          

              // stop timer
              times[i] = omp_get_wtime() - start_time;
          
              // check the results: easy to make correctness errors with openMP
              checkResults(xr,xi,xr_check,xi_check,Xr_o, Xi_o, N);
              // print the results of the DFT
              if (DEBUG) {
                  printResults(Xr_o,Xi_o,N);
              }
          }

          double avg, std_dev;

          compute_statistics(times, NRUNS, &avg, &std_dev);
          printf("DFTW (%2d threads) - Average: %fs - Std. deviation: %fs\n", n_threads, avg, std_dev);
      }

	  // take out the garbage
	  free(xr); free(xi);
	  free(Xi_o); free(Xr_o);
	  free(xr_check); free(xi_check);

	  return 0;
	}

    // DFT/IDFT routine
    // idft: 1 direct DFT, -1 inverse IDFT (Inverse DFT)
    int DFT(int idft, double* xr, double* xi, double* Xr_o, double* Xi_o, int N){
        #pragma omp parallel
        {
            double local_Xr, local_Xi;
            #pragma omp for
            for (int k=0 ; k<N ; k++) {
                local_Xr = Xr_o[k];
                local_Xi = Xi_o[k];

                for (int n=0 ; n<N ; n++) {
                    // Real part of X[k]
                    local_Xr += xr[n] * cos(n * k * PI2 / N) + idft * xi[n] * sin(n * k * PI2 / N);
                    // Imaginary part of X[k]
                    local_Xi += -idft * xr[n] * sin(n * k * PI2 / N) + xi[n] * cos(n * k * PI2 / N);
                } 

                // normalize if you are doing IDFT
                if (idft == -1) {
                    local_Xr /= N;
                    local_Xi /= N;
                }

                Xr_o[k] = local_Xr;
                Xi_o[k] = local_Xi;
            }
        }    
        return 1; 
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
	
	// set the initial signal 
    // be careful with this 
    // rand() is NOT thread safe in case
	int fillInput(double* xr, double* xi, int N){
	    srand(time(0));
	    for(int n=0; n < 100000;n++) // get some random number first 
	    	rand();
	    for(int n=0; n < N;n++){
	       // Generate random discrete-time signal x in range (-1,+1)
	       //xr[n] = ((double)(2.0 * rand()) / RAND_MAX) - 1.0;
	       //xi[n] = ((double)(2.0 * rand()) / RAND_MAX) - 1.0;
	       // constant real signal
	       xr[n] = 1.0;
	       xi[n] = 0.0;
	    }
		return 1; 
	}

	// set to zero the output vector
	int setOutputZero(double* Xr_o, double* Xi_o, int N){
	for(int n=0; n < N;n++){
	       Xr_o[n] = 0.0;
	       Xi_o[n] = 0.0; 
	    }
		return 1;
	}

	// check if x = IDFT(DFT(x))
	int checkResults(double* xr, double* xi, double* xr_check, double* xi_check, double* Xr_o, double* Xi_r, int N){
		// x[0] and x[1] have typical rounding error problem
		// interesting there might be a theorem on this
		for(int n=0; n < N;n++){
			if (fabs(xr[n] - xr_check[n]) > R_ERROR)
			    printf("ERROR - x[%d] = %f, inv(X)[%d]=%f \n",n,xr[n], n,xr_check[n]);
			if (fabs(xi[n] - xi_check[n]) > R_ERROR)
			    printf("ERROR - x[%d] = %f, inv(X)[%d]=%f \n",n,xi[n], n,xi_check[n]);

		}
		//printf("Xre[0] = %f \n",Xr_o[0]);
		return 1;
	}

	// print the results of the DFT
	int printResults(double* xr, double* xi, int N){
		for(int n=0; n < N;n++)
			    printf("Xre[%d] = %f, Xim[%d] = %f \n", n, xr[n], n, xi[n]);
		return 1;
	}
