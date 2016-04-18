#include <iostream>
#include "CSRMatrix.h"
#include "ELLPACKMatrix.h"
#include "mmio.h"
#include <omp.h>
#include <ctime>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>

using namespace std;

__global__ 
void CSRMult(const int *irp, const int* ja, const double* as, const double *v, double *res, const int rows) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = 0; i < rows; i++) {
		res[i] = 0;
		for (int j = irp[i]; j < irp[i + 1] - 1; j++) {
			res[i] += as[j] * v[ja[j]];
		}
	}
	return res;
}

int main() {
	double start_par, end_par;
	double time_ini, time_end, time_cpu;
	double totalTime;
	vector<double> v1;
	vector<double> res;
	int max_threads = omp_get_max_threads();
	char* file = "matrices/cage4.mtx";


	// ----- CONVERSION -----

	time_ini = clock();
	CSRMatrix m(file);
	time_end = clock();
	time_cpu = (time_end - time_ini) / CLOCKS_PER_SEC;
	cout << "Conversion to CSR : " << time_cpu << " seconds" << endl << endl;

	time_ini = clock();
	ELLPACKMatrix m2(file);
	time_end = clock();
	time_cpu = (time_end - time_ini) / CLOCKS_PER_SEC;
	cout << "Conversion to ELLPACK : " << time_cpu << endl;

	v1.resize(m.getRows(), 2);


	// ----- OPENMP -----

	// --- CSR

	/*
	for (int i = 1; i <= max_threads; i++) {
		totalTime = 0;
		for (int j = 0; j < 10; j++) {
			start_par = omp_get_wtime();
			res = m.OpenMPmult(v1, i);
			end_par = omp_get_wtime();
			totalTime += end_par - start_par;
		}
		totalTime /= 10;
		cout << i << " threads : FLOPS = " << 2 * m.getNz() / totalTime << endl << endl;
	}

	cout << endl;

	// ---- ELLPACK
	
	v1.resize(m2.getRows(), 2);

	for (int i = 1; i <= max_threads; i++) {
		totalTime = 0;
		for (int j = 0; j < 10; j++) {
			start_par = omp_get_wtime();
			res = m2.OpenMPmult(v1, i);
			end_par = omp_get_wtime();
			totalTime += end_par - start_par;
		}
		totalTime /= 10;
		cout << i << " threads : FLOPS = " << 2 * m2.getNz() / totalTime << endl << endl;
	}

	cout << endl;*/

	// ----- CUDA -----

	// ---- CSR

	int *irp, *ja;
	int *_irp, *_ja;
	double *as, *v, *result;
	double *_as, *_v, *_result;
	vector<int> vIrp = m.getIrp();
	vector<int> vJa = m.getJa();
	vector<double> vAs = m.getAs();

	irp = (int *)malloc(m.getRows() + 1 * sizeof(int));
	ja = (int *)malloc(m.getNz() * sizeof(int));
	as = (double *)malloc(m.getNz() * sizeof(double));
	v = (double *)malloc(m.getRows() * sizeof(double));
	result = (double *)malloc(m.getRows() * sizeof(double));

	for (int i = 0; i < m.getRows() + 1; i++) {
		irp[i] = vIrp[i];
		v[i] = 2;
		result[i] = 0;
	}
	for (int i = 0; i < m.getNz(); i++) {
		ja[i] = vJa[i];
		as[i] = vAs[i];
	}

	checkCudaErrors(cudaMalloc((void**)&_irp, sizeof(int) * m.getRows()));
	checkCudaErrors(cudaMalloc((void**)&_ja, sizeof(int) * m.getNz()));
	checkCudaErrors(cudaMalloc((void**)&_as, sizeof(double) * m.getNz()));
	checkCudaErrors(cudaMalloc((void**)&_v, sizeof(double) * m.getRows()));
	checkCudaErrors(cudaMalloc((void**)&_result, sizeof(double) * m.getRows()));

	checkCudaErrors(cudaMemcpy(_irp, irp, sizeof(int) * m.getRows(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_ja, ja, sizeof(int) * m.getNz(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_as, as, sizeof(double) * m.getNz(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_v, v, sizeof(double) * m.getRows(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_result, result, sizeof(double) * m.getRows(), cudaMemcpyHostToDevice));

	int BLOCK_DIM = 256;

	StopWatchInterface* timer = 0;
	sdkCreateTimer(&timer);
	timer->start();
	CSRMult << <nz, BLOCK_DIM >> >(_irp, _ja, as, _v, _result, m.getRows());

	checkCudaErrors(cudaDeviceSynchronize());
	timer->stop();

	checkCudaErrors(cudaMemcpy(result, _result, sizeof(double) * m.getRows(), cudaMemcpyDeviceToHost));

	printf("\n");
	cout << "timer: " << timer->getTime() << std::endl;
	printf("\n");

	cudaFree(_irp);
	cudaFree(_ja);
	cudaFree(_as);
	cudaFree(_v);
	cudaFree(_result);

	free(irp);
	free(ja);
	free(as);
	free(v);
	free(result);

	// ---- ELLPACK

	return 0; 
}