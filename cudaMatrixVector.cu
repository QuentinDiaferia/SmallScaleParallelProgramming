#include <iostream>
#include "CSRMatrix.h"
#include "ELLPACKMatrix.h"
#include "mmio.h"
#include <ctime>
#include <cuda.h>

#include <cuda_runtime.h>

using namespace std;

__global__ 
void CSRMult(const int *irp, const int* ja, const double* as, const double *v, double *result, const int rows) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < rows) {
		double sum = 0;
		for (int j = irp[i]; j < irp[i + 1]; j++) {
			sum += as[j] * v[ja[j-1]];
		}
		result[i] = sum ;
	}
}

__global__ 
void ELLPACKMult(const int maxnz, const int* ja, const double* as, const double *v, double *res, const int rows) {
	/*
	for (int i = 0; i < rows; i++) {
		res[i] = 0;
		for (int j = 0; j < maxnz; j++) {
			res[i] += as[i][j] * v[ja[i][j]];
		}
	}
	*/
}

int main() {
	double time_ini, time_end, time_cpu, total_time;
	char* file = "matrices/cage4.mtx";

	// CONVERSION

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

	// CUDA

	int *irp, *ja;
	int *_irp, *_ja;
	double *as, *v, *result;
	double *_as, *_v, *_result;
	vector<int> vIrp = m.getIrp();
	vector<int> vJa = m.getJa();
	vector<double> vAs = m.getAs();


	irp = (int *)malloc((m.getRows() + 1 ) * sizeof(int));
	ja = (int *)malloc(m.getNz() * sizeof(int));
	as = (double *)malloc(m.getNz() * sizeof(double));
	v = (double *)malloc(m.getRows() * sizeof(double));
	result = (double *)malloc(m.getRows() * sizeof(double));

	// Vector<> to simple arrays

	for (int i = 0; i < m.getRows() + 1; i++) {
		irp[i] = vIrp[i];
		if (i < m.getRows()) {
			v[i] = 2.0;
			result[i] = 0.0;
		}
	}
	for (int i = 0; i < m.getNz(); i++) {
		ja[i] = vJa[i];
		as[i] = vAs[i];
	}

	cudaMalloc((void**)&_irp, sizeof(int) * (m.getRows() + 1));
	cudaMalloc((void**)&_ja, sizeof(int) * m.getNz());
	cudaMalloc((void**)&_as, sizeof(double) * m.getNz());
	cudaMalloc((void**)&_v, sizeof(double) * m.getRows());
	cudaMalloc((void**)&_result, sizeof(double) * m.getRows());

	cudaMemcpy(_irp, irp, sizeof(int) * (m.getRows() + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(_ja, ja, sizeof(int) * m.getNz(), cudaMemcpyHostToDevice);
	cudaMemcpy(_as, as, sizeof(double) * m.getNz(), cudaMemcpyHostToDevice);
	cudaMemcpy(_v, v, sizeof(double) * m.getRows(), cudaMemcpyHostToDevice);
	cudaMemcpy(_result, result, sizeof(double) * m.getRows(), cudaMemcpyHostToDevice);

	int BLOCK_DIM = 128;
	int GRID_DIM = m.getRows() / 128 + 1;

	total_time = 0.0;
	for (int i = 0; i < 10; i++) {

		time_ini = clock();

		CSRMult<<<GRID_DIM, BLOCK_DIM>>>(_irp, _ja, as, _v, _result, m.getRows());
		cudaDeviceSynchronize();

		time_end = clock();
		time_cpu = (time_end - time_ini) / CLOCKS_PER_SEC;
		total_time += time_cpu;
	}

	total_time /= 10;

	cudaMemcpy(result, _result, sizeof(double) * m.getRows(), cudaMemcpyDeviceToHost);

	for (int i = 0; i < m.getRows(); i++)
		cout << result[i] << endl;

	cout << endl << "average time  : " << total_time << endl;
	cout << "FLOPS  : " << 2 * m.getNz() / total_time << endl << endl;



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

	return 0;
}