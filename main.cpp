#include <iostream>
#include "CSRMatrix.h"
#include "ELLPACKMatrix.h"
#include "mmio.h"
#include <omp.h>
#include <ctime>

using namespace std;

int main() {
	double start_par, end_par;
	double time_ini, time_end, time_cpu;
	double totalTime;
	vector<double> v1;
	vector<double> res;
	int max_threads = omp_get_max_threads();
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

	v1.resize(m.getRows(), 2);
	v1.resize(m2.getRows(), 2);

	// OPENMP

	// -- CSR

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

	// -- ELLPACK

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

	cout << endl;

	return 0; 
}