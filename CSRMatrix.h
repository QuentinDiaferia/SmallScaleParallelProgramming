#ifndef CSRMATRIX_H
#define CSRMATRIX_H

#include <iostream>
#include <vector>
#include <stlib>
#include "mmio.h"
#include <omp.h>

using namespace std;

class CSRMatrix {
private:
	int rows;
	int cols;
	int nz;
	vector<int> irp;
	vector<int> ja;
	vector<double> as;
public:
	// CONSTRUCTORS
	CSRMatrix(char* matrixFile);

	// METHODS
	int getRows();
	int getCols();
	int getNz();
	vector<int> getIrp();
	vector<int> getJa();
	vector<double> getAs();
	vector<double> mult(const vector<double> v);
	vector<double> OpenMPmult(const vector<double> v, int nthreads);
};

#endif