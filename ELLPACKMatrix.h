#ifndef ELLPACKMATRIX_H
#define ELLPACKMATRIX_H

#include <iostream>
#include <vector>
#include <stdlib>
#include "mmio.h"
#include <omp.h>

using namespace std;

class ELLPACKMatrix {
private:
	int rows;
	int cols;
	int nz;
	int maxnz;
	vector< vector<int> > ja;
	vector< vector<double> > as;
public:
	// CONSTRUCTORS
	ELLPACKMatrix(char* matrixFile);

	// METHODS
	int getRows();
	int getCols();
	int getNz();
	int getMaxnz();
	vector< vector<int> > getJa();
	vector< vector<double> > getAs();
	vector<double> mult(const vector<double> v);
	vector<double> OpenMPmult(const vector<double> v, int nthreads);
};

#endif
