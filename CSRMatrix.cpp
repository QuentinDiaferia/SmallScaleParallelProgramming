#include "CSRMatrix.h"

using namespace std;

// CONSTRUCTORS
CSRMatrix::CSRMatrix(char* matrixFile) {
	cout << "Conversion to CSR" << endl;
	
	int ret_code;
	MM_typecode matcode;
	FILE *f;
	int n;
	int i, j, minj;
	int *I, *J;
	double *val;

	cout << "reading file... ";

	f = fopen(matrixFile, "r");

	mm_read_banner(f, &matcode);
	ret_code = mm_read_mtx_crd_size(f, &rows, &cols, &nz);
	I = (int *)malloc(nz * sizeof(int));
	J = (int *)malloc(nz * sizeof(int));
	val = (double *)malloc(nz * sizeof(double));

	for (i = 0; i < nz; i++) {
		fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
		I[i]--;  /* adjust from 1-based to 0-based */
		J[i]--;
	}

	if (f != stdin) fclose(f);

	cout << "file read" << endl;

	cout << "ja construction... ";

	// JA CONSTRUCTION

	ja.assign(J, J + nz);

	cout << "ja built" << endl;
	cout << "ordering... ";

	// ordering by row
	for (i = 1; i < nz; i++) {
		int elem1 = I[i];
		int elem2 = J[i];
		double elem3 = val[i];
		for (j = i; j > 0 && I[j - 1] > elem1; j--) {
			I[j] = I[j - 1];
			J[j] = J[j - 1];
			val[j] = val[j - 1];
		}
		I[j] = elem1;
		J[j] = elem2;
		val[j] = elem3;
	}

	cout << "ordered" << endl;
	cout << "as construction... ";

	// AS CONSTRUCTION

	as.assign(val, val + nz);

	cout << "as built" << endl;
	cout << "irp construction... ";

	// IRP CONSTRUCTION

	irp.resize(rows + 1);

	irp[0] = 0;
	minj = 0;
	for (i = 1; i < rows + 1; i++) {
		n = 0;
		for (j = minj; I[j] == i - 1; j++) {
			n++;
		}
		minj = j;
		irp[i] = irp[i - 1] + n;
	}

	cout << "irp built" << endl;

	free(I);
	free(J);
	free(val);
}

int CSRMatrix::getRows() {
	return rows;
}

int CSRMatrix::getCols() {
	return cols;
}

int CSRMatrix::getNz() {
	return nz;
}

vector<int> CSRMatrix::getIrp() {
	return irp;
}

vector<int> CSRMatrix::getJa() {
	return ja;
}

vector<double> CSRMatrix::getAs() {
	return as;
}

vector<double> CSRMatrix::mult(const vector<double> v) {
	vector<double> res(rows);
	for (int i = 0; i < rows; i++) {
		res[i] = 0;
		for (int j = irp[i]; j < irp[i + 1] - 1; j++) {
			res[i] += as[j] * v[ja[j]];
		}
	}
	return res;
}

vector<double> CSRMatrix::OpenMPmult(const vector<double> v, int nthreads) {
	vector<double> res(rows);
	int i, j;

	omp_set_num_threads(nthreads);
	#pragma omp parallel for \
	private (i, j) shared (res) \
	schedule (static, 1)
	for (i = 0; i < rows; i++) {
		res[i] = 0;
		for (j = irp[i]; j < irp[i + 1] - 1; j++) {
			res[i] += as[j] * v[ja[j]];
		}
	}
	return res;
}