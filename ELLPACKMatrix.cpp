#include "ELLPACKMatrix.h"

using namespace std;

// CONSTRUCTORS
ELLPACKMatrix::ELLPACKMatrix(char* matrixFile) {
	cout << "Conversion to ELLPACK" << endl;

	int ret_code;
	MM_typecode matcode;
	FILE *f;
	int i;
	int *I, *J;
	double *val;

	int* nzTab;

	cout << "reading file... ";

	f = fopen(matrixFile, "r");

	mm_read_banner(f, &matcode);
	ret_code = mm_read_mtx_crd_size(f, &rows, &cols, &nz);
	I = (int *)malloc(nz * sizeof(int));
	J = (int *)malloc(nz * sizeof(int));
	val = (double *)malloc(nz * sizeof(double));
	nzTab = (int *)malloc(rows * sizeof(int));

	// MAXNZ CONSTRUCTION

	for (i = 0; i < rows; i++) {
		nzTab[i] = 0;
	}

	for (i = 0; i < nz; i++) {
		fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
		I[i]--;  /* adjust from 1-based to 0-based */
		J[i]--;

		nzTab[I[i]]++;
	}

	if (f != stdin) fclose(f);

	cout << "file read" << endl;

	cout << "maxnz construction... ";

	maxnz = nzTab[0];
	for (i = 1; i < rows; i++) {
		if (nzTab[i] > maxnz) {
			maxnz = nzTab[i];
		}
	}

	cout << "maxnz built = " << maxnz << endl;
	cout << "ja and as construction... ";

	// JA AND AS CONSTRUCTION

	ja.resize(rows);
	as.resize(rows);

	for (i = 0; i < nz; i++) {
		ja[I[i]].push_back(J[i]);
		as[I[i]].push_back(val[i]);
	}

	for (i = 0; i < rows; i++) {
		//cout << "row " << i << " : ja... ";
		if (ja[i].size() < maxnz) {
			ja[i].resize(maxnz, ja[i][ja[i].size() - 1]);
		}
		//cout << "ja done. as... ";
		if (as[i].size() < maxnz) {
			as[i].resize(maxnz, 0);
		}
		//cout << i << "as done. over." << endl;
	}

	cout << "ja and as built" << endl;

	free(nzTab);
	free(I);
	free(J);
	free(val);
}

// METHODS
int ELLPACKMatrix::getRows() {
	return rows;
}

int ELLPACKMatrix::getCols() {
	return cols;
}

int ELLPACKMatrix::getNz() {
	return nz;
}

int ELLPACKMatrix::getMaxnz() {
	return maxnz;
}

vector< vector<int> > ELLPACKMatrix::getJa() {
	return ja;
}

vector< vector<double> > ELLPACKMatrix::getAs() {
	return as;
}

vector<double> ELLPACKMatrix::mult(const vector<double> v) {
	vector<double> res(rows);
	for (int i = 0; i < rows; i++) {
		res[i] = 0;
		for (int j = 0; j < maxnz; j++) {
			res[i] += as[i][j] * v[ja[i][j]];
		}
	}
	return res;
}

vector<double> ELLPACKMatrix::OpenMPmult(const vector<double> v, int nthreads) {
	vector<double> res(rows);
	/*int i, j;
	omp_set_num_threads(nthreads);
	#pragma omp parallel for \
	private (i, j) shared (res) \
	schedule (static, 1)
	for (i = 0; i < rows; i++) {
		res[i] = 0;
		for (j = 0; j < maxnz; j++) {
			res[i] += as[i][j] * v[ja[i][j]];
		}
	}*/
	return res;
}