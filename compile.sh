g++ -c CSRMatrix.cpp ELLPACKMatrix.cpp
gcc -c mmio.c
nvcc -o cudacsr cudaMatrixVector.cu CSRMatrix.o ELLPACKMatrix.o mmio.o