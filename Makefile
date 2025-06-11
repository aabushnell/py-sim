all: sims cudmat

sims: sim/bin/sims.o
	g++ -shared sim/bin/sims.o -o sim/lib/libsims.so

sim/bin/sims.o: cpp/sims.cpp cpp/sims.h 
	g++ -c -Wall -fpic cpp/sims.cpp -o sim/bin/sims.o

cudmat: cpp/cuda_mat.cu cpp/cuda_mat.h
	nvcc -shared -Xcompiler -fPIC -lcublas cpp/cuda_mat.cu -o sim/lib/libcudmat.so
