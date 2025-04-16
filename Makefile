all: sims cudmat

sims: bin/sims.o
	g++ -shared bin/sims.o -o lib/libsims.so

bin/sims.o: src/cpp/sims.cpp src/cpp/sims.h 
	g++ -c -Wall -fpic src/cpp/sims.cpp -o bin/sims.o

cudmat: src/cpp/cuda_mat.cu src/cpp/cuda_mat.h
	nvcc -shared -Xcompiler -fPIC -lcublas src/cpp/cuda_mat.cu -o lib/libcudmat.so
