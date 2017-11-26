NVCC=nvcc -O3 -arch sm_35

all: dense

dense: matrix_io.o dense_experiment.o
	$(NVCC) -o dense matrix_io.o dense_experiment.o

io: matrix_io.o matrix_io_test.o
	$(NVCC) -o io matrix_io.o matrix_io_test.o

dense_experiment.o: dense_experiment.c
	$(NVCC) -c dense_experiment.c

matrix_io.o: matrix_io.c matrix_io.h
	$(NVCC) -c matrix_io.c

matrix_io_test: matrix_io_test.c
	$(NVCC) -c matrix_io_test.c


clean:
	rm -f *.o dense io
