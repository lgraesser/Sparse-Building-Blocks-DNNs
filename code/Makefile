NVCC=nvcc -arch=sm_35

all: dense io sparse_test

dense: matrix_io.o dense_experiment.o
	$(NVCC) -lcublas -o $@ $^

sparse_test: sparse_conversion_test.o sparse_conversion.o matrix_io.o
	$(NVCC) -lcusparse -o $@ $^

io: matrix_io.o matrix_io_test.o
	$(NVCC) -o $@ $^

%.o: %.c
	$(NVCC) -x cu -I. -dc $< -o $@

%.o: %.cu
	$(NVCC) -x cu -I. -dc $< -o $@

.PHONY: clean
clean:
	rm -f *.o dense io sparse_test
