NVCC=nvcc -arch=sm_35

all: dense io

dense: matrix_io.o dense_experiment.o
	$(NVCC) -lcublas -o $@ $^

io: matrix_io.o matrix_io_test.o
	$(NVCC) -o $@ $^

%.o: %.c
	$(NVCC) -x cu -I. -dc $< -o $@

clean:
	rm -f *.o dense io
