NVCC=nvcc -arch=sm_35

all: mm io

mm: matrix_io.o mm_experiment.o mm.o
	$(NVCC) -lcublas -o $@ $^

io: matrix_io.o matrix_io_test.o
	$(NVCC) -o $@ $^

%.o: %.c
	$(NVCC) -x cu -I. -dc $< -o $@

.PHONY: clean
clean:
	rm -f *.o dense io
