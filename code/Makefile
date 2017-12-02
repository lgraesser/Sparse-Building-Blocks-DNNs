NVCC=nvcc -arch=sm_35
SRC = src
BIN = bin
OBJ = obj
all: mm io sparse_test

mm: matrix_io.o mm_experiment.o mm.o
	$(NVCC) -lcublas -o $(BIN)/$@ $(patsubst %.o,$(OBJ)/%.o,$^)

sparse_test: sparse_conversion_test.o sparse_conversion.o matrix_io.o
	$(NVCC) -lcusparse -o $(BIN)/$@ $(patsubst %.o,$(OBJ)/%.o,$^)

io: matrix_io.o matrix_io_test.o
	$(NVCC) -o $(BIN)/$@ $(patsubst %.o,$(OBJ)/%.o,$^)

%.o: $(SRC)/%.c
	$(NVCC) -x cu -I. -dc $< -o $(OBJ)/$@

%.o: $(SRC)/%.cu
	$(NVCC) -x cu -I. -dc $< -o $(OBJ)/$@

.PHONY: clean
clean:
	rm -f $(OBJ)/*
	rm -f $(BIN)/*
