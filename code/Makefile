NVCC=nvcc -arch=sm_35
SRC = src
BIN = bin
OBJ = obj
TARGETS = mm io sparse_test conv_test

all: $(patsubst %,$(BIN)/%,$(TARGETS))

# the first two are the patterns the rest is the list of strings, e.g. x.o->obj/x.o
$(BIN)/mm: $(patsubst %.o,$(OBJ)/%.o,matrix_io.o mm_test.o mm.o sparse_conversion.o)
	$(NVCC) -lcublas -lcusparse -o $@ $^

$(BIN)/sparse_test: $(patsubst %.o,$(OBJ)/%.o,sparse_conversion_test.o sparse_conversion.o matrix_io.o)
	$(NVCC) -lcusparse -o $@ $^

$(BIN)/conv_test: $(patsubst %.o,$(OBJ)/%.o,conv_test.o conv.o sparse_conversion.o matrix_io.o)
	$(NVCC) -lcusparse -lcudnn -o $@ $^

$(BIN)/io: $(patsubst %.o,$(OBJ)/%.o,matrix_io.o matrix_io_test.o)
	$(NVCC) -o $@ $^

$(OBJ)/%.o: $(SRC)/%.c
	$(NVCC) -x cu -I. -dc $< -o $@

$(OBJ)/%.o: $(SRC)/%.cu
	$(NVCC) -x cu -I. -dc $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJ)/*
	rm -f $(BIN)/*
