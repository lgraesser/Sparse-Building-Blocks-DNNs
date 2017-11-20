#!/usr/bin/env python

if __name__ == '__main__':
        import sys
        if len(sys.argv) == 3:
            dims = map(int,sys.argv[1].strip().split(','))
            if len(dims)<2:
                print("!error!please provide at least 2 dimensions!")
                sys.exit(1)
            s_factor = float(sys.argv[2])
        else:
            print("Usage: python generate_sparse_mat.py d1,d2,... sparsity_factor[0-1]")
            sys.exit(1);

        import numpy as np
        from itertools import product
        RANGE_WIDTH=10
        def savetxt_compact(f, matrix, fmt="%.6g", delimiter=','):
                for row in matrix:
                    line = delimiter.join("0" if value == 0 else fmt % value for value in row)
                    f.write(line + '\n')

        mask = np.random.binomial(1,1-s_factor,dims)
        arr = np.random.uniform(-RANGE_WIDTH,RANGE_WIDTH,dims)

        print("N_DIM=%d" % len(dims))
        for i,d in enumerate(dims):
            print("DIM%d=%d" % (i,d))

        for matrix_id in product(*(range(i) for i in dims[:-2])):
            print("MATRIX_ID=%s" % str(matrix_id))
            c_mat = arr[matrix_id]*mask[matrix_id]
            savetxt_compact(sys.stdout,c_mat,fmt="%.8f",delimiter=",")
