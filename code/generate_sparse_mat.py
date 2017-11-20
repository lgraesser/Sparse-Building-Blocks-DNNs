#!/usr/bin/env python

if __name__ == '__main__':
        import sys
        if len(sys.argv) == 4:
            n,m = map(int,sys.argv[1:3])
            s_factor = float(sys.argv[3])
        else:
            print("Usage: python generate_sparse_mat.py N_rows M_columns sparsity_factor[0-1]")
            sys.exit(1);

        import numpy as np
        RANGE_WIDTH=10
        def savetxt_compact(f, matrix, fmt="%.6g", delimiter=','):
                for row in matrix:
                    line = delimiter.join("0" if value == 0 else fmt % value for value in row)
                    f.write(line + '\n')

        mask = np.random.binomial(1,1-s_factor,(n,m))
        arr = np.random.uniform(-RANGE_WIDTH,RANGE_WIDTH,(n,m))

        print("ROWS=%d" % n)
        print("COLUMNS=%d" % m)
        non_zeros = sum(sum(mask))
        print("SPARSITY=%.2f (%d/%d)" % (1-float(non_zeros)/n/m,non_zeros,n*m))

        savetxt_compact(sys.stdout,arr*mask,fmt="%.8f",delimiter=",")
