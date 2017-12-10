## Experiments

### CONVOLUTION
- Our dense vs cudnn
- What is the effect of sparse kernels in terms of speedup?
alg_types: cudnn_conv,dense_imp/wPitch,dense_imp_w/o_Pitch,sparse_imp/wPitch,sparse_imp_w/o_Pitch
matrix_sizes: b11*b11,b10*b10,512*512 512*512,224*224, 128*128
kernel_sizes: 3*3, 5*5, 7*7
kernel_sparsity: 0,0.5,0.9,0.95,0.99
numberOfIterations = 1,1000

What_to_time = copy_time reduces?


### MM
alg_types: dense_mm,sparse_mm,sparse_imp
matrix1_sizes: 32*b12,32*b11,32*b10,32*b10
matrix2_sizes: b11*b11,b10*b10,512*512 512*512,224*224, 128*128
kernel_sparsity: 0,0.5,0.9,0.95,0.99
numberOfIterations = 1,1000
