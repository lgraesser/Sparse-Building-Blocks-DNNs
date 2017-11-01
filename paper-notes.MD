## GPU Project: Sparse Matrix - Vector Multiplication Paper notes

### Thoughts:

Ways to categorize matrices
- Statistics of non-zero rows - mean, mode, std dev
- Sample blocks, count number of non-zero elements, could also calculate statistics of the block e.g. mean, mode, std-dev
- Is the matrix diagonal?
- To check - do the papers which use models to choose the format take into account the conversion time? Can the number of iterations be parameterized? Maybe doesn't need to be since will be captured in total execution time

### Survey

"Any matrix with enough zeros that it pays to take advantage of them"
James Wilkinson

NZ = non zero coefficients
M x N matrix

NZ << M x N

- Problem with the usual factorization methods found in LAPACK etc. is that the normally destroy the sparsity of a matrix by introducing fill-in --> the matrices become unmanageable
-  Sparse Matrix-Vector multiplication (SpMV) is a memory bound problem
- History of its efficient implementation is mostly a story of data structures and of their match to the architecture of the computers employed to run the iterative solver codes
- NVIDIA cuSPARSE library (2015), CUSP (2016)
- Problem arises through mismatch between SIMD GPGPU architecture and the irregular data access patterns of many sparse matrices
- This leads to the development of data structures so that the access patterns are more regular ie. this is a story of storage formats

#### Section 2: Storage formats for sparse matrices
- Standard format for matrix representation: one linear array and two integer values (rows, cols)
- Sparse matrices - need an auxiliary index to avoid storing all of the elements to map between index and location in memory
- Cost of rebuilding the map is THE issue
- Problem: sparse storage means that coefficients stored in adjacent positions in the sparse matrix may operate on vector entries that are quite far apart

" Multiple factors contribute to determine the overall performance
- The match between the data structure and the underlying computing architecture, including the possibility of exploiting special hardware instructions
- The suitability of the data structure to decomposition into independent, load balanced work units
- The amount of overhead due to the explicit storage of indices
- The amount of padding with explicit zeros that may be necessary
- The interaction between the data structure and the distribution of nonzeros (pattern) within the sparse matrix
- The relation between the sparsity pattern and the sequence of memory accesses especially into the x vector"

Formats - three basic, widely used.
1. COOrdinate (COO)
2. Compressed Sparse Rows (CSR)
3. Compressed Sparse Columns (CSC)

1. COO
Requires three arrays. One for storing the non zero elements of the matrix, one for storing the row index, one for the column index
- Five memory reads, one memory write and two floating point ops per non zero coefficient

2. CSR
Requires three arrays. One for storing the non zero elements of the matrix, one for storing the column index, one for marking the boundaries of each row.
- Three memory reads, and two floating point ops per non zero coefficient

3. CSC
V. similar to (2) except  the matrix values are first grouped by column and a row index is stored and column boundaries are marked

More formats
4. ELLPACK?ITPACK - two 2D arrays with M rows and MAXNZR columns (i.e. max number of nonzeros in any row). One array contains the non zero elements, the other the column indices. Rows shorted than MAXNZR are padded with zeros and appropriate column indices
- One memory write per outer op, three memory reads and two FP ops per inner op
- Overhead: memory and redundant ops. Acceptable if the max number of non zeros per row is not much larger than average and the regularity of the data structure allows for faster code.

5. JAD format
- Variant of ELLPACK: sort matrix by number of non zero rows and block

6. DIA
- 2D array containing in each column the coefficients along a diagonal of the matrix, and an integer array offset that determines where each diagonal starts. No indirect addressing required

#### Section 3: GPGPUS

"The main optimization issue to support a GPGPU target then revolves around how an algorithm should be implemented to take advantage of the full throughput of the device. To make good use of the memory access features of the architecture, we need to maximize the regularity of memory accesses to ensure coalesced accesses. In the SpMV context, the ELLPACK format entails a regular access pattern to read the sparse matrix values and the y input vector, provided that we choose a memory layout for arrays ensuring their alignment to the appropriate boundaries in memory. The access pattern is easily exploited by assigning each row of the matrix to one thread; threads in a warp will work on a block of contiguous rows."

Section 4: A survey of sparse matrix formats on GPGPUs

Three main research directions for GPGPUs, not mutually exclusive
1. Applying novel sparse matrix formats, typically derived from classic ones
2. Applying architecture-specific optimizations to existing formats
3. Applying automated performance tuning of the matrix format and
parameters

Auto-tuning methods - determine some parameters to decide on optimal storage scheme
Hybrid approaches are also being explored - i.e. using multiple formats depending on the sparsity pattern

Novel sparse matrix formats:
- COO variants: COO often has bad memory footprint, and can't know in advance which set of coefficients will be used to calculate a single element of the resulting vector
	- ALIGNED_COO (not available)
	- Sliced COO - decomposing matrix into slices based on sparsity patterns (available)
	- Lossless compression of index data (BRO_COO)
	- Blocking ? (BCCOO)
- CSR Variants: problems caused by lack of coalescing, load imbalance, and thread divergence
	- Bell and Garland - explore assigning different number of threads per row to improve memory access patterns
	- Led to work determining optimal number of threads per row based on number of non zero elements per row (e.g. half or full warp)
	- Guo and Gropp - simple auto-tuning: Sort rows in increasing order of nonzero elements per row and partition into several ranges, then assign a given number of threads for different ranges of the matrix rows so as to balance the threads workload.
	- Baxter - two step approach - fast but complex to implement and modify
	- CSR stream  / CSR adaptive
	- Other formats: The Compressed Sparse Row with Segmented Interleave Combination (SIC), Row-grouped CSR (RgCSR) - requires block level sync, BIN-CSR - partitions matrix int bins, bin being a portion of the matrix that is accessed concurrently by a group of threads, and all rows in each bin should have the same length (so shorted rows are padded), Compressed Multi-Row Storage (CMRS), prefetch compressed row storage (PCSR), BCSR - block storage of 2D small dense blocks
-CSC Variants are bad for this, and there is little work on it
- ELLPACK Variants: most effective for GPGPUs. Efficient for matrices with regular or partially regular sparsity patterns ie. with little variation in the number of non zero elements per row.
	- ELLPACK-R: stores number of non zero elements per row to avoid wasted computation of padded zeros (reference implementation)
	- SELL / SELL-C : generalization of ELLPACK-R. Re-order the rows and partition into several slices of similar lengths (slice = set of adjacent rows). Each slice packed into ELLPACK-R format. Slice = 1 --> CSR. Slice = matrix --> ELLPACK
	- Permuting of rows and / or columns to guarantee coalesce memory access
	- SELL-C-sigma - sort rows before slicing
	- BELLPARK - blocked ELLPACK. Sort rows in decreasing number of nonzero elements, then split into blocks - and each block stored in ELLPACK format. May end up with dense blocks. Block size needs to be tuned. Only leads to performance improvements on matrices with small dense block substructures
	- BSELLPACK
	- General problem is the pre-processing required with these algorithms
	- Adaptive ELL - ELLPACK based format. Each warp can process one to multiple rows depending on the sparsity pattern
 - JAD: requires preprocessing the matrix.
	 - ELLPACK-RP has been proposed but only tested on a limited set of matrices
	 - TJAD: significant performance boost for matrices that are neither highly uniform nor highly nonuniform in the number of nonzeros in the rows
-DIA: suitable for matrices with a natural diagonal format. No indirection but can waste storage and computational resources
-Hybrids:
	- COO + ELLPACK: allocates first K nonzero per row to ELLPACK and rest to COO. At least 1/3 of the matrix rows should contain K or more elements
	- CSR + ELLPACK: rows with number of non zero elements > warp size are stored in CSR format, rest in ELLPACK
	- Matam and Khotapalli - CSR + ELLPACK, analyze the sparse matrix structure and choose right data structure to represent it
	- Cocktail Format - Su and Keutzer (2012) - partitions the input sparse matrix into several sub matrices, each with a given format
	- BCOO + BCSR + ELL

Automated tuning and Performance Optimization
- Choi et all 2010 - auto tunes based on matrix dependent params
- SMAT - provide data in CSR format and SMAT automatically determines the optimal storage format and implementation on a given architecture. Use ML
- ML approach using classification trees to select amongst CSR, ELLPACK, COO, and ELL-COO (Sedaghati et al, 2015
- Probabilistic method for selecting target format (Li et al, 2015)

Note: you need to take into account conversion overhead for a format
Assume a matrix provided in COO format

Conversion approaches
Conversion to CSR from an already sorted COO is extremely fast, since it only involves counting the entries in each row and applying a scan primitive to compute the offsets.
- Therefore, CSR conversion is the baseline for device side, since it involves very little time beyond the absolute minimum needed to transfer the coefficient data from host to device memory across the PCI bus.
- Conversion to HYB is implemented by having a CSR format on the host, copying its data to the device, invoking the cuSPARSE conversion to HYB, and then releasing the temporary CSR data.
- For both ELL and HLL, we have implemented the default CPU-side conversion as well as methods split between the CPU and the GPU; the latter are the ones whose times are reported in the tables. Some preprocessing (essentially, counting offsets and maximum row occupancy) is performed on the CPU, and then the coefficients and auxiliary data are copied onto the GPU, where a CUDA kernel uses them to allocate and populate the ELL/HLL data structure.
- The HDI conversion at this time is only implemented on the host; the resulting data structure is then copied on the device side.

See paper for number of iterations required for breakeven point

#### Section 5: Lessons learned

The performance data discussed in this section allow us to state some rules of thumb:
- It is important to consider the conversion overhead: if the matrix (structure) is only used for a few products, it does not pay off to search for sophisticated data structures; CSR suffices. Having the same structure but different coefficients may call for a specialized coefficient update functionality, but the payoff is highly application dependent.
- It is possible to apply a global renumbering of the equations to minimize the matrix bandwidth; this may significantly affect the cache access patterns, but we do not discuss the issue in this article for space reasons.
- Reordering only the rows of the matrix, as necessary in some formats (e.g., JAD), also entails an overhead because we have to reshuffle, at some point, either the output or the input vectors.
- The memory footprint can be a big problem, even for some formats like HLL; however:
- If the matrix comes from a PDE discretization, one should certainly try an ELLPACKlike format first, depending on the memory available, unless:
- If the matrix not only comes from a PDE but also has a piecewise diagonal structure, then HDI is probably a very good idea, and
- If the matrix comes from a graph problem and has a power-law distribution in the number of entries, then the only option among the ones considered here is HYB, but even HYB is not very satisfactory; in this case, it may be appropriate to investigate more specialized formats [Ashari et al. 2014; Yang et al. 2011].

### Parallel Sparse Direct Methods: A short tutorial
https://pdfs.semanticscholar.org/bfbe/882ec19e2fae60fdbab805e49569cd292085.pdf
- TO READ
- Utku 

### A lightweight optimization selection method for Sparse Matrix-Vector Multiplication, Jan 2016
https://arxiv.org/pdf/1511.02494.pdf
- TO READ
- Laura

### Accelerating Sparse Matrix Vector Multiplication in Iterative Methods Using GPU
http://ieeexplore.ieee.org/document/6047229/
-TO READ:  Contains some work analyzing sparse matrix structure
- Utku

### SMAT: An input adaptive auto-tuner for sparse matrix-vector
multiplication, Li et al, 2013
https://arxiv.org/abs/1210.2536
- TO READ - uses ML to select compression format
- Laura

###  A model-driven blocking strategy for load balanced sparse matrix-vector multiplication on GPUs, Sedaghati et al, 2015
https://ac.els-cdn.com/S0743731514002081/1-s2.0-S0743731514002081-main.pdf?_tid=1a8fcc74-bdd2-11e7-9796-00000aacb35d&acdnat=1509409754_e68b4a6351498b88c6b95658a7dbe1cb
- TO READ:  ML approach using classification trees to select amongst CSR, ELLPACK, COO, and ELL-COO
- Utku

### Automatic selection of sparse matrix representation on GPUs,  Sedaghati et al, 2015
http://web.cse.ohio-state.edu/~pouchet.2/doc/ics-article.15b.pdf
- TO READ
- Laura

### Graph Clustering survey
http://www.leonidzhukov.net/hse/2015/networks/papers/GraphClustering_Schaeffer07.pdf
- MAYBE TO READ
- Utku?
