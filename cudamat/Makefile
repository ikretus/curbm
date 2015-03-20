all: libcudamat.so

libcudamat.so: cudamat.cu cudamat_kernels.cu cudamat.cuh cudamat_kernels.cuh
	nvcc -O3 -v \
		-gencode=arch=compute_11,code=sm_11 \
		-gencode=arch=compute_20,code=sm_20 \
		-gencode=arch=compute_30,code=sm_30 \
		-gencode=arch=compute_32,code=sm_32 \
		-gencode=arch=compute_35,code=sm_35 \
		-gencode=arch=compute_37,code=sm_37 \
		-gencode=arch=compute_50,code=sm_50 \
		-gencode=arch=compute_50,code=compute_50 \
		--compiler-options '-fPIC' -o libcudamat.so \
		--shared cudamat.cu cudamat_kernels.cu -lcublas

clean:
	rm -rf *.pyc *.so
