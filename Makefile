# Compiler and flags
NVCC = nvcc
CFLAGS = -O3 -arch=sm_86 -std=c++17 -rdc=true -g -lineinfo

# Include and library paths for cuBLAS
CUBLAS_INCLUDES = -I/usr/local/cuda/include
CUBLAS_LIBS = -L/usr/local/cuda/lib64 -lcublas -lcublasLt

# CUTLASS include path
CUTLASS_INCLUDES = -I/work/hdd/bche/hshah6/cutlass/include

# Kernel files
KERNELS = $(wildcard kernels/*.cuh)
CPU_KERNELS = $(wildcard cpu_kernels/*.cuh)

# Default target
all: output_verification_rand test_gpt2 test_gpt2_kernels next_token_generation verify_kv_cache

# Individual targets
.output_verification_rand: output_verification_rand
output_verification_rand: output_verification_rand.cu $(KERNELS) $(CPU_KERNELS)
	$(NVCC) $(CFLAGS) $(CUBLAS_INCLUDES) $(CUTLASS_INCLUDES) -c output_verification_rand.cu -o output_verification_rand.o
	$(NVCC) $(CFLAGS) -o output_verification_rand output_verification_rand.o $(CUBLAS_LIBS)

.test_gpt2: test_gpt2
test_gpt2: test_gpt2.cu $(KERNELS)
	$(NVCC) $(CFLAGS) $(CUBLAS_INCLUDES) $(CUTLASS_INCLUDES) -c test_gpt2.cu -o test_gpt2.o
	$(NVCC) $(CFLAGS) -o test_gpt2 test_gpt2.o $(CUBLAS_LIBS)

.test_gpt2_kernels: test_gpt2_kernels
test_gpt2_kernels: test_gpt2_kernels.cu $(KERNELS)
	$(NVCC) $(CFLAGS) $(CUBLAS_INCLUDES) $(CUTLASS_INCLUDES) -c test_gpt2_kernels.cu -o test_gpt2_kernels.o
	$(NVCC) $(CFLAGS) -o test_gpt2_kernels test_gpt2_kernels.o $(CUBLAS_LIBS)

.next_token_generation: next_token_generation
next_token_generation: next_token_generation.cu $(KERNELS)
	$(NVCC) $(CFLAGS) $(CUBLAS_INCLUDES) $(CUTLASS_INCLUDES) -c next_token_generation.cu -o next_token_generation.o
	$(NVCC) $(CFLAGS) -o next_token_generation next_token_generation.o $(CUBLAS_LIBS)

# for milestone 3
.local_attn_verify: local_attn_verify
local_attn_verify: local_attn_verify.cu $(KERNELS)
	$(NVCC) $(CFLAGS) $(CUBLAS_INCLUDES) $(CUTLASS_INCLUDES) -g -c local_attn_verify.cu -o local_attn_verify.o
	$(NVCC) $(CFLAGS) -g -o local_attn_verify local_attn_verify.o $(CUBLAS_LIBS)

.verify_kv_cache: verify_kv_cache
verify_kv_cache: verify_kv_cache.cu $(KERNELS)
	$(NVCC) $(CFLAGS) $(CUBLAS_INCLUDES) $(CUTLASS_INCLUDES) -c verify_kv_cache.cu -o verify_kv_cache.o
	$(NVCC) $(CFLAGS) -o verify_kv_cache verify_kv_cache.o $(CUBLAS_LIBS)

# Clean target
clean:
	rm -f *.o output_verification_rand test_gpt2 model_output next_token_generation local_attn_verify verify_kv_cache

.PHONY: all output_verification_rand model_output test_gpt2 test_gpt2_kernels next_token_generation local_attn_verify verify_kv_cache clean
