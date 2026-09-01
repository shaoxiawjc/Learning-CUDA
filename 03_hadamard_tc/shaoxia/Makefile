NVCC      ?= nvcc
ARCH      ?= sm_80
NVCCFLAGS  = -arch=$(ARCH) -O3 -std=c++17 --expt-relaxed-constexpr -I. -Ifht
LDFLAGS    = -lcurand

SRCS = bench.cu hadamard.cu hadamard_tensor_core.cu hadacore.cu fht/fast_hadamard_transform_cuda.cu
OBJS = $(SRCS:.cu=.o)

bench: $(OBJS)
	$(NVCC) -arch=$(ARCH) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) bench

.PHONY: clean
