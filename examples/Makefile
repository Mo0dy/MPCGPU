# Makefile

# Compiler and compiler flags
# if need to debug with compute-sanitizer, use -G option
NVCC = nvcc
CFLAGS = --compiler-options -Wall -O3 -I../include -I../external/GLASS -I../external/GBD-PCG/include -lcublas -G -g

test: multi.exe

multi.exe: test_mpc_n.cu
	$(NVCC) $(CFLAGS) test_mpc_n.cu -o multi.exe

clean:
	rm -f *.exe
