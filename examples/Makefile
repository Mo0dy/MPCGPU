# Makefile

# Compiler and compiler flags
# if need to debug with compute-sanitizer, use -G option
NVCC = nvcc
CFLAGS = --compiler-options -Wall  -O3 -I../include -I../include/common -I../GLASS  -I../GBD-PCG/include

examples: qp.exe

qp.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=100 -DSTATE_SIZE=10 qp_solve.cu -o qp.exe

clean:
	rm -f qp.exe
