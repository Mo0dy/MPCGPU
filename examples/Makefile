# Makefile

# Compiler and compiler flags
# if need to debug with compute-sanitizer, use -G option
NVCC = nvcc
CFLAGS = --compiler-options -Wall -O3 -I../include -I../include/common -I../GLASS -I../GBD-PCG/include

examples: qp.exe qp_block.exe

# make sure the KNOT_POINTS = N, STATE_SIZE = nx, CONTROL_SIZE = nu match the settings in ocpqp.m file

qp.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=10 -DSTATE_SIZE=20 -DCONTROL_SIZE=1 qp_solve.cu -o qp.exe

qp_block.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=10 -DSTATE_SIZE=20 -DCONTROL_SIZE=1 qp_block_solve.cu -o qp_block.exe

clean:
	rm -f qp.exe qp_block.exe
