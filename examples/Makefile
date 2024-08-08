# Makefile

# Compiler and compiler flags
# if need to debug with compute-sanitizer, use -G option
NVCC = nvcc
CFLAGS = --compiler-options -Wall -O3 -I../include -I../include/common -I../GLASS -I../GBD-PCG/include

examples: qp_trans_ldl.exe qp_trans_chol.exe

# make sure the KNOT_POINTS = N, STATE_SIZE = nx, CONTROL_SIZE = nu match the settings in ocpqp.m file

qp_trans_ldl.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=10 -DSTATE_SIZE=20 -DCONTROL_SIZE=1 -DCHOL_OR_LDL=false qp_block_solve.cu -o qp_trans_ldl.exe

qp_trans_chol.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=10 -DSTATE_SIZE=20 -DCONTROL_SIZE=1 -DCHOL_OR_LDL=true qp_block_solve.cu -o qp_trans_chol.exe

clean:
	rm -f qp_trans_ldl.exe qp_trans_chol.exe
