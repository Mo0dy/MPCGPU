# Makefile

# Compiler and compiler flags
NVCC = nvcc
CFLAGS = --compiler-options -Wall  -O3 -Iinclude -Iinclude/common -IGLASS  -IGBD-PCG/include -lcublas


examples: examples/indy.exe examples/pcg.exe

examples/indy.exe:
	$(NVCC) $(CFLAGS) examples/track_indy7_pcg.cu -o examples/indy.exe
examples/pcg.exe:
	$(NVCC) $(CFLAGS) examples/track_iiwa_pcg.cu -o examples/pcg.exe

clean:
	rm -f examples/*.exe
