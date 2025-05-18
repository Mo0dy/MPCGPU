#!/usr/bin/env sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/qdldl/build/out
./examples/pcg.exe
./examples/qdldl.exe
