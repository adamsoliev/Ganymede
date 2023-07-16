#!/usr/bin/bash

SRC_FILE=$1
SYSROOT=/opt/riscv/riscv64-unknown-elf
GCCTOOLCHAIN=/opt/riscv/

# Generate IR
clang-15 -O3 -Xclang -disable-llvm-passes -S -emit-llvm $SRC_FILE.c -o $SRC_FILE.ll --target=riscv64 -march=rv64gc --sysroot=$SYSROOT --gcc-toolchain=$GCCTOOLCHAIN 

# Strip metadata
opt-15 -mem2reg -S $SRC_FILE.ll -o $SRC_FILE-mem2reg.ll
rm $SRC_FILE.ll
mv $SRC_FILE-mem2reg.ll $SRC_FILE.ll

# Generate assembly
clang-15 -S $1.ll --target=riscv64 -march=rv64gc

