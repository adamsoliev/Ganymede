#!/usr/bin/bash

SRC_FILE=$1
SYSROOT=/opt/riscv/riscv64-unknown-elf
GCCTOOLCHAIN=/opt/riscv/

#clang-15 $SRC_FILE.ll -o $SRC_FILE --target=riscv64 -march=rv64gc --sysroot=/opt/riscv/riscv64-unknown-elf --gcc-toolchain=/opt/riscv/
clang-15 $SRC_FILE.ll -o $SRC_FILE --target=riscv64 -march=rv64gc --sysroot=$SYSROOT --gcc-toolchain=$GCCTOOLCHAIN
./$SRC_FILE
rm $SRC_FILE

