#!/usr/bin/bash

SRC_FILE=$1
SYSROOT=/opt/riscv/riscv64-unknown-elf
GCCTOOLCHAIN=/opt/riscv/

clang-15 -S $SRC_FILE.c --target=riscv64 -march=rv64gc --sysroot=$SYSROOT --gcc-toolchain=$GCCTOOLCHAIN

