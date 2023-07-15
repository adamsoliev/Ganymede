
SRC_FILE=$1

clang-15 -S $1.c --target=riscv64 -march=rv64gc

