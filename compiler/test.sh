#!/bin/bash

assert() {
    expected="$1"
    input="$2"

    ./baikalc "$input" > tmp.s || exit

    riscv64-linux-gnu-gcc -static -o tmp tmp.s
    qemu-riscv64-static ./tmp

    actual="$?"

    if [ "$actual" = "$expected" ]; then
        echo "$input => $actual"
    else
        echo "$input => $expected expected, but got $actual"
        exit 1
    fi
}



assert 0 "int main() { return 0; }"

echo OK