#!/bin/bash

# 1
echo "TESTSUITE #1"
assert() {
    expected="$(( ($1 % 256 + 256) % 256 ))"
    input="$2"

    # ./build/baikalc "$input" > ./build/tmp.s || exit
    ./build/baikalc "$input" || exit

    # riscv64-linux-gnu-gcc -static -o ./build/tmp ./build/tmp.s
    riscv64-linux-gnu-gcc -static -o ./build/tmp ./build/tmp.s
    qemu-riscv64-static ./build/tmp

    actual="$?"

    if [ "$actual" = "$expected" ]; then
        echo "$input => $actual"
    else
        echo "$input => $expected expected, but got $actual"
        exit 1
    fi
}

assert 0 "int main() { return 0; }"
assert 2 "int main() { return 2; }"
assert -2 "int main() { return -2; }"

# 2
echo 
echo "TESTSUITE #2"

python3 tests.py
exit_code=$?
if [[ $exit_code -eq 0 ]]; then
    echo "Script succeeded"
else
    echo "Script failed"
    exit 1
fi

echo OK