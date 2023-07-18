#!/bin/bash

# 1
echo "TESTSUITE #1 - ensure generated .s file is runnable and returns expected value"
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
assert 5 "int main() { return 2 + 3; }"
assert 2 "int main() { return 4 - 2; }"
assert 9 "int main() { return 3 * 3; }"

# 2
echo -e "\nTESTSUITE #2 - ensure generated .ll and .s files match expected files"
python3 tests.py
exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    echo "Script failed"
    exit 1
fi

echo -e "\nOK"