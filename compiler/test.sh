#!/bin/bash

echo "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
echo "             RUNNING SCANNER TESTS              "
echo "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
cd ./tests/scanner # dump way to do this
python3 runner.py
exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    echo "Script failed"
    exit 1
fi
cd ../../

# echo -e "\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
# echo "             RUNNING PARSER TESTS               "
# echo "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"
# cd ./tests/parser # dump way to do this
# python3 runner.py
# exit_code=$?
# if [[ $exit_code -ne 0 ]]; then
#     echo "Script failed"
#     exit 1
# fi
# cd ../../


# echo "TESTSUITE #1 - ensure generated .s file is runnable and returns expected value"
assert() {
    expected="$(( ($1 % 256 + 256) % 256 ))"
    input="$2"

    # ./build/ganymede "$input" > ./build/tmp.s || exit
    ./build/ganymede "-s" "$input" -o ./build/tmp.s || exit

    # riscv64-linux-gnu-gcc -static -o ./build/tmp ./build/tmp.s
    riscv64-linux-gnu-gcc -static -o ./build/tmp ./build/tmp.s
    qemu-riscv64-static ./build/tmp

    actual="$?"

    if [ "$actual" = "$expected" ]; then
        echo "$input => $actual" | paste -s -d ' '
    else
        echo "$input => $expected expected, but got $actual"
        exit 1
    fi
}

# assert 2 "$(cat tests/parser/0001.c)"
# assert -2 "$(cat tests/parser/0002.c)"
# assert 4 "$(cat tests/parser/0003.c)"
# assert 4 "$(cat tests/parser/0004.c)"
# assert 1 "$(cat tests/parser/0005.c)"
# assert 1 "$(cat tests/parser/0007.c)"
# assert 0 "$(cat tests/parser/0008.c)"

# assert 0 "int main() { return 0; }"
# assert 2 "int main() { return 2; }"
# assert -2 "int main() { return -2; }"
# assert 5 "int main() { return 2 + 3; }"
# assert 2 "int main() { return 4 - 2; }"
# assert 9 "int main() { return 3 * 3; }"
# assert 3 "int main() { return 9 / 3; }"

# assert 16 "int main() { return 2 + 3 + 3 + 3 + 3 + 2; }"
# assert 6 "int main() { return 20 - 3 - 3 - 3 - 3 - 2; }"
# assert 81 "int main() { return 3 * 3 * 3 * 3; }"
# assert 3 "int main() { return 81 / 3 / 3 / 3; }"

# assert 21 "int main() { return 4 + 3 * 3 * 3 - 10; }"
# assert 26 "int main() { return 4 + 3 * 3 * 3 - 10 / 2; }"

# assert 23 "int main() { int a = 23; return a; }"

# 2
# echo -e "\nTESTSUITE #2 - ensure generated .ll and .s files match expected files"
# python3 tests.py
# exit_code=$?
# if [[ $exit_code -ne 0 ]]; then
#     echo "Script failed"
#     exit 1
# fi


echo -e "\nOK"