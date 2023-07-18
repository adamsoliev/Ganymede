#!/usr/bin/python3

import sys
import subprocess

all_tests_pass = True

def execute_command(name):
    # command = f"./build/baikalc ./build/{name}.c"
    with open(f"./tests/{name}.c", "r") as file:
        content = file.read()
    if content: 
        subprocess.run(f"./build/baikalc '{content}'", shell=True)

def compare_files(name):
    ll_file = f"./tests/{name}.ll"
    s_file = f"./tests/{name}.s"
    tmp_ll_file = "./build/ir.ll"
    tmp_s_file = "./build/tmp.s"

    # Read the contents of the files into variables
    with open(ll_file, "r") as file:
        ll_content = file.read().strip().replace(" ", "").replace("\n", "").replace("\t", "")
    with open(tmp_ll_file, "r") as file:
        tmp_ll_content = file.read().strip().replace(" ", "").replace("\n", "").replace("\t", "")
    with open(s_file, "r") as file:
        s_content = file.read().strip().replace(" ", "").replace("\n", "").replace("\t", "")
    with open(tmp_s_file, "r") as file:
        tmp_s_content = file.read().strip().replace(" ", "").replace("\n", "").replace("\t", "")

    # Compare the stripped contents of the files
    if ll_content != tmp_ll_content:
        print(f"--------------------- {ll_file} ---------------------")
        all_tests_pass = False
        subprocess.run(f"diff -y {ll_file} {tmp_ll_file}", shell=True)

    if s_content != tmp_s_content:
        print(f"--------------------- {s_file} ---------------------")
        all_tests_pass = False
        subprocess.run(f"diff -y {s_file} {tmp_s_file}", shell=True)

def main():
    names = ["0001", "0002", "0003", "0004", "0005", "0006"]
    for name in names:
        print(f"{name}.c")
        execute_command(name)
        compare_files(name)
    return int(not all_tests_pass)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)