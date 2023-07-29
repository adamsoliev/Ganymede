#!/usr/bin/python3

import sys
import subprocess

all_tests_pass = True

def execute_command(name):
    command = f"./../../build/ganymede -f ./{name}.c -o ./{name}.temp.output"
    subprocess.run(command, shell=True)

def cleanup(name):
    command = f"rm ./{name}.temp.output"
    subprocess.run(command, shell=True)

def compare_files(name):
    global all_tests_pass
    cf = f"./{name}.c"
    outputf = f"./{name}.output"
    tempoutputf = f"./{name}.temp.output"

    # Read the contents of the files into variables
    with open(outputf, "r") as file:
        outputf_content = file.read().strip().replace(" ", "").replace("\n", "").replace("\t", "")
    with open(tempoutputf, "r") as file:
        tempoutputf_content = file.read().strip().replace(" ", "").replace("\n", "").replace("\t", "")
    
    if not tempoutputf_content or not outputf_content: 
        print("Empty file(s)")
        all_tests_pass = False
        return

    # Compare the stripped contents of the files
    if outputf_content != tempoutputf_content:
        print(f"--------------------- {cf} ---------------------")
        all_tests_pass = False
        subprocess.run(f"diff -y {outputf} {tempoutputf}", shell=True)

def main():
    names = ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010"]
    for name in names:
        execute_command(name)
        compare_files(name)
        cleanup(name)
        if all_tests_pass: 
            print(f"{name}.c                        [OK]")

    return int(not all_tests_pass)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)