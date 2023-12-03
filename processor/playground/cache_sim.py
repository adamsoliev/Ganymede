#!/usr/bin/python3


"""
Direct mapped cache
    4KiB
    32-bit address -> [tag] [index] [byte offset]
    cache size is 2^n, so n bits are used for index
    block size is 2^m words (2^(m + 2) bytes) 1,048,576 * 4 bytes => 4 KiB

"""

print("Hello World")


"""
direct mapped 
set associative
    one-way
    two-way
fully associative
"""

