#include <stdio.h>
#include <stdlib.h>

#include <bitset>
#include <iostream>

#include "Vprocessor.h"
#include "verilated.h"

void tick(Vprocessor *tb);

int main(int argc, char **argv) {
        Verilated::commandArgs(argc, argv);
        Vprocessor *tb = new Vprocessor;

        for (int i = 0; i < 3000; i++) {
                tick(tb);
                if (i == 0) tb->reset == 0;
                tb->reset = 1;
        }
        return 0;
}

void tick(Vprocessor *tb) {
        tb->eval();
        tb->clk = 1;
        tb->eval();
        tb->clk = 0;
        tb->eval();
}