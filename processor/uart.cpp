#include <stdio.h>
#include <stdlib.h>
#include <bitset>
#include <iostream>
#include "Vuart.h"
#include "verilated.h"

void tick(Vuart *tb);

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Vuart *tb = new Vuart;

    char input = 'Z';
    int output = 0;
    for (int i = 0; i < 12; i++) {
        tick(tb);
        if (i == 0) {
            tb->wr = 1;
            tb->data = input;
        } else {
            if (i > 1 && i < 10) {
                output = ((tb->tx & 1) ? 0x80 : 0) | (output >> 1);
            }
            tb->wr = 0;
        }
    }
    printf("input char:  %c\n", input);
    printf("output char: %c\n", output);
    return 0;
}

void tick(Vuart *tb) {
    tb->eval();
    tb->clk = 1;
    tb->eval();
    tb->clk = 0;
    tb->eval();
}
