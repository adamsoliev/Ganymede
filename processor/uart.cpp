#include <stdio.h>
#include <stdlib.h>
#include "Vuart.h"
#include "verilated.h"

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Vuart *tb = new Vuart;
    tb->i_clk = 0;
    for (int k = 0; k < 10000000000000000000000; k++) {
        tb->i_clk = ~tb->i_clk & 1;
        tb->eval();

        // printf("k = %2d, ", k);
        printf("clk = %3d, ", tb->i_clk);
        printf("led = %3d\n", tb->o_led);
    }

    return 0;
}