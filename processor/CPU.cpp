#include <stdio.h>
#include <stdlib.h>

#include "VCPU.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

void tick(int tickcount, VCPU *tb, VerilatedVcdC *tfp) {
        tb->eval();
        if (tfp) tfp->dump(tickcount * 10 - 2);
        tb->clk_i = 1;
        tb->eval();
        if (tfp) tfp->dump(tickcount * 10);
        tb->clk_i = 0;
        tb->eval();
        if (tfp) {
                tfp->dump(tickcount * 10 + 5);
                tfp->flush();
        }
}

int main(int argc, char **argv) {
        unsigned tickcount = 0;

        // Call commandArgs first!
        Verilated::commandArgs(argc, argv);

        // Instantiate our design
        VCPU *tb = new VCPU;

        // Generate a trace
        Verilated::traceEverOn(true);
        VerilatedVcdC *tfp = new VerilatedVcdC;
        tb->trace(tfp, 00);
        tfp->open("CPUtrace.vcd");

        for (int k = 0; k < 200; k++) {
                tick(++tickcount, tb, tfp);
        }
}
