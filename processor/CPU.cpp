#include <stdio.h>
#include <stdlib.h>

#include "VCPU.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

int main(int argc, char **argv) {
        Verilated::commandArgs(argc, argv);

        VCPU *tb = new VCPU;
        Verilated::traceEverOn(true);

        VerilatedVcdC *tfp = new VerilatedVcdC;
        tb->trace(tfp, 99);
        tfp->open("CPUtrace.vcd");

        tb->rst_i = 1;

        for (int time = 0; time < 20; time++) {
                tb->rst_i = 0;
                tb->clk_i ^= 1;
                tb->eval();
                tfp->dump(time);
                tfp->flush();
        }
}
