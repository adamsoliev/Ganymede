#include <fstream>
#include <vector>

#include "VCPU.h"
#include "verilated.h"

void tick(VCPU *tb);

int main(int argc, char **argv) {
        Verilated::commandArgs(argc, argv);
        VCPU *tb = new VCPU;
        tb->rst == 0;
        for (int i = 0; i < 10; i++) {
                tick(tb);
                tb->rst = 1;
        }
        // double cycles = (double)tb->cycles;
        // double instructions = (double)tb->instructions;
        // printf("%-*s run at %.2f CPI\n", 10, test.c_str(), cycles / instructions);
        delete tb;

        return 0;
}

void tick(VCPU *tb) {
        tb->eval();
        tb->clk = 1;
        tb->eval();
        tb->clk = 0;
        tb->eval();
}