#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "VCPU.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

int main(int argc, char **argv) {
        if (argc < 2) {
                printf("Test name missing\n");
                exit(EXIT_FAILURE);
        }
        std::string test = argv[1];
        std::string isrcFilePath = "./test/mem_instr-rv64ui-p-";
        std::string dsrcFilePath = "./test/mem_data-rv64ui-p-";

        std::string ifileName = isrcFilePath + test;
        std::ifstream isrc(ifileName, std::ios::binary);
        std::ofstream idst("./test/mem_instr", std::ios::binary);
        idst << isrc.rdbuf();
        idst << '\n';
        idst.close();
        isrc.close();

        std::string dfileName = dsrcFilePath + test;
        std::ifstream dsrc(dfileName, std::ios::binary);
        std::ofstream ddst("./test/mem_data", std::ios::binary);
        ddst << dsrc.rdbuf();
        ddst << '\n';
        ddst.close();
        dsrc.close();

        Verilated::commandArgs(argc, argv);

        VCPU *tb = new VCPU;
        Verilated::traceEverOn(true);

        VerilatedVcdC *tfp = new VerilatedVcdC;
        tb->trace(tfp, 99);
        tfp->open("CPUtrace.vcd");

        tb->rst_i = 1;

        int time = 0;
        while (!Verilated::gotFinish() && time < 3000) {
                tb->rst_i = 0;
                tb->clk_i ^= 1;
                tb->eval();
                tfp->dump(time);
                tfp->flush();
                time++;
        }
        tfp->close();
        delete tb;
        delete tfp;
        exit(EXIT_SUCCESS);
}
