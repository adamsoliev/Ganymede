#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <vector>

#include "VCPU.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

int main(int argc, char **argv) {
        std::string isrcFilePath = "./test/no_hazard-";
        std::vector<std::string> tests = {"add"};

        for (std::string test : tests) {
                std::string ifileName = isrcFilePath + test;
                std::ifstream isrc(ifileName, std::ios::binary);
                std::ofstream idst("./test/mem_instr", std::ios::binary);
                idst << isrc.rdbuf();
                idst << '\n';
                idst.close();
                isrc.close();

                Verilated::commandArgs(argc, argv);

                VCPU *tb = new VCPU;
                Verilated::traceEverOn(true);

                VerilatedVcdC *tfp = new VerilatedVcdC;
                tb->trace(tfp, 99);
                tfp->open("CPUtrace.vcd");

                tb->rst_i = 1;

                for (int time = 0; time < 50; time++) {
                        if (Verilated::gotFinish()) {
                                printf("FAILED TEST");
                                exit(EXIT_FAILURE);
                        }
                        tb->rst_i = 0;
                        tb->clk_i ^= 1;
                        tb->eval();
                        tfp->dump(time);
                        tfp->flush();
                }
        }
        exit(EXIT_SUCCESS);
}
