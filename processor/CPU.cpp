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
        // std::vector<std::string> tests = {
        // clang-format off
                // "add", 
                // "sub", 
                // "or", 
                // "and", 
                // "add", "bge",  "slli",  "srai", "subw",
                // "addi",  "bgeu",  "or",  "slliw",  "sraiw",  
                // "addiw",  "blt",  "ori",  "sllw",  "sraw ",  "xor",
                // "addw",  "bltu",  "slt",  "srl",  "xori"
                // "and",  "bne",  "slti",  "srli",
                // "andi",  "lui",  "sltiu",  "srliw",
                // "auipc",  "jal",  "sltu",  "srlw",
                // "beq",  "jalr",  "sll",  "sra",  "sub", 
                // "ld", "lw", "lwu", "lh", "lhu", "lb", "lbu", 
                // "sd", "sw", "sh", "sb"
                // "sd", "sw", "sh", "sb", "simple", "fence_i", "ma_data", 
                // };  // clang-format on

        // for (std::string test : tests) {
                // copy files
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

                // for (int time = 0; time < 3000; time++) {
                int time = 0;
                while (!Verilated::gotFinish()) {
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
        // }
        exit(EXIT_SUCCESS);
}
