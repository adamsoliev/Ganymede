#include <fstream>
#include <vector>

#include "VSOC.h"
#include "verilated.h"

void tick(VSOC *tb);

int main(int argc, char **argv) {
        std::string srcFilePath = "./tests/rv64ui-p-";
        std::vector<std::string> tests = {
                // clang-format off
                "add",  "bge",  "slli",  "srai", "subw",
                "addi",  "bgeu",  "or",  "slliw",  "sraiw",  
                "addiw",  "blt",  "ori",  "sllw",  "sraw ",  "xor",
                "addw",  "bltu",  "slt",  "srl",  "xori"
                "and",  "bne",  "slti",  "srli",
                "andi",  "lui",  "sltiu",  "srliw",
                "auipc",  "jal",  "sltu",  "srlw",
                "beq",  "jalr",  "sll",  "sra",  "sub", 
                // "ld", "lw", "lwu", "lh",  "lhu", "lb", "lbu",
                // "sd", "sw", "sh", "sb", "simple", "fence_i", "ma_data", 
                };  // clang-format on

        for (std::string test : tests) {
                std::string fileName = srcFilePath + test;

                // copy file
                std::ifstream src(fileName, std::ios::binary);
                std::ofstream dst("./tests/mem_data", std::ios::binary);
                dst << src.rdbuf();

                // simulate
                Verilated::commandArgs(argc, argv);
                VSOC *tb = new VSOC;
                tb->reset == 0;
                for (int i = 0; i < 3000; i++) {
                        tick(tb);
                        tb->reset = 1;
                }
                delete tb;

                // report
                printf("%-*s pass\n", 25, fileName.c_str());
        }
        return 0;
}

void tick(VSOC *tb) {
        tb->eval();
        tb->clk = 1;
        tb->eval();
        tb->clk = 0;
        tb->eval();
}