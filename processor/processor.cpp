#include <fstream>
#include <vector>

#include "Vprocessor.h"
#include "verilated.h"

void tick(Vprocessor *tb);

int main(int argc, char **argv) {
        std::string srcFilePath = "./tests/rv64ui-p-";
        std::vector<std::string> tests = {
                // clang-format off
                "add",  "bge",  "lb",  "ma_data", "slli",  "srai", "subw",
                "addi",  "bgeu",  "lbu",  "or",  "slliw",  "sraiw",  "sw",
                "addiw",  "blt",  "ld",  "ori",  "sllw",  "sraw ",  "xor",
                "addw",  "bltu",  "lh",  "sb",  "slt",  "srl",  "xori"
                "and",  "bne",  "lhu",  "sd",  "slti",  "srli",
                "andi",  "fence_i",  "lui",  "sh",  "sltiu",  "srliw",
                "auipc",  "jal",  "lw",  "simple",  "sltu",  "srlw",
                "beq",  "jalr",  "lwu",  "sll",  "sra",  "sub",
                };
        // clang-format on

        for (std::string test : tests) {
                std::string fileName = srcFilePath + test;

                Verilated::commandArgs(argc, argv);
                Vprocessor *tb = new Vprocessor;

                // copy file
                std::ifstream src(fileName, std::ios::binary);
                std::ofstream dst("./tests/mem_data", std::ios::binary);
                dst << src.rdbuf();

                // simulate
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

void tick(Vprocessor *tb) {
        tb->eval();
        tb->clk = 1;
        tb->eval();
        tb->clk = 0;
        tb->eval();
}