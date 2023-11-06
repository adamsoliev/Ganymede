#include <fstream>
#include <vector>

#include "VSOC.h"
#include "verilated.h"

void tick(VSOC *tb);

int main(int argc, char **argv) {
        std::string isrcFilePath = "./test/mem_instr-rv64ui-p-";
        std::string dsrcFilePath = "./test/mem_data-rv64ui-p-";
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
                "ld"
                // "ld", "lw", "lwu", "lh",  "lhu", "lb", "lbu",
                // "sd", "sw", "sh", "sb", "simple", "fence_i", "ma_data", 
                };  // clang-format on

        for (std::string test : tests) {
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
                // printf("%-*s pass\n", 25, fileName.c_str());
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