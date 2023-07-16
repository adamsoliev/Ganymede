#include "baikalc.h"

int main(int argc, char **argv) {
    struct Token *token = b_scan(argv[1]);
    // print(token);
    struct decl *program = parse(token);
    // print_decl(program, 0);

    semantic_analysis(program);
    irgen(program);
    codegen(program);

    return 0;
}