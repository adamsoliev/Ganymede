#include "baikalc.h"

int main(int argc, char **argv) {
    struct Token *token = b_scan(argv[1]);
    // print(token);
    struct decl *program = parse(token);
    // print_decl(program, 0);

    semantic_analysis(program);
    // generateIR(program);

    FILE *out = stdout;

    fprintf(out, "  .globl main\n");
    fprintf(out, "main:\n");
    // prologue
    fprintf(out, "  addi sp, sp, 32\n");
    // fprintf(out,"  sd ra, 16(sp)\n");
    fprintf(out, "  sd s0, 24(sp)\n");
    fprintf(out, "  addi s0, sp, 32\n");
    fprintf(out, "j .L.return.main\n");

    // epilogue
    fprintf(out, ".L.return.main:\n");
    fprintf(out, "  li a5, 0\n");  // return 0
    fprintf(out, "  mv a0,a5\n");
    // fprintf(out,"  ld ra, 16(sp)\n");
    fprintf(out, "  ld s0,24(sp)\n");
    fprintf(out, "  addi sp, sp, 32\n");
    fprintf(out, "  jr ra\n");
    return 0;
}