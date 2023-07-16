
#include "baikalc.h"

static FILE* out;

static void prologue() {
    //
}

static void epilogue() {
    //
}

void codegen(struct decl* program) {
    out = stdout;

    fprintf(out, "  .file   \"0001.c\"\n");
    fprintf(out, "  .globl  main\n");
    fprintf(out, "  .type   main,@function\n");
    fprintf(out, "main:\n");

    prologue();

    fprintf(out, "  li a0, 2\n");  // return 2
    fprintf(out, "  ret\n");

    epilogue();
}