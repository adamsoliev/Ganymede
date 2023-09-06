#include "ganymede.h"

// -------------- ASSUME UNLIMITED REGISTERS

// function
static void function(struct ExtDecl *func) {
        fprintf(outfile, "  .globl %s\n", func->decltor->name);
        fprintf(outfile, "%s:\n", func->decltor->name);

        // prologue
        fprintf(outfile, "  addi    sp,sp,-16\n");
        fprintf(outfile, "  sd      s0,8(sp)\n");
        fprintf(outfile, "  addi    s0,sp,16\n");

        // body
        fprintf(outfile, "  li      a5,0\n");

        // epilogue
        fprintf(outfile, "  mv      a0,a5\n");
        fprintf(outfile, "  ld      s0,8(sp)\n");
        fprintf(outfile, "  addi    sp,sp,16\n");
        fprintf(outfile, "  jr      ra\n");
}

void codegen(struct ExtDecl *program) {
        //
        function(program);
}
