#include "ganymede.h"

// -------------- ASSUME UNLIMITED REGISTERS

static void decl(struct ExtDecl *decl) { return; }

static void stmt(struct stmt *statement) {
        switch (statement->kind) {
                case STMT_COMPOUND:
                        for (struct block *cur = statement->body; cur != NULL; cur = cur->next) {
                                if (cur->decl != NULL) {
                                        decl(cur->decl);
                                } else {
                                        stmt(cur->stmt);
                                }
                        }
                        break;
                case RETURN: {
                        fprintf(outfile, "  li      a5,0\n");
                        break;
                }
                default: error("codegen: stmt: unimplemented");
        }
}

// function
static void function(struct ExtDecl *func) {
        fprintf(outfile, "  .globl %s\n", func->decltor->name);
        fprintf(outfile, "%s:\n", func->decltor->name);

        // prologue
        fprintf(outfile, "  addi    sp,sp,-16\n");
        fprintf(outfile, "  sd      s0,8(sp)\n");
        fprintf(outfile, "  addi    s0,sp,16\n");

        // body
        stmt(func->compStmt);

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
