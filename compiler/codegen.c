#include "ganymede.h"

// -------------- ASSUME UNLIMITED REGISTERS
static char *nextr(void) {
        static int regCount = 0;
        int size = 0;
        if (regCount < 10)
                size = 1;
        else if (regCount < 100)
                size = 2;
        else {
                printf("Too many registers in use\n");
                exit(1);
        }

        char *reg = calloc(size + 1, sizeof(char));
        snprintf(reg, sizeof(reg), "r%d", regCount);
        regCount++;
        return reg;
}

struct expr *const_fold(struct expr *expression) {
        if (expression->kind == ADD || expression->kind == SUB || expression->kind == MUL ||
            expression->kind == DIV) {
                struct expr *lhs = const_fold(expression->lhs);
                struct expr *rhs = const_fold(expression->rhs);
                if (lhs->kind == INT && rhs->kind == INT) {
                        int result;
                        if (expression->kind == ADD)
                                result = lhs->ivalue + rhs->ivalue;
                        else if (expression->kind == SUB)
                                result = lhs->ivalue - rhs->ivalue;
                        else if (expression->kind == MUL)
                                result = lhs->ivalue * rhs->ivalue;
                        else
                                result = lhs->ivalue / rhs->ivalue;
                        expression->kind = INT;
                        expression->ivalue = result;
                        return expression;
                }
                return expression;
        } else if (expression->kind == INT) {
                return expression;
        }
        return NULL;
}

char *expr(struct expr *expression) {
        if (expression->kind == ADD || expression->kind == SUB || expression->kind == MUL ||
            expression->kind == DIV) {
                char *lhs = expr(expression->lhs);
                char *rhs = expr(expression->rhs);
                if (expression->kind == ADD)
                        fprintf(outfile, "  add     %s,%s,%s\n", lhs, lhs, rhs);
                else if (expression->kind == SUB)
                        fprintf(outfile, "  sub     %s,%s,%s\n", lhs, lhs, rhs);
                else if (expression->kind == MUL)
                        fprintf(outfile, "  mul     %s,%s,%s\n", lhs, lhs, rhs);
                else
                        fprintf(outfile, "  div     %s,%s,%s\n", lhs, lhs, rhs);
                return lhs;
        }
        if (expression->kind == INT) {
                char *reg = nextr();
                fprintf(outfile, "  li      %s,%d\n", reg, expression->ivalue);
                return reg;
        }
        return NULL;
}

static void decl(struct ExtDecl *decl) {
        char *reg = expr(const_fold(decl->expr));
        return;
}

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
