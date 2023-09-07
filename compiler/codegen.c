#include "ganymede.h"

// will hold ht, where (name *, ExtDecl *)
static struct scope *scope = &(struct scope){};

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

void enter_scope(void) {
        struct scope *new_scope = calloc(1, sizeof(struct scope));
        new_scope->next = scope;
        new_scope->vars = ht_create();
        scope = new_scope;
}

void leave_scope(void) {
        struct scope *old_scope = scope;
        scope = scope->next;
        ht_destroy(old_scope->vars);
        free(old_scope);
}

struct ExtDecl *find_var(char *name) {
        for (struct scope *cur = scope; cur != NULL; cur = cur->next) {
                struct ExtDecl *extDecl = ht_get(cur->vars, name);
                if (extDecl != NULL) {
                        return extDecl;
                }
        }
        return NULL;
}

struct expr *const_fold(struct expr *expression) {
        if (expression->kind == SIZEOF) {
                if (expression->lhs->kind == IDENT) {
                        struct ExtDecl *extDecl = find_var(expression->lhs->strLit);
                        assert(extDecl != NULL && extDecl->declspec != NULL);
                        assert(extDecl->declspec->type == INT);
                        struct expr *expr = new_expr(INT, NULL, NULL);
                        expr->ivalue = sizeof(int);
                        return expr;
                }
                error("codegen: sizeof unimplemented");
        }
        if (expression->kind == ADD || expression->kind == SUB || expression->kind == MUL ||
            expression->kind == DIV || expression->kind == RSHIFT || expression->kind == LSHIFT ||
            expression->kind == LT || expression->kind == GT || expression->kind == LEQ ||
            expression->kind == GEQ || expression->kind == EQ || expression->kind == NEQ ||
            expression->kind == ANDAND || expression->kind == OROR || expression->kind == AND ||
            expression->kind == OR || expression->kind == XOR) {
                if (expression->kind == AND)
                        assert(expression->lhs != NULL && expression->rhs != NULL);
                struct expr *rhs = const_fold(expression->rhs);
                struct expr *lhs = const_fold(expression->lhs);
                assert(lhs->kind == rhs->kind);
                if (lhs->kind == INT && rhs->kind == INT) {
                        int result = 0;
                        if (expression->kind == ADD)
                                result = lhs->ivalue + rhs->ivalue;
                        else if (expression->kind == SUB)
                                result = lhs->ivalue - rhs->ivalue;
                        else if (expression->kind == MUL)
                                result = lhs->ivalue * rhs->ivalue;
                        else if (expression->kind == DIV)
                                result = lhs->ivalue / rhs->ivalue;
                        else if (expression->kind == RSHIFT)
                                result = lhs->ivalue >> rhs->ivalue;
                        else if (expression->kind == LSHIFT)
                                result = lhs->ivalue << rhs->ivalue;
                        else if (expression->kind == LT)
                                result = lhs->ivalue < rhs->ivalue;
                        else if (expression->kind == GT)
                                result = lhs->ivalue > rhs->ivalue;
                        else if (expression->kind == LEQ)
                                result = lhs->ivalue <= rhs->ivalue;
                        else if (expression->kind == GEQ)
                                result = lhs->ivalue >= rhs->ivalue;
                        else if (expression->kind == EQ)
                                result = lhs->ivalue == rhs->ivalue;
                        else if (expression->kind == NEQ)
                                result = lhs->ivalue != rhs->ivalue;
                        else if (expression->kind == ANDAND)
                                result = lhs->ivalue && rhs->ivalue;
                        else if (expression->kind == OROR)
                                result = lhs->ivalue || rhs->ivalue;
                        else if (expression->kind == AND)
                                result = lhs->ivalue & rhs->ivalue;
                        else if (expression->kind == OR)
                                result = lhs->ivalue | rhs->ivalue;
                        else if (expression->kind == XOR)
                                result = lhs->ivalue ^ rhs->ivalue;
                        else
                                error("codegen: inner op unimplemented");
                        expression->kind = INT;
                        expression->ivalue = result;
                        return expression;
                }
                error("codegen: type unimplemented");
        } else if (expression->kind == INT) {
                return expression;
        }
        error("codegen: outer op unimplemented");
        return NULL;
}

char *expr(struct expr *expression) {
        expression = const_fold(expression);
        if (expression->kind == INT) {
                char *reg = nextr();
                fprintf(outfile, "  li      %s,%d\n", reg, expression->ivalue);
                return reg;
        }
        return NULL;
}

static void decl(struct ExtDecl *decl) {
        char *reg = expr(decl->expr);
        return;
}

static void stmt(struct stmt *statement) {
        switch (statement->kind) {
                case STMT_COMPOUND:
                        for (struct block *cur = statement->body; cur != NULL; cur = cur->next) {
                                if (cur->decl != NULL) {
                                        ht_set(scope->vars, cur->decl->decltor->name, cur->decl);
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
        enter_scope();

        fprintf(outfile, "  .globl %s\n", func->decltor->name);
        fprintf(outfile, "%s:\n", func->decltor->name);

        // prologue
        fprintf(outfile, "  # prologue\n");
        fprintf(outfile, "  addi    sp,sp,-16\n");
        fprintf(outfile, "  sd      s0,8(sp)\n");
        fprintf(outfile, "  addi    s0,sp,16\n");

        // body
        fprintf(outfile, "  # body\n");
        stmt(func->compStmt);

        // epilogue
        fprintf(outfile, "  # epilogue\n");
        fprintf(outfile, "  mv      a0,a5\n");
        fprintf(outfile, "  ld      s0,8(sp)\n");
        fprintf(outfile, "  addi    sp,sp,16\n");
        fprintf(outfile, "  jr      ra\n");

        leave_scope();
}

void codegen(struct ExtDecl *program) {
        //
        function(program);
}
