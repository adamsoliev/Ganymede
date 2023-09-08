#pragma clang diagnostic ignored "-Wgnu-empty-initializer"

#include "ganymede.h"

// will hold ht, where (name *, ExtDecl *)
static struct scope *scope = &(struct scope){};

#define EXTDECL_SIZE 10
struct varlist {
        struct ExtDecl *extDecl;
        char *label;
};
struct varlist *list[EXTDECL_SIZE];
static int Index = 0;

static struct varlist *new_varlist(struct ExtDecl *extDecl, char *label) {
        struct varlist *varlist = calloc(1, sizeof(struct varlist));
        varlist->extDecl = extDecl;
        varlist->label = label;
        return varlist;
}

// -------------- ASSUME UNLIMITED REGISTERS
static char *nextr(void) {  // next register
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

static char *nextl(void) {  // next label
        static int labCount = 0;
        int size = 1;  // single digit labCount
        assert(labCount < (EXTDECL_SIZE - 1));
        char *label = calloc(size + 3, sizeof(char));
        snprintf(label, sizeof(label), ".LC%d", labCount);
        labCount++;
        return label;
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
        } else if (expression->kind == ADD || expression->kind == SUB || expression->kind == MUL ||
                   expression->kind == DIV || expression->kind == RSHIFT ||
                   expression->kind == LSHIFT || expression->kind == LT || expression->kind == GT ||
                   expression->kind == LEQ || expression->kind == GEQ || expression->kind == EQ ||
                   expression->kind == NEQ || expression->kind == ANDAND ||
                   expression->kind == OROR || expression->kind == AND || expression->kind == OR ||
                   expression->kind == XOR) {
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
        } else if (expression->kind == INT || expression->kind == CHAR ||
                   expression->kind == STRCONST) {
                return expression;
        } else if (expression->kind == QMARK) {
                struct expr *eq = const_fold(expression->lhs);
                struct expr *option1 = const_fold(expression->rhs->lhs);
                struct expr *option2 = const_fold(expression->rhs->rhs);
                assert(eq->kind == INT);
                if (eq->ivalue == 0)
                        return option2;
                else
                        return option1;
        } else if (expression->kind == IDENT) {
                char *name = expression->strLit;
                struct ExtDecl *extDecl = ht_get(scope->vars, name);
                assert(extDecl != NULL && extDecl->expr->kind == INT);
                struct expr *expr = new_expr(INT, NULL, NULL);
                expr->ivalue = extDecl->expr->ivalue;
                return expr;
        } else {
                error("codegen: outer op unimplemented");
        }
        return NULL;
}

char *expr(struct expr *expression) {
        expression = const_fold(expression);
        if (expression->kind == INT) {
                char *reg = nextr();
                fprintf(outfile, "  li      %s,%d\n", reg, expression->ivalue);
                return reg;
        }
        if (expression->kind == CHAR) {
                char *reg = nextr();
                int cval = 0;
                if (expression->strLit[0] == '\\') {
                        if (expression->strLit[1] == 'n')
                                cval = '\n' - '\0';
                        else if (expression->strLit[1] == 't')
                                cval = '\t' - '\0';
                        else if (expression->strLit[1] == 'v')
                                cval = '\v' - '\0';
                        else if (expression->strLit[1] == 'b')
                                cval = '\b' - '\0';
                        else if (expression->strLit[1] == 'r')
                                cval = '\r' - '\0';
                        else if (expression->strLit[1] == 'f')
                                cval = '\f' - '\0';
                        else if (expression->strLit[1] == 'a')
                                cval = '\a' - '\0';
                        else if (expression->strLit[1] == '0')
                                cval = '\0' - '\0';
                        else if (expression->strLit[1] == '\\')
                                cval = '\\' - '\0';
                        else if (expression->strLit[1] == '\'')
                                cval = '\'' - '\0';
                        else if (expression->strLit[1] == '\"')
                                cval = '\"' - '\0';
                        else if (expression->strLit[1] == '\?')
                                cval = '\?' - '\0';
                        else
                                error("codegen: escape char unimplemented");
                } else {
                        cval = expression->strLit[0] - '\0';
                }
                fprintf(outfile, "  li      %s,%d\n", reg, cval);
                return reg;
        }
        return NULL;
}

static void decl(struct ExtDecl *decl, bool isGlobal) {
        if (isGlobal) {
                struct expr *result = const_fold(decl->expr);
                fprintf(outfile, "%s:\n", decl->decltor->name);
                if (decl->expr->kind == STRCONST) {
                        fprintf(outfile, "          .string %s\n", result->strLit);
                        fprintf(outfile,
                                "          .zero   %d\n",
                                decl->decltor->row - (int)strlen(result->strLit) + 1);
                } else
                        fprintf(outfile, "          .word   %d\n", result->ivalue);
        } else {
                char *reg = expr(decl->expr);
        }
        return;
}

static void stmt(struct stmt *statement) {
        switch (statement->kind) {
                case STMT_COMPOUND:
                        for (struct block *cur = statement->body; cur != NULL; cur = cur->next) {
                                if (cur->decl != NULL) {
                                        if (cur->decl->expr->kind == STRCONST) {
                                                assert(Index < (EXTDECL_SIZE - 1));
                                                char *reg = nextr();
                                                char *label = nextl();
                                                struct varlist *varlist =
                                                        new_varlist(cur->decl, label);
                                                list[Index++] = varlist;
                                                fprintf(outfile,
                                                        "  lui     %s,%%hi(%s)\n",
                                                        reg,
                                                        label);
                                                fprintf(outfile,
                                                        "  addi    %s,%s,%%lo(%s)\n",
                                                        reg,
                                                        reg,
                                                        label);
                                        } else {
                                                // vars that fit in a register
                                                ht_set(scope->vars,
                                                       cur->decl->decltor->name,
                                                       cur->decl);
                                                decl(cur->decl, false);
                                        }
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
        // fprintf(outfile, "\n  .globl %s\n", func->decltor->name);
        fprintf(outfile, "\n%s:\n", func->decltor->name);

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
}

void codegen(struct ExtDecl *program) {
        enter_scope();
        for (struct ExtDecl *cur = program; cur != NULL; cur = cur->next) {
                if (cur->decltor->kind == DECLARATION) {
                        decl(cur, true);
                        ht_set(scope->vars, cur->decltor->name, cur);
                } else if (cur->decltor->kind == FUNCTION) {
                        function(cur);
                        ht_set(scope->vars, cur->decltor->name, cur);
                } else
                        error("codegen: unimplemented\n");
        }
        leave_scope();

        for (int i = 0; i < Index; i++) {
                char *label = list[i]->label;
                struct ExtDecl *extDecl = list[i]->extDecl;
                fprintf(outfile, "%s:\n", label);
                fprintf(outfile, "          .string %s\n", extDecl->expr->strLit);
                fprintf(outfile,
                        "          .zero    %d\n",
                        extDecl->decltor->row - (int)strlen(extDecl->expr->strLit) + 1);
        }
}
