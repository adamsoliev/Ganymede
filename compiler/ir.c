#include "baikalc.h"

static FILE *out;

static void type_ir(struct type *t) {
    if (!t) return;
    if (t->kind == TYPE_FUNCTION) {
        fprintf(out, "define");
        type_ir(t->subtype);
        // param_list_ir(t->params);
        return;
    }
    if (t->kind == TYPE_INTEGER) {
        fprintf(out, " i32");
    }
}

static int expr_ir(struct expr *e) {
    if (!e) return -1;
    switch (e->kind) {
        case EXPR_INTEGER_LITERAL: return e->integer_value;
        case EXPR_ADD: {
            int left = expr_ir(e->left);
            int right = expr_ir(e->right);
            return left + right;
        }
        case EXPR_SUB: {
            int left = expr_ir(e->left);
            int right = expr_ir(e->right);
            return left - right;
        }
        case EXPR_MUL: {
            int left = expr_ir(e->left);
            int right = expr_ir(e->right);
            return left * right;
        }
        case EXPR_DIV: {
            int left = expr_ir(e->left);
            int right = expr_ir(e->right);
            return left / right;
        }
    }
}

static void stmt_ir(struct stmt *s) {
    if (!s) return;
    if (s->kind == STMT_RETURN) {
        fprintf(out, " ret");
        int value = expr_ir(s->expr);
        fprintf(out, " i32 %d", value);
    }
}

static void decl_ir(struct decl *d) {
    for (struct decl *decl = d; decl; decl = decl->next) {
        type_ir(decl->type);
        fprintf(out, " @%s()", decl->name);  // name_ir(decl->name);
        if (decl->value) {
            fprintf(out, " =");
            expr_ir(decl->value);
        }
        if (decl->code) {
            fprintf(out, " {\n");
            stmt_ir(decl->code);
            fprintf(out, "\n}");
        }
        decl_ir(decl->next);
    }
}

void irgen(struct decl *d) {
    //
    out = fopen("./build/ir.ll", "w+");
    if (out == NULL) {
        error(true, "cannot open file\n");
    }
    fprintf(out, "target triple = \"riscv64-unknown-unknown\"\n");

    decl_ir(d);
    fprintf(out, "\n");

    fclose(out);
}
