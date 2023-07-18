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

static void expr_ir(struct expr *e) {
    if (!e) return;
    switch (e->kind) {
        case EXPR_INTEGER_LITERAL:
            fprintf(out, " i32 %d", e->integer_value);
            break;
        case EXPR_ADD: {
            if (e->left->kind != EXPR_INTEGER_LITERAL ||
                e->right->kind != EXPR_INTEGER_LITERAL) {
                fprintf(stderr, "type error: add requires integer operands\n");
                exit(1);
            }
            fprintf(out,
                    " i32 %d",
                    e->left->integer_value + e->right->integer_value);
            break;
        }
        case EXPR_SUB: {
            if (e->left->kind != EXPR_INTEGER_LITERAL ||
                e->right->kind != EXPR_INTEGER_LITERAL) {
                fprintf(stderr, "type error: add requires integer operands\n");
                exit(1);
            }
            fprintf(out,
                    " i32 %d",
                    e->left->integer_value - e->right->integer_value);
            break;
        }
    }
}

static void stmt_ir(struct stmt *s) {
    if (!s) return;
    if (s->kind == STMT_RETURN) {
        fprintf(out, " ret");
        expr_ir(s->expr);
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
        printf("Error: cannot open file\n");
        exit(1);
    }
    fprintf(out, "target triple = \"riscv64-unknown-unknown\"\n");

    decl_ir(d);
    fprintf(out, "\n");

    fclose(out);
}
