
#include "ganymede.h"

static FILE *out;
static struct decl *current_fn = NULL;

void prologue();
void epilogue();

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

static void expr_as(struct expr *e) {
    if (!e) return;
    switch (e->kind) {
        case EXPR_INTEGER_LITERAL:
            fprintf(out, "li        a1, %d\n", e->integer_value);
            return;
        case EXPR_SUB: {
            expr_as(e->left);
            fprintf(out, "mv        t0, a1\n");
            expr_as(e->right);
            fprintf(out, "sub       a1, t0, a1\n");
        } break;
        default: assert(false);
    }
}

static void stmt_ir(struct stmt *s) {
    if (!s) return;
    while (s) {
        switch (s->kind) {
            case STMT_RETURN: {
                expr_as(s->expr);
            } break;
            case STMT_BLOCK: {
                stmt_ir(s->body);
            } break;
            case STMT_DECL: {
                fprintf(out, " @%s()", s->decl->name);
                if (s->decl->value) {
                    fprintf(out, " =");
                    expr_as(s->decl->value);
                }
            } break;
        }
        s = s->next;
    }
}

static void decl_as(struct decl *d) {
    for (struct decl *decl = d; decl; decl = decl->next) {
        if (decl->type->kind == TYPE_FUNCTION) {
            current_fn = decl;

            prologue();
            stmt_ir(decl->code);
            epilogue();
        }
    }
}

void prologue() {
    fprintf(out, ".globl    %s\n", current_fn->name);
    fprintf(out, ".type     %s,@function\n", current_fn->name);
    fprintf(out, "main:\n");
    fprintf(out, "addi      sp, sp, -16\n");
    fprintf(out, "sd        s0, 8(sp)\n");
    fprintf(out, "addi      s0, sp, 16\n");
}

void epilogue() {
    fprintf(out, "mv        a0, a1\n");
    fprintf(out, "ld        s0, 8(sp)\n");
    fprintf(out, "addi      sp, sp, 16\n");
    fprintf(out, "jr        ra\n");
}

void codegen(struct decl *d) {
    // out = fopen("/home/adam/dev/ganymede/compiler/build/program.s", "w+");
    out = outfile;
    if (out == NULL) {
        error(true, "cannot open file\n");
    }

    decl_as(d);
    fprintf(out, "\n");

    fclose(out);
}