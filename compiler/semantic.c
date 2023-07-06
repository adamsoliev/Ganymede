#include "baikalc.h"

struct symbol *symbol_create(enum symbol_t kind, struct type *type,
                             char *name) {
    struct symbol *s = calloc(sizeof(*s), 1);
    s->kind = kind;
    s->type = type;
    s->name = name;
    s->which = 0;
    return s;
}

// detect duplicate symbols & variable without a declaration
void decl_resolve(struct decl *d) {
    if (!d) return;
    enum symbol_t kind = scope_level() > 1 ? SYMBOL_LOCAL : SYMBOL_GLOBAL;
    d->symbol = symbol_create(kind, d->type, d->name);
    expr_resolve(d->value);
    scope_bind(d->name, d->symbol);

    if (d->code) {
        scope_enter();
        param_list_resolve(d->type->params);
        stmt_resolve(d->code);
        scope_exit();
    }
    decl_resolve(d->next);
}

void expr_resolve(struct expr *e) {
    if (!e) return;
    if (e->kind == EXPR_NAME) {
        e->symbol = scope_lookup(e->name);
    } else {
        expr_resolve(e->left);
        expr_resolve(e->right);
    }
}

// stmt_resolve()
// param_list_resolve()

// type checking
bool type_equal(struct type *a, struct type *b) {
    if (a->kind != b->kind) return false;
    /*
    if (a and b are atomic types) return true;
    else if (both are array) return type_equal(a->subtype, b->subtype);
    else if (both are function) return type_equal(a->subtype, b->subtype) and type_equal(a->params, b->params);
    else return false;
    */
}

struct type *type_copy(struct type *t) {
    /*
    return a duplicate of t, ensuring to duplicate any substructure recursively
    */
}

void type_delete(struct type *t) {
    /*
    free all memory allocated for t, including any substructure
    */
}

// complete this for all kinds of expressions
struct type *expr_typecheck(struct expr *e) {
    if (!e) return 0;
    struct type *lt = expr_typecheck(e->left);
    struct type *rt = expr_typecheck(e->right);
    struct type *result;
    switch (e->kind) {
        case EXPR_INTEGER_LITERAL:
            result = type_create(TYPE_INTEGER, 0, 0);
            break;
        case EXPR_STRING_LITERAL:
            result = type_create(TYPE_STRING, 0, 0);
            break;
        case EXPR_ADD:
            if (lt->kind != TYPE_INTEGER || rt->kind != TYPE_INTEGER) {
                printf("type error: add requires integer operands\n");
                exit(1);
            }
            result = type_create(TYPE_INTEGER, 0, 0);
            break;
        case EXPR_EQ:
        case EXPR_NE:
            if (!type_equals(lt, rt)) {
                /* display an error */
            }
            if (lt->kind == TYPE_VOID || lt->kind == TYPE_ARRAY ||
                lt->kind == TYPE_FUNCTION) {
                /* display an error */
            }
            result = type_create(TYPE_BOOLEAN, 0, 0);
            break;
        case EXPR_DEREF:
            if (lt->kind == TYPE_ARRAY) {
                if (rt->kind != TYPE_INTEGER) {
                    /* error: index not an integer */
                }
                result = type_copy(lt->subtype);
            } else {
                /* error: not an array */
                /* but we need to return a valid type */
                result = type_copy(lt);
            }
            break;
            /* more cases here */
    }
    type_delete(lt);
    type_delete(rt);
    return result;
}

struct type *decl_typecheck(struct decl *d) {
    if (d->value) {
        struct type *t;
        t = expr_typecheck(d->value);
        if (!type_equals(t, d->symbol->type)) {
            /* display an error */
        }
    }
    if (d->code) {
        stmt_typecheck(d->code);
    }
}

// enforce constraints to each kind of statement
struct type *stmt_typecheck(struct stmt *s) {
    struct type *t;
    switch (s->kind) {
        case STMT_EXPR:
            t = expr_typecheck(s->expr);
            type_delete(t);
            break;
        case STMT_IF_THEN:
            t = expr_typecheck(s->expr);
            if (t->kind != TYPE_BOOLEAN) {
                /* display an error */
            }
            type_delete(t);
            stmt_typecheck(s->body);
            stmt_typecheck(s->else_body);
            break;
            /* more cases here */
    }
}

// write a function typePrint that displays printf-like format strings,
// but suppors symbols like %T for types and %E for expressions, and so forth.
// e.g.,
// myprintf(
// "error: cannot add a %T (%E) to a %T (%E)\n",
// lt,e->left,rt,e->right
// );

void semantic_analysis(struct decl *d) {
    scope_enter();
    decl_resolve(d);
    decl_typecheck(d);
    scope_exit();
};
