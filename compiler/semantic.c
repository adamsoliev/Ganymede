#include "baikalc.h"

/*
    Traverse the AST and perform semantic analysis.
*/

struct scope {
    struct scope *next;
    struct HashMap *vars;
};

// Forward declarations
struct symbol *symbol_create(enum symbol_t kind, struct type *type, char *name);
void scope_enter(void);
void scope_exit(void);
int scope_level(void);
void scope_bind(const char *name, struct symbol *sym);
struct symbol *scope_lookup(const char *name);
struct symbol *scope_lookup_current(const char *name);
void decl_resolve(struct decl *d);
void expr_resolve(struct expr *e);
void stmt_resolve(struct stmt *s);
bool type_equals(struct type *a, struct type *b);
struct type *type_copy(struct type *t);
void type_delete(struct type *t);
struct type *type_create(enum type_t kind, struct type *subtype,
                         struct param_list *params);
struct type *expr_typecheck(struct expr *e);
void decl_typecheck(struct decl *d);
void stmt_typecheck(struct stmt *s);

static struct scope *cur_scope = &(struct scope){};
static struct decl *cur_func;

struct symbol *symbol_create(enum symbol_t kind, struct type *type,
                             char *name) {
    struct symbol *s = calloc(1, sizeof(*s));
    s->kind = kind;
    s->type = type;
    s->name = name;
    s->which = 0;
    return s;
}

void scope_enter(void) {
    struct scope *sc = calloc(1, sizeof(struct scope));
    sc->next = cur_scope;  // so that cur_scope is always at the head
    cur_scope = sc;
}

void scope_exit(void) { cur_scope = cur_scope->next; }

int scope_level(void) {
    int level = 0;
    for (struct scope *sc = cur_scope; sc; sc = sc->next) {
        level++;
    }
    return level;
}

void scope_bind(const char *name, struct symbol *sym) {
    if (!cur_scope->vars) {
        cur_scope->vars = calloc(1, sizeof(struct HashMap));
    }
    hashmap_put(cur_scope->vars, name, sym);
}

struct symbol *scope_lookup(const char *name) {
    for (struct scope *sc = cur_scope; sc; sc = sc->next) {
        struct symbol *sym = hashmap_get(sc->vars, name);
        if (sym) return sym;
    }
    return NULL;
}

struct symbol *scope_lookup_current(const char *name) {
    return hashmap_get(cur_scope->vars, name);
}

// // detect duplicate symbols & variable without a declaration
void decl_resolve(struct decl *d) {
    if (!d) return;
    enum symbol_t kind = scope_level() > 1 ? SYMBOL_LOCAL : SYMBOL_GLOBAL;
    d->symbol = symbol_create(kind, d->type, d->name);
    // expr_resolve(d->value);
    scope_bind(d->name, d->symbol);

    if (d->code) {
        scope_enter();
        // param_list_resolve(d->type->params);
        stmt_resolve(d->code);
        scope_exit();
    }
    decl_resolve(d->next);
}

void expr_resolve(struct expr *e) {
    if (!e) return;
    // if (e->kind == EXPR_NAME) {
    //     e->symbol = scope_lookup(e->name);
    // } else {
    //     expr_resolve(e->left);
    //     expr_resolve(e->right);
    // }
    if (e->kind == EXPR_INTEGER_LITERAL) {
        return;
    }
}

void stmt_resolve(struct stmt *s) {
    if (!s) return;
    if (s->kind == STMT_RETURN) {
        expr_resolve(s->expr);
    }
}

// param_list_resolve()

// type checking
bool type_equals(struct type *a, struct type *b) {
    // if (a and b are atomic types) return true;
    if (a->kind == b->kind) return true;
    // else if (both are array) return type_equal(a->subtype, b->subtype);
    // else if (both are function) return type_equal(a->subtype, b->subtype) and type_equal(a->params, b->params);
    else
        return false;
}

// return a duplicate of t, ensuring to duplicate any substructure recursively
struct type *type_copy(struct type *t) {
    if (!t) return 0;
    struct type *result = calloc(1, sizeof(*result));
    result->kind = t->kind;
    result->subtype = type_copy(t->subtype);
    // result->params = param_list_copy(t->params);
    return result;
}

// free all memory allocated for t, including any substructure
void type_delete(struct type *t) {
    if (!t) return;
    type_delete(t->subtype);
    // param_list_delete(t->params);
    free(t->subtype);
    free(t->params);
    free(t);
}

struct type *type_create(enum type_t kind, struct type *subtype,
                         struct param_list *params) {
    struct type *t = calloc(1, sizeof(*t));
    t->kind = kind;
    t->subtype = subtype;
    t->params = params;
    return t;
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
            // case EXPR_STRING_LITERAL:
            //     result = type_create(TYPE_STRING, 0, 0);
            //     break;
            case EXPR_ADD:
            case EXPR_SUB:
                if (lt->kind != TYPE_INTEGER || rt->kind != TYPE_INTEGER) {
                    printf("type error: add requires integer operands\n");
                    exit(1);
                }
                result = type_create(TYPE_INTEGER, 0, 0);
                break;
            // case EXPR_EQ:
            // case EXPR_NE:
            //     if (!type_equals(lt, rt)) {
            //         /* display an error */
            //     }
            //     if (lt->kind == TYPE_VOID || lt->kind == TYPE_ARRAY ||
            //         lt->kind == TYPE_FUNCTION) {
            //         /* display an error */
            //     }
            //     result = type_create(TYPE_BOOLEAN, 0, 0);
            //     break;
            // case EXPR_DEREF:
            //     if (lt->kind == TYPE_ARRAY) {
            //         if (rt->kind != TYPE_INTEGER) {
            //             /* error: index not an integer */
            //         }
            //         result = type_copy(lt->subtype);
            //     } else {
            //         /* error: not an array */
            //         /* but we need to return a valid type */
            //         result = type_copy(lt);
            //     }
            //     break;
            // /* more cases here */
    }
    type_delete(lt);
    type_delete(rt);
    return result;
}

void decl_typecheck(struct decl *d) {
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
void stmt_typecheck(struct stmt *s) {
    struct type *t;
    switch (s->kind) {
        case STMT_EXPR:
            t = expr_typecheck(s->expr);
            type_delete(t);
            break;
            // case STMT_IF_THEN:
            //     t = expr_typecheck(s->expr);
            //     if (t->kind != TYPE_BOOLEAN) {
            /* display an error */
            // }
            // type_delete(t);
            // stmt_typecheck(s->body);
            // stmt_typecheck(s->else_body);
            // break;
            /* more cases here */
        case STMT_RETURN:
            t = expr_typecheck(s->expr);
            if (!type_equals(t, cur_func->type->subtype)) {
                /* display an error */
                printf("type error: return type mismatch\n");
            }
            type_delete(t);
            break;
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
    // prog is a set of declarations
    for (struct decl *decl = d; decl; decl = decl->next) {
        if (decl->type->kind == TYPE_FUNCTION) {
            cur_func = decl;
        }
        scope_enter();
        decl_resolve(d);  // name resolution
        decl_typecheck(d);
        scope_exit();
    }
};
