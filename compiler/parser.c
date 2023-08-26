#include "ganymede.h"

static struct Token *ct;

// either function or declaration
struct Node {
        struct Node *next;
        struct declspec *declspec;
        struct decltor *decltor;
};

struct declspec {
        //
        enum {
                D_INT,
        } type;
};

struct decltor {
        //
        char *name;
        enum {
                FUNCTION,
                DECLARATION,
        } kind;
};

void copystr(char **dest, char **src, int len);

void consume(enum TokenKind kind) {
        if (ct->kind != kind) {
                error("Expected %s, got %s",
                      token_names[kind],
                      token_names[ct->kind]);
        }
        ct = ct->next;
}

// function-definition ::=
// 	declarator declaration-list? compound-statement
struct Node *function(struct declspec *declspec, struct decltor *decltor){
        //
};

// declaration ::=
// 	init-declarator-list? ";"
struct Node *declaration(struct declspec *declspec, struct decltor *decltor) {
        if (ct->kind == ASSIGN) {
                consume(ASSIGN);
                if (ct->kind == INTCONST) {
                        consume(INTCONST);
                } else {
                        error("Expected number, got %s", token_names[ct->kind]);
                }
        }
        consume(SEMIC);
};

struct declspec *declaration_specifiers() {
        struct declspec *declspec = calloc(1, sizeof(struct declspec));
        if (ct->kind == INT) {
                consume(INT);
                declspec->type = D_INT;
                return declspec;
        }
        error("Expected int, got %s", token_names[ct->kind]);
};

struct decltor *declarator() {
        struct decltor *decltor = calloc(1, sizeof(struct decltor));
        if (ct->kind == IDENT) {
                copystr(&decltor->name, &ct->start, ct->len);
                consume(IDENT);
                if (ct->kind == OPAR) {
                        consume(OPAR);
                        decltor->kind = FUNCTION;
                        consume(CPAR);
                        return decltor;
                } else {
                        decltor->kind = DECLARATION;
                        return decltor;
                }
        }
        error("Expected identifier, got %s", token_names[ct->kind]);
};

// function-definition
// declaration
struct Node *parse(struct Token *tokens) {
        ct = tokens;
        struct Node head = {};
        struct Node *cur = &head;
        while (tokens->kind != EOI) {
                struct declspec *declspec = declaration_specifiers();
                struct decltor *decltor = declarator();
                if (decltor->kind == FUNCTION) {
                        cur->next = function(declspec, decltor);
                } else {
                        cur->next = declaration(declspec, decltor);
                }
                cur = cur->next;
        }
        return head.next;
};

// UTILS
void copystr(char **dest, char **src, int len) {
        if (*dest == NULL) {
                *dest = calloc(len, sizeof(char));
        }
        strncpy(*dest, *src, len);
        (*dest)[len] = '\0';
};
