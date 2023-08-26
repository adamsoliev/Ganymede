#include "ganymede.h"

static struct Token *ct;

// either function or declaration
struct ExtDecl {
        struct ExtDecl *next;
        struct declspec *declspec;
        struct decltor *decltor;
        struct expr *expr;    // for declaration
        struct block *block;  // for function
};

struct declspec {
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

struct expr {
        enum {
                E_INT,
        } kind;
        int value;
};

// statement or declaration
struct block {
        struct block *next;
        struct ExtDecl *decl;
        struct stmt *stmt;
};

struct stmt {
        struct expr *expr;
        enum {
                S_RETURN,
        } kind;
};

void copystr(char **dest, char **src, int len);
void consume(enum TokenKind kind);
struct ExtDecl *function(struct declspec **declspec, struct decltor **decltor);
struct ExtDecl *declaration(struct declspec **declspec, struct decltor **decltor);
struct expr *expr();
struct declspec *declaration_specifiers();
struct decltor *declarator();

void consume(enum TokenKind kind) {
        if (ct->kind != kind) {
                error("Expected %s, got %s", token_names[kind], token_names[ct->kind]);
        }
        ct = ct->next;
}

// function-definition ::=
//      declarator ("{" declaration* or statement* "}")? ;
struct ExtDecl *function(struct declspec **declspec, struct decltor **decltor) {
        struct ExtDecl *func = calloc(1, sizeof(struct ExtDecl));
        func->declspec = *declspec;
        func->decltor = *decltor;

        struct block head = {};
        struct block *cur = &head;
        if (ct->kind == OCBR) {
                consume(OCBR);
                while (ct->kind != CCBR) {
                        // declaration or statement
                        switch (ct->kind) {
                                case IDENT:
                                case CASE:
                                case DEFAULT:
                                case IF:
                                case SWITCH:
                                case WHILE:
                                case DO:
                                case FOR:
                                case GOTO:
                                case CONTINUE:
                                case BREAK: break;
                                case RETURN:
                                        consume(RETURN);
                                        cur = cur->next = calloc(1, sizeof(struct block));
                                        cur->stmt = calloc(1, sizeof(struct stmt));
                                        cur->stmt->kind = S_RETURN;
                                        if (ct->kind != SEMIC) {
                                                cur->stmt->expr = expr();
                                        }
                                        break;
                                default:
                                        // declaration
                                        {
                                                struct declspec *declspec =
                                                        declaration_specifiers();
                                                struct decltor *decltor = declarator();
                                                func = func->next =
                                                        declaration(&declspec, &decltor);
                                        }
                        }
                }
                consume(CCBR);
        }
        return func;
};

// declaration ::=
// 	    declspec decltor ("=" expr)? ("," decltor ("=" expr)?)* ";"
struct ExtDecl *declaration(struct declspec **declspec, struct decltor **decltor) {
        struct ExtDecl head = {};
        struct ExtDecl *cur = &head;
        cur = cur->next = calloc(1, sizeof(struct ExtDecl));
        cur->declspec = *declspec;
        cur->decltor = *decltor;
        while (ct->kind != SEMIC) {
                if (ct->kind == ASSIGN) {
                        consume(ASSIGN);
                        cur->expr = expr();
                }
                if (ct->kind == COMMA) {
                        consume(COMMA);
                        cur = cur->next = calloc(1, sizeof(struct ExtDecl));
                        cur->declspec = *declspec;
                        cur->decltor = declarator();
                }
        }
        consume(SEMIC);
        return head.next;
};

struct declspec *declaration_specifiers() {
        struct declspec *declspec = calloc(1, sizeof(struct declspec));
        if (ct->kind == INT) {
                consume(INT);
                declspec->type = D_INT;
                return declspec;
        }
        return declspec;
};

// declarator ::=
// 	    pointer? (identifier or "(" declarator ")")
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
        return decltor;
};

struct expr *expr() {
        struct expr *expr = calloc(1, sizeof(struct expr));
        if (ct->kind == INTCONST) {
                expr->kind = E_INT;
                expr->value = ct->value;
                consume(INTCONST);
                return expr;
        }
        return expr;
};

// direct-declarator ::=
// 	    "[" type-qualifier-list? assignment-expression? "]"
// 	    "[" "static" type-qualifier-list? assignment-expression "]"
// 	    "[" type-qualifier-list "static" assignment-expression "]"
// 	    "[" type-qualifier-list? "*" "]"
// 	    "(" parameter-type-list ")"
// 	    "(" identifier-list? ")"

// function-definition
// declaration
struct ExtDecl *parse(struct Token *tokens) {
        ct = tokens;
        struct ExtDecl head = {};
        struct ExtDecl *cur = &head;
        while (tokens->kind != EOI) {
                struct declspec *declspec = declaration_specifiers();
                struct decltor *decltor = declarator();  // #1 declarator
                if (decltor->kind == FUNCTION) {
                        cur = cur->next = function(&declspec, &decltor);
                } else {
                        cur = cur->next = declaration(&declspec, &decltor);
                }
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
