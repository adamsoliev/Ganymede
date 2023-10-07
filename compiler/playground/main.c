#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// DATA STRUCTURES
// clang-format off
enum TokenKind { /* KEYWORDS */
                INT,
                IF,
                RETURN,
                OPAR,
                CPAR,
                OCBR,
                CCBR,
                LT,     // <
                GT,     // >
                LE,     // <=
                GE,     // >=
                SEMIC,
                ASGN,
                IDENT,
                ICON,
};
// clang-format on

struct Token {
        enum TokenKind kind;
        union {
                int64_t icon;
                const char *scon;  // identifier string | string literal
        } value;
        struct Token *next;
};

struct Edecl {
        /* DECL */
        uint64_t type;
        char *name;
        struct Expr *value; /* can act like 
                                - value for decl
                                - value for 'return'
                                - ident for 'goto' 
                                - expr for expr-stmt
                            */

        enum EdeclKind { FUNC, DECL, S_IF, S_RETURN, S_COMP } kind;
        /* STMT */
        struct Expr *cond;
        struct Edecl *then;
        struct Edecl *body;  // compound stmt

        struct Edecl *next;
};

enum ExprType { E_ICON, E_IDENT, E_LT, E_GT, E_LE };
struct Expr {
        enum ExprType kind;
        uint64_t value;
        char *ident;
        struct Expr *lhs;
        struct Expr *rhs;
};

/* GLOBALS */
int LEN; /* used in the scanning step to keep track of string length for identifiers and scon */
#define TYPE_INT 0x0000000000000003  // 0000,0000,0011

/* --------- HASH TABLE --------- */
struct KeyValuePair {
        const char *key;
        int64_t value;
};

#define TABLE_SIZE 4096
struct KeyValuePair ht[TABLE_SIZE];

int hash(const char *key) {
        unsigned hash = 1;
        int c;
        while ((c = *key++)) {
                hash = hash * 263 + c;
        }
        return (int)(hash % TABLE_SIZE);
}

void insert(const char *key, int64_t value) {
        int index = hash(key);
        ht[index].key = key;
        ht[index].value = value;
}

uint64_t get(const char *key) {
        int index = hash(key);
        return ht[index].value;
}
/* --------- END --------- */

// UTILS
bool iswhitespace(char c) { return c == ' ' || c == '\t' || c == '\n'; }
bool isidentifier(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool isicon(char c) { return c >= '0' && c <= '9'; }
bool ispunctuation(char c) {
        return c == '(' || c == ')' || c == '{' || c == '}' || c == '>' || c == '<' || c == '=' ||
               c == ';';
}

// FORWARD DECLARATIONS
struct Edecl *declaration(struct Token **token);
struct Expr *expr(struct Token **token);
struct Expr *primary(struct Token **token);
struct Edecl *stmt(struct Token **token);
void cg_stmt(struct Edecl *lstmt);

struct Token *newtoken(enum TokenKind kind, const char *lexeme) {
        struct Token *token = (struct Token *)malloc(sizeof(struct Token));
        assert(token != NULL);
        token->kind = kind;
        switch (kind) {
                case INT:
                case IF:
                case RETURN: break;
                case ICON: token->value.icon = strtoll(lexeme, NULL, 10); break;
                case IDENT: token->value.scon = strndup(lexeme, LEN); break;
                case OPAR:
                case CPAR:
                case OCBR:
                case CCBR:
                case LT:
                case GT:
                case LE:
                case ASGN:
                case SEMIC: break;
                default: assert(0);
        }
        return token;
}

struct Expr *newexpr(enum ExprType kind) {
        struct Expr *expr = malloc(sizeof(struct Expr));
        expr->kind = kind;
        return expr;
}

void addtoken(struct Token **head, struct Token **tail, struct Token *newtoken) {
        if (*tail == NULL)
                *head = *tail = newtoken;
        else {
                (*tail)->next = newtoken;
                *tail = newtoken;
        }
}

void consume(struct Token **token, enum TokenKind kind) {
        if ((*token)->kind != kind) {
                printf("Expected %d, but got %d\n", kind, (*token)->kind);
                assert(0);
        }
        *token = (*token)->next;
}

/* ----------------------------------------------------------------- */
/* ---------------------------- SCANNER ---------------------------- */
/* ----------------------------------------------------------------- */
void scan(const char *program, struct Token **tokenlist) {
        int length = strlen(program);
        int start = 0;
        int current = 0;
        struct Token *head = NULL;
        struct Token *tail = NULL;

        while (current < length) {
                // skip whitespace
                while (current < length && iswhitespace(program[current])) current++;

                while (current < length && !iswhitespace(program[current])) {
                        start = current;
                        enum TokenKind kind = -1;
                        if (strncmp(program + current, "int", 3) == 0) { /* KEYWORDS */
                                current += 3;
                                kind = INT;
                        } else if (strncmp(program + current, "if", 2) == 0) {
                                current += 2;
                                kind = IF;
                        } else if (strncmp(program + current, "return", 6) == 0) {
                                current += 6;
                                kind = RETURN;
                        } else if (isidentifier(program[current])) { /* IDENTIFIER */
                                while (isidentifier(program[current])) current++;
                                LEN = current - start;
                                kind = IDENT;
                        } else if (ispunctuation(program[current])) { /* PUNCTUATION */
                                if (program[current] == '(') {
                                        current++;
                                        kind = OPAR;
                                } else if (program[current] == ')') {
                                        current++;
                                        kind = CPAR;
                                } else if (program[current] == '{') {
                                        current++;
                                        kind = OCBR;
                                } else if (program[current] == '}') {
                                        current++;
                                        kind = CCBR;
                                } else if (program[current] == '<') {
                                        current++;
                                        if (program[current] == '=') {
                                                current++;
                                                kind = LE;
                                        } else
                                                kind = LT;
                                } else if (program[current] == '>') {
                                        current++;
                                        kind = GT;
                                } else if (program[current] == ';') {
                                        current++;
                                        kind = SEMIC;
                                } else if (program[current] == '=') {
                                        current++;
                                        kind = ASGN;
                                } else {
                                        assert(0);
                                }
                        } else if (isicon(program[current])) { /* INT LITERAL */
                                while (isicon(program[current])) current++;
                                kind = ICON;
                        } else {
                                printf("Unrecognized char: %c\n", program[current]);
                                assert(0);
                        }
                        struct Token *token = newtoken(kind, program + start);
                        addtoken(&head, &tail, token);
                }
        }
        *tokenlist = head;
}

void printTokens(struct Token *head) {
        struct Token *current = head;
        while (current != NULL) {
                if (current->kind == ICON) {
                        printf("ICON, Value: %lu\n", current->value.icon);
                } else if (current->kind == IDENT) {
                        printf("IDENT, Value: %s\n", current->value.scon);
                } else {
                        printf("%d\n", current->kind);
                }
                current = current->next;
        }
}

/* ---------------------------------------------------------------- */
/* ---------------------------- PARSER ---------------------------- */
/* ---------------------------------------------------------------- */
struct Edecl *parse(struct Token *head) {
        struct Token *current = head;
        struct Edecl *decl = malloc(sizeof(struct Edecl)); /* FUNCTION */
        decl->kind = FUNC;
        while (current != NULL) {
                decl->type |= TYPE_INT;
                consume(&current, INT);

                decl->name = strdup(current->value.scon);
                consume(&current, IDENT);

                consume(&current, OPAR);
                consume(&current, CPAR);

                struct Edecl *ldeclhead = malloc(sizeof(struct Edecl));
                struct Edecl *ldecltail = ldeclhead;

                consume(&current, OCBR);

                while (current->kind != CCBR) {
                        /* LOCAL LEVEL */
                        if (current->kind == INT) {
                                ldecltail = ldecltail->next = declaration(&current);
                        } else if (current->kind == IF) {
                                struct Edecl *lstmt = malloc(sizeof(struct Edecl));
                                lstmt->kind = S_IF;
                                consume(&current, IF);
                                consume(&current, OPAR);

                                /* EXPR */
                                struct Expr *cond = expr(&current);
                                lstmt->cond = cond;

                                consume(&current, CPAR);
                                consume(&current, OCBR);

                                /* STMT */
                                struct Edecl *then = stmt(&current);
                                lstmt->then = then;

                                consume(&current, CCBR);

                                ldecltail = ldecltail->next = lstmt;
                        } else {
                                struct Edecl *lstmt = stmt(&current);
                                ldecltail = ldecltail->next = lstmt;
                        }
                }
                assert(current->kind == CCBR);
                consume(&current, CCBR);

                decl->body = ldeclhead->next;
        }
        return decl;
}

struct Edecl *declaration(struct Token **token) {
        struct Edecl *ldecl = malloc(sizeof(struct Edecl));

        struct Token *current = *token;

        ldecl->type |= TYPE_INT;
        consume(&current, INT);

        ldecl->name = strdup(current->value.scon);
        consume(&current, IDENT);

        consume(&current, ASGN);

        struct Expr *value = malloc(sizeof(struct Expr));
        value->value = current->value.icon;
        ldecl->value = value;
        consume(&current, ICON);

        insert(ldecl->name, ldecl->value->value);

        consume(&current, SEMIC);

        *token = current;
        return ldecl;
}

struct Edecl *stmt(struct Token **token) {
        struct Token *current = *token;

        struct Edecl *lstmt = malloc(sizeof(struct Edecl));
        lstmt->kind = S_RETURN;
        consume(&current, RETURN);

        lstmt->value = expr(&current);

        consume(&current, SEMIC);

        *token = current;
        return lstmt;
}

struct Expr *expr(struct Token **token) {
        struct Token *current = *token;
        struct Expr *lhs = primary(&current);

        switch (current->kind) {
                enum ExprType ekind = -1;
                case GT: ekind = E_GT; goto found;
                case LT: ekind = E_LT; goto found;
                case LE:
                        ekind = E_LE;
                        goto found;
                found: {
                        consume(&current, current->kind);
                        assert(ekind != -1);
                        struct Expr *parent = newexpr(ekind);
                        parent->lhs = lhs;
                        parent->rhs = primary(&current);
                        *token = current;
                        return parent;
                };
                default: break;
        }
        *token = current;
        return lhs;
}

struct Expr *primary(struct Token **token) {
        struct Token *current = *token;
        struct Expr *expr = malloc(sizeof(struct Expr));

        if (current->kind == IDENT) {
                expr->ident = strdup(current->value.scon);
                expr->kind = E_IDENT;
                consume(&current, IDENT);
        } else if (current->kind == ICON) {
                expr->value = current->value.icon;
                expr->kind = E_ICON;
                consume(&current, ICON);
        } else
                assert(0);
        *token = current;
        return expr;
}

/* ----------------------------------------------------------------- */
/* ---------------------------- CODEGEN ---------------------------- */
/* ----------------------------------------------------------------- */
void codegen(struct Edecl *decl) {
        printf("\n  .globl %s\n", decl->name);
        printf("\n%s:\n", decl->name);

        // prologue
        printf("  addi    sp,sp,-16\n");
        printf("  sd      s0,8(sp)\n");
        printf("  addi    s0,sp,16\n");

        // body
        struct Edecl *body = decl->body;
        while (body != NULL) {
                cg_stmt(body);
                body = body->next;
        }

        // epilogue
        printf(".Lend:\n");
        printf("  mv      a0,a5\n");
        printf("  ld      s0,8(sp)\n");
        printf("  addi    sp,sp,16\n");
        printf("  jr      ra\n");
}

void cg_stmt(struct Edecl *lstmt) {
        if (lstmt->kind == S_IF) {
                struct Expr *lhs = lstmt->cond->lhs;
                struct Expr *rhs = lstmt->cond->rhs;
                int value = get(lhs->ident);

                if (lstmt->cond->kind == E_GT) {
                        printf("  li      a3,%d\n", value);
                        printf("  li      a4,%lu\n", rhs->value);
                        printf("  ble     a3,a4,.L1end\n");
                } else if (lstmt->cond->kind == E_LT) {
                        printf("  li      a3,%lu\n", rhs->value);
                        printf("  li      a4,%d\n", value);
                        printf("  ble     a3,a4,.L1end\n");
                } else if (lstmt->cond->kind == E_LE) {
                        printf("  li      a3,%d\n", value);
                        printf("  li      a4,%lu\n", rhs->value);
                        printf("  bgt     a3,a4,.L1end\n");
                } else
                        assert(0);

                cg_stmt(lstmt->then);
                printf(".L1end:\n");
        } else if (lstmt->kind == S_RETURN) {
                printf("  li      a5,%lu\n", lstmt->value->value);
                printf("  j      .Lend\n");
        } else
                ; /* declaration */
}

int main(int argc, char **argv) {
        if (argc < 2) assert(0);

        struct Token *tokenlist = NULL;
        scan(argv[1], &tokenlist);
        struct Edecl *decllist = parse(tokenlist);
        codegen(decllist);

        return 0;
}
