#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// DATA STRUCTURES
// clang-format off
enum TokenKind { /* KEYWORDS */
        /* 0 */        INT,
        /* 1 */        IF,
        /* 2 */        RETURN,
        /* 3 */        OPAR,
        /* 4 */        CPAR,
        /* 5 */        OCBR,
        /* 6 */        CCBR,
        /* 7 */        GT,
        /* 8 */        SEMIC,
        /* 9 */        ASGN,
       /* 10 */        IDENT,
       /* 11 */        ICON,
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

struct Expr {
        enum ExprType { E_ICON, E_IDENT, E_GT } kind;
        uint64_t value;
        char *ident;
        struct Expr *lhs;
        struct Expr *rhs;
};

/* GLOBALS */
int LEN; /* used in the scanning step to keep track of string length for identifiers and scon */
#define TYPE_INT 0x0000000000000003  // 0000,0000,0011

// UTILS
bool iswhitespace(char c) { return c == ' ' || c == '\t' || c == '\n'; }
bool isidentifier(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool isicon(char c) { return c >= '0' && c <= '9'; }
bool ispunctuation(char c) {
        return c == '(' || c == ')' || c == '{' || c == '}' || c == '>' || c == '=' || c == ';';
}

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
                case GT:
                case ASGN:
                case SEMIC: break;
                default: assert(0);
        }
        return token;
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

/* -------------- SCANNER -------------- */
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
                        if (strncmp(program + current, "int", 3) == 0) { /* KEYWORDS */
                                current += 3;
                                struct Token *token = newtoken(INT, program + start);
                                addtoken(&head, &tail, token);
                        } else if (strncmp(program + current, "if", 2) == 0) {
                                current += 2;
                                struct Token *token = newtoken(IF, program + start);
                                addtoken(&head, &tail, token);
                        } else if (strncmp(program + current, "return", 6) == 0) {
                                current += 6;
                                struct Token *token = newtoken(RETURN, program + start);
                                addtoken(&head, &tail, token);
                        } else if (isidentifier(program[current])) { /* IDENTIFIER */
                                while (isidentifier(program[current])) current++;
                                LEN = current - start;
                                struct Token *token = newtoken(IDENT, program + start);
                                addtoken(&head, &tail, token);
                        } else if (ispunctuation(program[current])) { /* PUNCTUATION */
                                if (program[current] == '(') {
                                        current++;
                                        struct Token *token = newtoken(OPAR, program + start);
                                        addtoken(&head, &tail, token);
                                } else if (program[current] == ')') {
                                        current++;
                                        struct Token *token = newtoken(CPAR, program + start);
                                        addtoken(&head, &tail, token);
                                } else if (program[current] == '{') {
                                        current++;
                                        struct Token *token = newtoken(OCBR, program + start);
                                        addtoken(&head, &tail, token);
                                } else if (program[current] == '}') {
                                        current++;
                                        struct Token *token = newtoken(CCBR, program + start);
                                        addtoken(&head, &tail, token);
                                } else if (program[current] == '>') {
                                        current++;
                                        struct Token *token = newtoken(GT, program + start);
                                        addtoken(&head, &tail, token);
                                } else if (program[current] == ';') {
                                        current++;
                                        struct Token *token = newtoken(SEMIC, program + start);
                                        addtoken(&head, &tail, token);
                                } else if (program[current] == '=') {
                                        current++;
                                        struct Token *token = newtoken(ASGN, program + start);
                                        addtoken(&head, &tail, token);
                                } else {
                                        assert(0);
                                }
                        } else if (isicon(program[current])) { /* INT LITERAL */
                                while (isicon(program[current])) current++;
                                struct Token *token = newtoken(ICON, program + start);
                                addtoken(&head, &tail, token);
                        } else {
                                printf("Unrecognized char: %c\n", program[current]);
                                assert(0);
                        }
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

/* -------------- PARSER -------------- */
struct Edecl *parse(struct Token *head) {
        struct Token *current = head;
        struct Edecl *decl = malloc(sizeof(struct Edecl)); /* FUNCTION */
        decl->kind = FUNC;
        while (current != NULL) {
                assert(current->kind == INT);
                decl->type |= TYPE_INT;
                consume(&current, INT);

                assert(current->kind == IDENT);
                decl->name = strdup(current->value.scon);
                consume(&current, IDENT);

                assert(current->kind == OPAR);
                consume(&current, OPAR);
                assert(current->kind == CPAR);
                consume(&current, CPAR);

                struct Edecl *ldeclhead = malloc(sizeof(struct Edecl));
                struct Edecl *ldecltail = ldeclhead;

                assert(current->kind == OCBR);
                consume(&current, OCBR);

                while (current->kind != CCBR) {
                        /* LOCAL LEVEL */
                        if (current->kind == INT) {
                                struct Edecl *ldecl = malloc(sizeof(struct Edecl));

                                assert(current->kind == INT);
                                ldecl->type |= TYPE_INT;
                                consume(&current, INT);

                                assert(current->kind == IDENT);
                                ldecl->name = strdup(current->value.scon);
                                consume(&current, IDENT);

                                assert(current->kind == ASGN);
                                consume(&current, ASGN);

                                assert(current->kind == ICON);
                                struct Expr *value = malloc(sizeof(struct Expr));
                                value->value = current->value.icon;
                                ldecl->value = value;
                                consume(&current, ICON);

                                assert(current->kind == SEMIC);
                                consume(&current, SEMIC);

                                ldecltail = ldecltail->next = ldecl;
                        } else if (current->kind == IF) {
                                struct Edecl *lstmt = malloc(sizeof(struct Edecl));
                                lstmt->kind = S_IF;
                                consume(&current, IF);

                                assert(current->kind == OPAR);
                                consume(&current, OPAR);

                                /* EXPR */
                                struct Expr *cond = malloc(sizeof(struct Expr));

                                /* - LHS */
                                struct Expr *lhs = malloc(sizeof(struct Expr));
                                assert(current->kind == IDENT);
                                lhs->ident = strdup(current->value.scon);
                                lhs->kind = E_IDENT;
                                consume(&current, IDENT);

                                /* - PARENT TYPE */
                                assert(current->kind == GT);
                                cond->kind = E_GT;
                                consume(&current, GT);

                                /* - RHS */
                                struct Expr *rhs = malloc(sizeof(struct Expr));
                                assert(current->kind == ICON);
                                rhs->value = current->value.icon;
                                rhs->kind = E_ICON;
                                consume(&current, ICON);

                                cond->lhs = lhs;
                                cond->rhs = rhs;

                                lstmt->cond = cond;

                                assert(current->kind == CPAR);
                                consume(&current, CPAR);

                                /* STMT */
                                struct Edecl *then = malloc(sizeof(struct Edecl));

                                assert(current->kind == OCBR);
                                consume(&current, OCBR);

                                assert(current->kind == RETURN);
                                then->kind = S_RETURN;
                                consume(&current, RETURN);

                                struct Expr *value = malloc(sizeof(struct Expr));
                                assert(current->kind == ICON);
                                value->kind = E_ICON;
                                value->value = current->value.icon;
                                consume(&current, ICON);

                                then->value = value;

                                assert(current->kind == SEMIC);
                                consume(&current, SEMIC);

                                assert(current->kind == CCBR);
                                consume(&current, CCBR);

                                lstmt->then = then;

                                ldecltail = ldecltail->next = lstmt;
                        } else {
                                struct Edecl *lstmt = malloc(sizeof(struct Edecl));

                                assert(current->kind == RETURN);
                                lstmt->kind = S_RETURN;
                                consume(&current, RETURN);

                                struct Expr *value = malloc(sizeof(struct Expr));
                                assert(current->kind == ICON);
                                value->kind = E_ICON;
                                value->value = current->value.icon;
                                consume(&current, ICON);

                                lstmt->value = value;

                                assert(current->kind == SEMIC);
                                consume(&current, SEMIC);

                                ldecltail = ldecltail->next = lstmt;
                        }
                }
                assert(current->kind == CCBR);
                consume(&current, CCBR);

                decl->body = ldeclhead->next;
        }
        return decl;
}

/* -------------- CODEGEN -------------- */
void codegen(void) {
        //
}

int main(void) {
        char *program = "int main() { int a = 23; if (a > 10) { return 3; } return 0; }";
        struct Token *tokenlist = NULL;
        scan(program, &tokenlist);
        struct Edecl *decllist = parse(tokenlist);
        printTokens(tokenlist);
        return 0;
}
