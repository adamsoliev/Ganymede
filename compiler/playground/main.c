#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// DATA STRUCTURES
// clang-format off
enum TokenType { /* KEYWORDS */
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
        enum TokenType type;
        union {
                int64_t icon;
                const char *scon;  // identifier string | string literal
        } value;
        struct Token *next;
};

struct Edecl {
        uint64_t type;
        char *name;
        uint64_t value;
        struct Edecl *next;
};

struct Stmt {
        enum StmtType { S_IF, S_RETURN, S_COMP } type;
        struct Expr *cond;
        struct Stmt *then;

        struct Expr *value; /* can act like 
                                - value for 'return'
                                - ident for 'goto' 
                                - expr for expr-stmt
                            */
};

struct Expr {
        enum ExprType { E_INT, E_IDENT, E_GT } type;
        uint64_t value;
        char *ident;
        struct Expr *lhs;
        struct Expr *rhs;
};

#define TYPE_INT 0x0000000000000003  // 0000,0000,0011

// UTILS
bool iswhitespace(char c) { return c == ' ' || c == '\t' || c == '\n'; }
bool isidentifier(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool isicon(char c) { return c >= '0' && c <= '9'; }
bool ispunctuation(char c) {
        return c == '(' || c == ')' || c == '{' || c == '}' || c == '>' || c == '=' || c == ';';
}

struct Token *newtoken(enum TokenType type, const char *lexeme) {
        struct Token *token = (struct Token *)malloc(sizeof(struct Token));
        assert(token != NULL);
        token->type = type;
        switch (type) {
                case INT:
                case IF:
                case RETURN: break;
                case ICON: token->value.icon = strtoll(lexeme, NULL, 10); break;
                case IDENT: token->value.scon = strdup(lexeme); break;
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

void consume(struct Token **token, enum TokenType type) {
        if ((*token)->type != type) {
                printf("Expected %d, but got %d\n", type, (*token)->type);
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

                start = current;

                while (current < length && !iswhitespace(program[current])) {
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
                if (current->type == ICON) {
                        printf("ICON, Value: %lu\n", current->value.icon);
                } else if (current->type == IDENT) {
                        printf("IDENT, Value: %s\n", current->value.scon);
                } else {
                        printf("%d\n", current->type);
                }
                current = current->next;
        }
}

/* -------------- PARSER -------------- */
void parse(struct Token *head) {
        struct Token *current = head;
        while (current != NULL) {
                struct Edecl *edecl = malloc(sizeof(struct Edecl));

                assert(current->type == INT);
                edecl->type |= TYPE_INT;
                consume(&current, INT);

                assert(current->type == IDENT);
                edecl->name = strdup(current->value.scon);
                consume(&current, IDENT);

                assert(current->type == OPAR);
                consume(&current, OPAR);
                assert(current->type == CPAR);
                consume(&current, CPAR);

                assert(current->type == OCBR);
                consume(&current, OCBR);

                while (current->type != CCBR) {
                        if (current->type == INT) {
                                struct Edecl *decl = malloc(sizeof(struct Edecl));

                                assert(current->type == INT);
                                decl->type |= TYPE_INT;
                                consume(&current, INT);

                                assert(current->type == IDENT);
                                decl->name = strdup(current->value.scon);
                                consume(&current, IDENT);

                                assert(current->type == ASGN);
                                consume(&current, ASGN);

                                assert(current->type == ICON);
                                decl->value = current->value.icon;
                                consume(&current, ICON);

                                assert(current->type == SEMIC);
                                consume(&current, SEMIC);
                        } else if (current->type == IF) {
                                struct Stmt *ifstmt = malloc(sizeof(struct Stmt));
                                ifstmt->type = S_IF;
                                consume(&current, IF);

                                assert(current->type == OPAR);
                                consume(&current, OPAR);

                                /* EXPR */
                                struct Expr *cond = malloc(sizeof(struct Expr));

                                /* - LHS */
                                struct Expr *lhs = malloc(sizeof(struct Expr));
                                assert(current->type == IDENT);
                                lhs->ident = strdup(current->value.scon);
                                lhs->type = E_IDENT;
                                consume(&current, IDENT);

                                /* - PARENT TYPE */
                                assert(current->type == GT);
                                cond->type = E_GT;
                                consume(&current, GT);

                                /* - RHS */
                                struct Expr *rhs = malloc(sizeof(struct Expr));
                                assert(current->type == ICON);
                                rhs->value = current->value.icon;
                                rhs->type = E_INT;
                                consume(&current, ICON);

                                cond->lhs = lhs;
                                cond->rhs = rhs;

                                assert(current->type == CPAR);
                                consume(&current, CPAR);

                                ifstmt->cond = cond;

                                /* STMT */
                                struct Stmt *then = malloc(sizeof(struct Stmt));

                                assert(current->type == OCBR);
                                consume(&current, OCBR);

                                assert(current->type == RETURN);
                                then->type = S_RETURN;
                                consume(&current, RETURN);

                                struct Expr *expr = malloc(sizeof(struct Expr));
                                assert(current->type == ICON);
                                expr->type = E_INT;
                                expr->value = current->value.icon;
                                consume(&current, ICON);

                                assert(current->type == SEMIC);
                                consume(&current, SEMIC);

                                assert(current->type == CCBR);
                                consume(&current, CCBR);

                                ifstmt->then = then;
                        } else {
                                struct Stmt *stmt = malloc(sizeof(struct Stmt));

                                assert(current->type == RETURN);
                                stmt->type = S_RETURN;
                                consume(&current, RETURN);

                                struct Expr *value = malloc(sizeof(struct Expr));
                                assert(current->type == ICON);
                                value->type = E_INT;
                                value->value = current->value.icon;
                                stmt->value = value;
                                consume(&current, ICON);

                                assert(current->type == SEMIC);
                                consume(&current, SEMIC);
                        }
                }

                assert(current->type == CCBR);
                consume(&current, CCBR);
        }
}

/* -------------- CODEGEN -------------- */
void codegen(void) {
        //
}

int main(void) {
        char *program = "int main() { int a = 23; if (a > 10) { return 3; } return 0; }";
        struct Token *tokenlist = NULL;
        scan(program, &tokenlist);
        parse(tokenlist);
        printTokens(tokenlist);
        return 0;
}
