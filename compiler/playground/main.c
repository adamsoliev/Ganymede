#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// DATA STRUCTURES
enum TokenType { /* KEYWORDS */
                 INT,
                 IF,
                 RETURN,
                 /* PUNCT */
                 OPAR,
                 CPAR,
                 OCBR,
                 CCBR,
                 GT,
                 SEMIC,
                 ASGN,
                 /* IDENT */
                 IDENT,
                 /* LITERALS */
                 ICON,
};

struct Token {
        enum TokenType type;
        union {
                int64_t icon;
                const char *scon;  // identifier string | string literal
        } value;
        struct Token *next;
};

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
void parse(void) {
        //
}

/* -------------- CODEGEN -------------- */
void codegen(void) {
        //
}

int main(void) {
        char *program = "int main() { int a = 23; if (a > 10) { return 3; } return 0; }";
        struct Token *tokenlist = NULL;
        scan(program, &tokenlist);
        printTokens(tokenlist);
        return 0;
}
