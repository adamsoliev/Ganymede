#pragma clang diagnostic ignored "-Wgnu-empty-initializer"
#pragma clang diagnostic ignored "-Wgnu-binary-literal"

#include "ganymede.h"

int error(char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    return -1;
}

typedef struct Token Token;

enum TokenKind {
    TK_VOID,
    TK_CHAR,
    TK_SHORT,
    TK_INT,
    TK_LONG,
    TK_FLOAT,
    TK_DOUBLE,
    TK_UNSIGNED,
    TK_SIGNED,
    TK_EOF,
};

struct Token {
    enum TokenKind kind;
    Token *next;
};

Token *new_token(enum TokenKind kind) {
    Token *tok = calloc(1, sizeof(Token));
    tok->kind = kind;
    return tok;
}

uint64_t declspec(Token *token) {
    uint64_t t = -1;
    while (token->kind != TK_EOF) {
        switch (token->kind) {
            case TK_VOID:
                if (t != -1) return error("Invalid void specifier\n");
                t = TYPE_VOID;
                break;
            case TK_CHAR:
                if (t != -1) return error("Invalid char specifier\n");
                t = TYPE_CHAR;
                break;
            case TK_SHORT:
                t = TYPE_SHORT;
            case TK_INT:
                if (t != -1) return error("Invalid int specifier\n");
                t = TYPE_INT;
                break;
            case TK_LONG:
                t = TYPE_LONG;
            case TK_FLOAT:
                t = TYPE_FLOAT;
            case TK_DOUBLE:
                t = TYPE_DOUBLE;
            case TK_UNSIGNED:
                t = TYPE_UNSIGNED;
            case TK_SIGNED:
                t = TYPE_SIGNED;
            default:
                error("Invalid token kind: %d\n", token->kind);
        }
        token = token->next;
    }
    return t;
}

void decl(void) {
    //
}

void testvoid1(void);
void testvoid2(void);
void testvoid3(void);
void testchar1(void);

int main(void) {
    testvoid1();
    testvoid2();
    testvoid3();
    testchar1();
    return 0;
}

void testvoid1(void) {
    Token token = {};
    Token *cur = &token;
    cur = cur->next = new_token(TK_VOID);
    cur = cur->next = new_token(TK_EOF);
    assert(declspec(token.next) == TYPE_VOID);
}

void testvoid2(void) {
    Token token = {};
    Token *cur = &token;
    cur = cur->next = new_token(TK_VOID);
    cur = cur->next = new_token(TK_INT);
    cur = cur->next = new_token(TK_EOF);
    assert(declspec(token.next) == -1);
}

void testvoid3(void) {
    Token token = {};
    Token *cur = &token;
    cur = cur->next = new_token(TK_CHAR);
    cur = cur->next = new_token(TK_VOID);
    cur = cur->next = new_token(TK_EOF);
    assert(declspec(token.next) == -1);
}

void testchar1(void) {
    Token token = {};
    Token *cur = &token;
    cur = cur->next = new_token(TK_CHAR);
    cur = cur->next = new_token(TK_EOF);

    assert(declspec(token.next) == TYPE_CHAR);
}
