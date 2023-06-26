
#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// scan.c

enum TokenKind {
    TK_IDENT,  // identifier or keyword
    TK_NUM,    // integer token
    TK_PUNCT,  // punctuation
    TK_EOF,    // end of file
};

struct Token {
    enum TokenKind kind;
    struct Token *next;
    char *buffer;
    int len;
};

struct Token *b_scan(char *stream);
void print(struct Token *token);
