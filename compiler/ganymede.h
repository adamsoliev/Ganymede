
#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// scan.c
extern FILE *outfile;
extern char *limit;

enum TokenKind {
        LT,      // <
        GT,      // >
        LEQ,     // <=
        GEQ,     // >=c getopt and optarg header
        LSHIFT,  // <<
        RSHIFT,  // >>
        DEREF,   // ->
        DECR,    // --
        EQ,      // ==
        NEQ,     // !=
        ADD,
        SUB,
        MUL,
        DIV,
        MOD,        // %
        ADDASSIGN,  // +=
        SUBASSIGN,  // -=
        MULASSIGN,  // *=
        DIVASSIGN,  // /=
        MODASSIGN,  // %=
        OROR,       // ||
        ANDAND,     // &&
        INCR,       // ++
        EOI,        // end of input
        IF,
        INT,
        OBR,        // [
        CBR,        // ]
        OCBR,       // {
        CCBR,       // }
        OPAR,       // (
        CPAR,       // )
        SEMIC,      // ;
        COMMA,      // ,
        TILDA,      // ~
        AND,        // &
        OR,         // |
        XOR,        // ^
        NOT,        // !
        ANDASSIGN,  // &=
        ORASSIGN,   // |=
        XORASSIGN,  // ^=
        NOTASSIGN,  // !=
        STRGIZE,    // #
        TKPASTE,    // ##
        ASSIGN,     // =
        QMARK,      // ?
        IDENT,
        INTCONST,
        FLOATCONST,
        STRCONST,
        CHARCONST,
        ELLIPSIS,
        AUTO,
        CASE,
        CHAR,
        CONST,
        CONTINUE,
        DEFAULT,
        DO,
        DOUBLE,
        ELSE,
        ENUM,
        EXTERN,
        FLOAT,
        FOR,
        GOTO,
        LONG,
        REGISTER,
        RETURN,
        SHORT,
        SIGNED,
        SIZEOF,
        STATIC,
        STRUCT,
        SWITCH,
        TYPEDEF,
        UNION,
        UNSIGNED,
        VOID,
        VOLATILE,
        WHILE,
        DOT,
        BREAK,
        COLON,
        RSHIFTASSIGN,
        LSHIFTASSIGN,
        INCLUDE,
        DEFINE,
        BACKSLASH,
};

struct Token {
        enum TokenKind kind;
        char *start;  // for IDENT
        int len;      // for IDENT
        struct Token *next;
        int value;  // for INTCONST
};

void error(char *fmt, ...);
void printTokens(struct Token *head, FILE *outfile);
struct Token *scan(char *stream);
