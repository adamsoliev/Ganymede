
#ifndef HEADER_FILE
#define HEADER_FILE

#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// scan.c
extern FILE* outfile;
extern char* limit;

enum Kind {
        LT,      // <
        GT,      // >
        LEQ,     // <=
        GEQ,     // >=
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
        STMT_EXPR,  // different from other stmt types
        STMT_COMPOUND,
        DOUBLECONST,
        LONGDOUBLECONST,
};

struct Token {
        enum Kind kind;
        char* start;  // for IDENT
        int len;      // for IDENT
        struct Token* next;
        union {
                int ivalue;
                float fvalue;
                double dvalue;
                long double ldvalue;
        };
};

extern char* token_names[];
void error(char* fmt, ...);
void printTokens(struct Token* head, FILE* outfile);
struct Token* scan(char* stream);

// parser
struct ExtDecl* parse(struct Token* tokens);
void printExtDecl(struct ExtDecl* extDecl, int level);

// hashmap.c

typedef struct ht ht;

typedef struct {
        const char* key;  // current key
        void* value;      // current value

        ht* _table;     // reference to hash table being iterated
        size_t _index;  // current index into ht._entries
} hti;

typedef struct {
        const char* key;  // key is NULL if this slot is empty
        void* value;
} ht_entry;

struct ht {
        ht_entry* entries;  // hash slots
        size_t capacity;    // size of _entries array
        size_t length;      // number of items in hash table
};

ht* ht_create(void);
void ht_destroy(ht* table);
void* ht_get(ht* table, const char* key);
const char* ht_set(ht* table, const char* key, void* value);
size_t ht_length(ht* table);
hti ht_iterator(ht* table);
bool ht_next(hti* it);
void ht_test(void);
char* strdup(const char* s);

#endif
