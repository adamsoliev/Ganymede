
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

//
extern FILE *outfile;

// scan.c
enum TokenKind {
    TK_IDENT,    // identifier or keyword
    TK_NUM,      // integer token
    TK_KEYWORD,  // keyword
    TK_STR,      // string literal
    TK_CHAR,     // character literal
    TK_PUNCT,    // punctuation
    TK_ERROR,    // error token
    TK_EOF,      // end of file
};

struct Token {
    enum TokenKind kind;
    struct Token *next;
    char *buffer;
    int len;
};

struct Token *b_scan(char *stream);
void print(struct Token *token);
bool equal(struct Token *token, char *str);
bool consume(struct Token **rest, struct Token *token, char *str);
struct Token *skip(struct Token *token, char *str);
void error(bool shouldExit, char *fmt, ...);

// parse.c

// 1. Declaration (properties of a symbol)
// Symbol could be constants, variables or functions.

struct decl {
    char *name;
    struct type *type;
    struct expr *value;
    struct stmt *code;
    struct decl *next;
    struct symbol *symbol;
};

// 2. Statement (if, for, while, do-while, switch, goto, break, continue, return)

enum stmt_t {
    STMT_DECL,
    STMT_EXPR,
    STMT_IF_ELSE,
    STMT_FOR,
    STMT_PRINT,
    STMT_RETURN,
    STMT_BLOCK
};

struct stmt {
    enum stmt_t kind;
    struct decl *decl;
    struct expr *init_expr;
    struct expr *expr;
    struct expr *next_expr;
    struct stmt *body;
    struct stmt *else_body;
    struct stmt *next;
};

// 3. Expression (assignment, arithmetic, logical, bitwise, ternary, comma, sizeof, cast, call, array, struct, pointer, postfix, prefix, constant, identifier)

enum expr_t {
    EXPR_ADD,
    EXPR_SUB,
    EXPR_MUL,
    EXPR_DIV,
    EXPR_DECL,
    EXPR_EQ,      // ==
    EXPR_NE,      // !=
    EXPR_LT,      // <
    EXPR_LE,      // <=
    EXPR_ASSIGN,  // =
    EXPR_BITNOT,  // ~
    EXPR_NOT,     // !
    EXPR_DEREF,   // *
    EXPR_ADDR,    // &
    EXPR_NAME,
    EXPR_INTEGER_LITERAL,
    EXPR_STRING_LITERAL,
    EXPR_FUNCALL,
    EXPR_ARGS
};

struct expr {
    enum expr_t kind;
    struct expr *left;
    struct expr *right;
    const char *name;
    int integer_value;
    const char *string_literal;
    struct symbol *symbol;
    struct decl *decl;  // for ('int i = 0'; i < 10; i++)
};

// 4. Types

enum type_t {
    TYPE_VOID,
    TYPE_BOOLEAN,
    TYPE_CHARACTER,
    TYPE_INTEGER,
    TYPE_STRING,
    TYPE_ARRAY,
    TYPE_FUNCTION,
    TYPE_POINTER
};

struct type {
    enum type_t kind;
    struct type *subtype;
    struct param_list *params;
};

// 5. Parameter list

struct param_list {
    char *name;
    struct type *type;
    struct param_list *next;
};

struct decl *parse(struct Token *token);
void print_decl(struct decl *decl, int level);
void print_stmt(struct stmt *stmt, int level);
void print_expr(struct expr *expr, int level);
void print_type(struct type *type, int level);

// semantic.c
enum symbol_t { SYMBOL_LOCAL, SYMBOL_PARAM, SYMBOL_GLOBAL };

struct symbol {
    enum symbol_t kind;  // local, param, global
    struct type *type;
    char *name;
    int which;
};

void scope_enter(void);
void scope_exit();
int scope_level(void);

void semantic_analysis(struct decl *d);

void scope_bind(const char *name, struct symbol *sym);
struct symbol *scope_lookup(const char *name);
struct symbol *scope_lookup_current(const char *name);

//
// hashmap.c
//

struct HashEntry {
    char *key;
    int keylen;
    void *val;
};

struct HashMap {
    struct HashEntry *buckets;
    int capacity;
    int used;
};

void *hashmap_get(struct HashMap *map, char *key);
void *hashmap_get2(struct HashMap *map, char *key, int keylen);
void hashmap_put(struct HashMap *map, char *key, void *val);
void hashmap_put2(struct HashMap *map, char *key, int keylen, void *val);
void hashmap_delete(struct HashMap *map, char *key);
void hashmap_delete2(struct HashMap *map, char *key, int keylen);
void hashmap_test(void);

// ir.c

void irgen(struct decl *d);

// codegen.c

void codegen();