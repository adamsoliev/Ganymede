#include "ganymede.h"

static const int MAX_IDENT_LEN = 20;
static const int MAX_INT_LEN = 10;
static const int MAX_PUNCT_LEN = 2;
static struct Token *currentToken;

static struct Token *new_token(enum TokenKind kind, char *buffer, int len) {
        struct Token *token = calloc(1, sizeof(struct Token));
        token->kind = kind;
        token->buffer = buffer;
        token->len = len;
        return token;
}

static void skip_line_comment(char **stream) {
        // Skip characters until the end of line or end of input
        while (**stream && **stream != '\n') {
                (*stream)++;
        }
        if (**stream == '\n') {
                (*stream)++;
        }
}

static void skip_block_comment(char **stream) {
        // Skip characters until the end of block comment (*/)
        while (**stream && !(**stream == '*' && *(*stream + 1) == '/')) {
                (*stream)++;
        }
        if (**stream == '*') {
                (*stream)++;
        }
        if (**stream == '/') {
                (*stream)++;
        }
        if (**stream == '\n') {
                (*stream)++;
        }
}

static bool LexIdentContinue(char **stream) {
        char *start = *stream;
        while (isalnum(**stream) || **stream == '_') {
                (*stream)++;
        }
        // keywords
        static char *kw[] = {
                "return",    "if",         "else",
                "for",       "while",      "int",
                "sizeof",    "char",       "struct",
                "union",     "short",      "long",
                "void",      "typedef",    "_Bool",
                "enum",      "static",     "goto",
                "break",     "continue",   "switch",
                "case",      "default",    "extern",
                "_Alignof",  "_Alignas",   "do",
                "signed",    "unsigned",   "const",
                "volatile",  "auto",       "register",
                "restrict",  "__restrict", "__restrict__",
                "_Noreturn", "float",      "double",
                "typeof",    "asm",        "_Thread_local",
                "__thread",  "_Atomic",    "__attribute__",
        };
        int num_keywords = sizeof(kw) / sizeof(kw[0]);
        for (int i = 0; i < num_keywords; i++) {
                if (strncmp(start, kw[i], *stream - start) == 0 &&
                    strlen(kw[i]) == (*stream - start)) {
                        currentToken =
                                new_token(TK_KEYWORD, start, *stream - start);
                        return true;
                }
        }
        // identifers
        currentToken = new_token(TK_IDENT, start, *stream - start);
        return true;
}

static bool LexNumContinue(char **stream) {
        char *start = *stream;
        (*stream)++;
        if (*start == '0') {
                if (**stream == 'x' || **stream == 'X') {
                        // hexadecimal
                        (*stream)++;
                        while (isxdigit(**stream)) {
                                (*stream)++;
                        }
                        if (**stream == 'l' || **stream == 'L') {
                                (*stream)++;
                        }
                        currentToken =
                                new_token(TK_NUM, start, *stream - start);
                        return true;
                } else if (isdigit(**stream)) {
                        // octal
                        // FIXME: check for **stream != '8' && **stream != '9'
                        while (isdigit(**stream)) {
                                (*stream)++;
                        }
                        if (**stream == 'l' || **stream == 'L') {
                                (*stream)++;
                        }
                        currentToken =
                                new_token(TK_NUM, start, *stream - start);
                        return true;
                } else if (**stream == 'b' || **stream == 'B') {
                        // binary
                        (*stream)++;
                        while (**stream == '0' || **stream == '1') {
                                (*stream)++;
                        }
                        if (**stream == 'l' || **stream == 'L') {
                                (*stream)++;
                        }
                        currentToken =
                                new_token(TK_NUM, start, *stream - start);
                        return true;
                }
        }
        while (isdigit(**stream) || strchr(".eE+-", **stream)) {
                // floating point
                if (**stream && *(*stream + 1) && strchr("eE", **stream) &&
                    strchr("+-", *(*stream + 1))) {
                        (*stream) += 2;
                }
                if (**stream == '.' || strchr("eE", **stream)) {
                        (*stream)++;
                }
                (*stream)++;
        }
        if (**stream == 'l' || **stream == 'L') {
                (*stream)++;
        }
        currentToken = new_token(TK_NUM, start, *stream - start);
        return true;
}

static bool LexPunctContinue(char **stream) {
        // Ignore comments
        if (**stream == '/') {
                if (*(*stream + 1) == '/') {
                        (*stream) += 2;
                        skip_line_comment(stream);
                        return false;
                } else if (*(*stream + 1) == '*') {
                        (*stream) += 2;
                        skip_block_comment(stream);
                        return false;
                }
        }

        char *start = *stream;
        char c = *start;
        (*stream)++;

        switch (c) {
                case '"':
                        // String literal
                        while (**stream && **stream != '"') {
                                if (**stream == '\\') {  // escape sequence
                                        (*stream)++;
                                }
                                (*stream)++;
                        }
                        if (**stream == '"') {
                                (*stream)++;
                                currentToken = new_token(
                                        TK_STR, start, *stream - start);
                        } else {
                                currentToken = new_token(
                                        TK_ERROR, start, *stream - start);
                        }
                        break;
                case '\'':
                        // Character literal
                        while (**stream && **stream != '\'') {
                                if (**stream == '\\') {  // escape sequence
                                        (*stream)++;
                                }
                                (*stream)++;
                        }
                        if (**stream == '\'') {
                                (*stream)++;
                                currentToken = new_token(
                                        TK_CHAR, start, *stream - start);
                        } else {
                                currentToken = new_token(
                                        TK_ERROR, start, *stream - start);
                        }
                        break;
                default:
                        // Other punctuation
                        while (**stream && ispunct(**stream) &&
                               *stream - start <= MAX_PUNCT_LEN &&
                               **stream == '=') {
                                (*stream)++;
                        }
                        currentToken =
                                new_token(TK_PUNCT, start, *stream - start);
                        break;
        }
        // return currentToken->kind != TK_ERROR;
        return true;
}

struct Token *b_scan(char *stream) {
        struct Token head = {};
        struct Token *cur = &head;
        while (*stream) {
                if (isspace(*stream)) {
                        stream++;
                        continue;
                } else if (isalpha(*stream) || *stream == '_') {
                        LexIdentContinue(&stream);
                } else if (isdigit(*stream) ||
                           (*stream == '.' && isdigit(*(stream + 1)))) {
                        LexNumContinue(&stream);
                } else if (ispunct(*stream)) {
                        if (!LexPunctContinue(&stream)) continue;
                } else {
                        error(true, "cannot tokenize char: %s\n", stream);
                }
                cur = cur->next = currentToken;
        }
        cur->next = new_token(TK_EOF, NULL, 0);
        return head.next;
}

void print(struct Token *token) {
        const char *TokenKindNames[] = {"TK_IDENT",
                                        "TK_NUM",
                                        "TK_KEYWORD",
                                        "TK_STR",
                                        "TK_CHAR",
                                        "TK_PUNCT",
                                        "TK_ERROR",
                                        "TK_EOF"};
        while (token->kind != TK_EOF) {
                // enum name
                fprintf(outfile, "%-13s", TokenKindNames[token->kind]);
                // token buffer
                for (int i = 0; i < token->len; i++) {
                        fprintf(outfile, "%c", token->buffer[i]);
                }
                fprintf(outfile, "\n");
                token = token->next;
        }
}

bool equal(struct Token *token, char *str) {
        // return strncmp(token->buffer, str, token->len) == 0;
        return memcmp(token->buffer, str, token->len) == 0 &&
               str[token->len] == '\0';
}

bool consume(struct Token **rest, struct Token *token, char *str) {
        if (equal(token, str)) {
                *rest = token->next;
                return true;
        }
        *rest = token;
        return false;
}

struct Token *skip(struct Token *token, char *str) {
        if (equal(token, str)) {
                return token->next;
        }
        error(true,
              "expected '%s', but got '%.*s'",
              str,
              token->len,
              token->buffer);
        return NULL;  // satisfy the compiler
}

void error(bool shouleExit, char *fmt, ...) {
        va_list args;
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        fprintf(stderr, "\n");
        if (shouleExit) exit(1);
}