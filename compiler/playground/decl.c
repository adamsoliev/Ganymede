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

Token *new_token(enum TokenKind kind) {
        Token *tok = calloc(1, sizeof(Token));
        tok->kind = kind;
        return tok;
}

uint64_t declspec(Token *token) {
        uint64_t t = 0;
        while (token->kind != TK_EOF) {
                switch (token->kind) {
                        case TK_VOID:
                                if (t) {
                                bterror:
                                        return error("too many basic types\n");
                                }
                                t = TYPE_VOID;
                                break;
                        case TK_CHAR:
                                if (t & TYPE_BMASK) goto bterror;
                                t = TYPE_CHAR;
                                break;
                        case TK_INT:
                                if (t & TYPE_BMASK) goto bterror;
                                t = TYPE_INT;
                                break;
                        case TK_FLOAT:
                                if (t & TYPE_BMASK) goto bterror;
                                t = TYPE_FLOAT;
                                break;
                        case TK_DOUBLE:
                                if (t & TYPE_BMASK) goto bterror;
                                t = TYPE_DOUBLE;
                                break;
                        case TK_SHORT:
                                if (t & 0x30) {
                                sterror:
                                        return error("invalid specifiers\n");
                                }
                                t |= TYPE_SHORT;
                                break;
                        case TK_LONG:
                                if (t & 0x30) goto sterror;
                                t |= TYPE_LONG;
                                break;
                        case TK_UNSIGNED:
                                if (t & 0xc0) goto sterror;
                                t |= TYPE_UNSIGNED;
                                break;
                        case TK_SIGNED:
                                if (t & 0xc0) goto sterror;
                                t |= TYPE_SIGNED;
                                break;
                        default: error("Invalid token kind: %d\n", token->kind);
                }
                token = token->next;
        }
        if (t & 0xf0) return error("invalid specifiers alongside void\n");
        return t;
}

void decl(void) {
        //
}

void testvoid(void);
void testchar(void);

int main(void) {
        testvoid();
        testchar();
        return 0;
}

Token *gentokens(int size, ...) {
        va_list args;
        va_start(args, size);
        Token token = {};
        Token *cur = &token;
        for (int i = 0; i < size; i++) {
                cur = cur->next = new_token(va_arg(args, enum TokenKind));
        }
        cur = cur->next = new_token(TK_EOF);

        va_end(args);
        return token.next;
}

void testvoid(void) {
        Token *token = gentokens(1, TK_VOID);
        assert(declspec(token) == TYPE_VOID);
        token = gentokens(2, TK_VOID, TK_INT);
        assert(declspec(token) == -1);
        token = gentokens(2, TK_CHAR, TK_VOID);
        assert(declspec(token) == -1);
        token = gentokens(2, TK_VOID, TK_SIGNED);
        assert(declspec(token) == -1);
        token = gentokens(2, TK_SIGNED, TK_VOID);
        assert(declspec(token) == -1);
        token = gentokens(2, TK_SHORT, TK_VOID);
        assert(declspec(token) == -1);
}

void testchar(void) {
        Token *token = gentokens(1, TK_CHAR);
        assert(declspec(token) == TYPE_CHAR);
        token = gentokens(2, TK_CHAR, TK_INT);
        assert(declspec(token) == -1);
        token = gentokens(2, TK_INT, TK_CHAR, TK_FLOAT);
        assert(declspec(token) == -1);
}
