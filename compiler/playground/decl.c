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
        uint64_t t, lcnt;
        t = lcnt = 0;
        while (token->kind != TK_EOF) {
                switch (token->kind) {
                        case TK_VOID:
                                if (t && (t & (~(TYPE_EXTERN | TYPE_CONST)))) {
                                bterror:
                                        return error("too many basic types\n");
                                }
                                t |= TYPE_VOID;
                                break;
                        case TK_CHAR:
                                if (t & TYPE_BMASK) goto bterror;
                                t |= TYPE_CHAR;
                                break;
                        case TK_INT:
                                if (t & TYPE_BMASK) goto bterror;
                                t |= TYPE_INT;
                                break;
                        case TK_FLOAT:
                                if (t & TYPE_BMASK) goto bterror;
                                t |= TYPE_FLOAT;
                                break;
                        case TK_DOUBLE:
                                if (t & TYPE_BMASK) goto bterror;
                                t |= TYPE_DOUBLE;
                                break;
                        case TK_SHORT:
                                if (t & 0x30) {
                                sterror:
                                        return error("invalid specifiers\n");
                                }
                                t |= TYPE_SHORT;
                                break;
                        case TK_LONG:
                                if (t & TYPE_SHORT) goto sterror;
                                if ((t & TYPE_LONG) && (lcnt >= 2)) goto sterror;
                                t |= TYPE_LONG;
                                ++lcnt;
                                break;
                        case TK_UNSIGNED:
                                if (t & 0xc0) goto sterror;
                                t |= TYPE_UNSIGNED;
                                break;
                        case TK_SIGNED:
                                if (t & 0xc0) goto sterror;
                                t |= TYPE_SIGNED;
                                break;
                        case TK_TYPEDEF: t |= TYPE_TYPEDEF; break;
                        case TK_EXTERN: t |= TYPE_EXTERN; break;
                        case TK_STATIC: t |= TYPE_STATIC; break;
                        case TK_CONST: t |= TYPE_CONST; break;
                        case TK_INLINE: t |= TYPE_INLINE; break;
                        /* ignored */
                        case TK_AUTO:
                        case TK_REGISTER:
                        case TK_RESTRICT:
                        case TK_VOLATILE: break;
                        default: error("Invalid token kind: %d\n", token->kind);
                }
                token = token->next;
        }

        // ADDME: struct/union, enum

        // -------------- Ugly but works
        // void
        if ((t & TYPE_BMASK) == TYPE_VOID) {
                if (t & TYPE_SMASK) return error("invalid void type\n");
                return t;
        }

        // no 'signed/unsigned' specifiers
        if (!(t & 0xc0)) {
                if ((t & TYPE_BMASK) == TYPE_CHAR) {
                        // char             // unsigned
                        if (t & 0x30) return error("invalid short/long char\n");
                        return (t | TYPE_UNSIGNED);
                } else if ((t & TYPE_BMASK) == TYPE_INT || !(t & TYPE_BMASK)) {
                        // short            // signed
                        // long             // signed
                        // long long        // signed
                        if (!(t & TYPE_BMASK)) {
                                if (t & TYPE_SHORT)
                                        return (TYPE_SHORT | TYPE_INT | TYPE_SIGNED);
                                else if (t & TYPE_LONG)
                                        return (TYPE_LONG | TYPE_INT | TYPE_SIGNED);
                                else
                                        return error(
                                                "invalid no signed/unsigned with no base type\n");
                        }
                        // short int        // signed
                        // int              // signed
                        // long int         // signed
                        // long long int    // signed
                        return (t | TYPE_SIGNED);
                } else if ((t & TYPE_BMASK) == TYPE_DOUBLE) {
                        // double           // signed
                        // long double      // signed
                        if (t & TYPE_SHORT) return error("invalid short double\n");
                        return (t | TYPE_SIGNED);
                } else if ((t & TYPE_BMASK) == TYPE_FLOAT) {
                        // float            // signed
                        if (t & 0x30) return error("invalid short/long float\n");
                        return (t | TYPE_SIGNED);
                } else {
                        return error("invalid no signed/unsiged combination\n");
                }
        } else if (t & TYPE_SIGNED) {
                if ((t & TYPE_BMASK) == TYPE_CHAR) {
                        // signed char
                        if (t & 0x30) return error("invalid short/long char\n");
                        return t;
                } else if ((t & TYPE_BMASK) == TYPE_INT || !(t & TYPE_BMASK)) {
                        // signed
                        // signed short
                        // signed long
                        // signed long long
                        if (!(t & TYPE_BMASK)) {
                                if (t & TYPE_SHORT)
                                        return (TYPE_SHORT | TYPE_INT | TYPE_SIGNED);
                                else if (t & TYPE_LONG)
                                        return (TYPE_LONG | TYPE_INT | TYPE_SIGNED);
                                else
                                        return (TYPE_INT | TYPE_SIGNED);
                        }
                        // signed int
                        // signed short int
                        // signed long int
                        // signed long int int
                        return t;
                } else {
                        return error("invalid signed type combination\n");
                }
        } else {
                assert(t & TYPE_UNSIGNED);
                if ((t & TYPE_BMASK) == TYPE_CHAR) {
                        // unsigned char
                        if (t & 0x30) return error("invalid short/long char\n");
                        return t;
                } else if ((t & TYPE_BMASK) == TYPE_INT || !(t & TYPE_BMASK)) {
                        // unsigned
                        // unsigned short
                        // unsigned long
                        // unsigned long long
                        if (!(t & TYPE_BMASK)) {
                                if (t & TYPE_SHORT)
                                        return (TYPE_SHORT | TYPE_INT | TYPE_UNSIGNED);
                                else if (t & TYPE_LONG)
                                        return (TYPE_LONG | TYPE_INT | TYPE_UNSIGNED);
                                else
                                        return (TYPE_INT | TYPE_UNSIGNED);
                        }
                        // unsigned int
                        // unsigned short int
                        // unsigned long int
                        // unsigned long long int
                        return t;
                } else {
                        return error("invalid unsiged type combination\n");
                }
        }
        return error("unknown type\n");
}

void decl(void) {
        //
}

void test_declspec(void);

int main(void) {
        test_declspec();
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

void test_declspec(void) {
        // void
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
        token = gentokens(2, TK_SIGNED, TK_VOID);
        assert(declspec(token) == -1);
        token = gentokens(2, TK_LONG, TK_VOID);
        assert(declspec(token) == -1);

        // char
        token = gentokens(1, TK_CHAR);
        assert(declspec(token) == (TYPE_CHAR | TYPE_UNSIGNED));
        token = gentokens(2, TK_SIGNED, TK_CHAR);
        assert(declspec(token) == (TYPE_CHAR | TYPE_SIGNED));
        token = gentokens(2, TK_UNSIGNED, TK_CHAR);
        assert(declspec(token) == (TYPE_CHAR | TYPE_UNSIGNED));
        token = gentokens(2, TK_CHAR, TK_UNSIGNED);
        assert(declspec(token) == (TYPE_CHAR | TYPE_UNSIGNED));
        token = gentokens(2, TK_CHAR, TK_INT);
        assert(declspec(token) == -1);
        token = gentokens(2, TK_CHAR, TK_DOUBLE);
        assert(declspec(token) == -1);
        token = gentokens(3, TK_INT, TK_CHAR, TK_FLOAT);
        assert(declspec(token) == -1);

        // short
        token = gentokens(1, TK_SHORT);
        assert(declspec(token) == (TYPE_SHORT | TYPE_INT | TYPE_SIGNED));
        token = gentokens(2, TK_SIGNED, TK_SHORT);
        assert(declspec(token) == (TYPE_SHORT | TYPE_INT | TYPE_SIGNED));
        token = gentokens(2, TK_SHORT, TK_INT);
        assert(declspec(token) == (TYPE_SHORT | TYPE_INT | TYPE_SIGNED));
        token = gentokens(3, TK_SIGNED, TK_SHORT, TK_INT);
        assert(declspec(token) == (TYPE_SHORT | TYPE_INT | TYPE_SIGNED));
        token = gentokens(2, TK_UNSIGNED, TK_SHORT);
        assert(declspec(token) == (TYPE_SHORT | TYPE_UNSIGNED | TYPE_INT));
        token = gentokens(3, TK_UNSIGNED, TK_SHORT, TK_INT);
        assert(declspec(token) == (TYPE_UNSIGNED | TYPE_SHORT | TYPE_INT));
        token = gentokens(3, TK_SHORT, TK_LONG, TK_INT);
        assert(declspec(token) == -1);
        token = gentokens(2, TK_SHORT, TK_DOUBLE);
        assert(declspec(token) == -1);

        // signed
        token = gentokens(1, TK_INT);
        assert(declspec(token) == (TYPE_INT | TYPE_SIGNED));
        token = gentokens(1, TK_SIGNED);
        assert(declspec(token) == (TYPE_INT | TYPE_SIGNED));
        token = gentokens(2, TK_SIGNED, TK_INT);
        assert(declspec(token) == (TYPE_INT | TYPE_SIGNED));
        token = gentokens(1, TK_UNSIGNED);
        assert(declspec(token) == (TYPE_INT | TYPE_UNSIGNED));
        token = gentokens(2, TK_UNSIGNED, TK_INT);
        assert(declspec(token) == (TYPE_INT | TYPE_UNSIGNED));
        token = gentokens(3, TK_UNSIGNED, TK_SIGNED, TK_INT);
        assert(declspec(token) == -1);
        token = gentokens(2, TK_UNSIGNED, TK_VOID);
        assert(declspec(token) == -1);
        token = gentokens(2, TK_UNSIGNED, TK_FLOAT);
        assert(declspec(token) == -1);
        token = gentokens(2, TK_SIGNED, TK_DOUBLE);
        assert(declspec(token) == -1);

        // long
        token = gentokens(1, TK_LONG);
        assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_SIGNED));
        token = gentokens(2, TK_SIGNED, TK_LONG);
        assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_SIGNED));
        token = gentokens(2, TK_LONG, TK_INT);
        assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_SIGNED));
        token = gentokens(3, TK_SIGNED, TK_LONG, TK_INT);
        assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_SIGNED));
        token = gentokens(2, TK_UNSIGNED, TK_LONG);
        assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_UNSIGNED));
        token = gentokens(3, TK_UNSIGNED, TK_LONG, TK_INT);
        assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_UNSIGNED));
        token = gentokens(3, TK_UNSIGNED, TK_LONG, TK_CHAR);
        assert(declspec(token) == -1);
        token = gentokens(2, TK_LONG, TK_FLOAT);
        assert(declspec(token) == -1);

        // long long
        token = gentokens(2, TK_LONG, TK_LONG);
        assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_SIGNED));
        token = gentokens(3, TK_SIGNED, TK_LONG, TK_LONG);
        assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_SIGNED));
        token = gentokens(3, TK_LONG, TK_LONG, TK_INT);
        assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_SIGNED));
        token = gentokens(4, TK_SIGNED, TK_LONG, TK_LONG, TK_INT);
        assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_SIGNED));
        token = gentokens(3, TK_UNSIGNED, TK_LONG, TK_LONG);
        assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_UNSIGNED));
        token = gentokens(4, TK_UNSIGNED, TK_LONG, TK_LONG, TK_INT);
        assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_UNSIGNED));
        token = gentokens(4, TK_LONG, TK_LONG, TK_LONG, TK_INT);
        assert(declspec(token) == -1);

        // float
        token = gentokens(1, TK_FLOAT);
        assert(declspec(token) == (TYPE_FLOAT | TYPE_SIGNED));
        token = gentokens(2, TK_FLOAT, TK_SIGNED);
        assert(declspec(token) == -1);
        token = gentokens(2, TK_FLOAT, TK_UNSIGNED);
        assert(declspec(token) == -1);
        token = gentokens(2, TK_SHORT, TK_FLOAT);
        assert(declspec(token) == -1);
        token = gentokens(2, TK_LONG, TK_FLOAT);
        assert(declspec(token) == -1);

        // double
        token = gentokens(1, TK_DOUBLE);
        assert(declspec(token) == (TYPE_DOUBLE | TYPE_SIGNED));
        token = gentokens(2, TK_LONG, TK_DOUBLE);
        assert(declspec(token) == (TYPE_LONG | TYPE_DOUBLE | TYPE_SIGNED));
        token = gentokens(2, TK_SHORT, TK_DOUBLE);
        assert(declspec(token) == -1);

        // const
        token = gentokens(2, TK_CONST, TK_VOID);
        assert(declspec(token) == (TYPE_CONST | TYPE_VOID));
        token = gentokens(2, TK_CONST, TK_CHAR);
        assert(declspec(token) == (TYPE_CONST | TYPE_UNSIGNED | TYPE_CHAR));
        token = gentokens(2, TK_CONST, TK_INT);
        assert(declspec(token) == (TYPE_CONST | TYPE_SIGNED | TYPE_INT));
        token = gentokens(2, TK_CONST, TK_FLOAT);
        assert(declspec(token) == (TYPE_CONST | TYPE_SIGNED | TYPE_FLOAT));

        // extern
        token = gentokens(2, TK_EXTERN, TK_VOID);
        assert(declspec(token) == (TYPE_EXTERN | TYPE_VOID));
        token = gentokens(2, TK_EXTERN, TK_CHAR);
        assert(declspec(token) == (TYPE_EXTERN | TYPE_CHAR | TYPE_UNSIGNED));
        token = gentokens(2, TK_EXTERN, TK_INT);
        assert(declspec(token) == (TYPE_EXTERN | TYPE_INT | TYPE_SIGNED));
        token = gentokens(2, TK_EXTERN, TK_FLOAT);
        assert(declspec(token) == (TYPE_EXTERN | TYPE_FLOAT | TYPE_SIGNED));
        token = gentokens(2, TK_EXTERN, TK_DOUBLE);
        assert(declspec(token) == (TYPE_EXTERN | TYPE_DOUBLE | TYPE_SIGNED));
}
