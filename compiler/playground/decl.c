// #pragma clang diagnostic ignored "-Wgnu-empty-initializer"
// #pragma clang diagnostic ignored "-Wgnu-binary-literal"

// Token *token;

// int error(char *fmt, ...) {
//         va_list args;
//         va_start(args, fmt);
//         vprintf(fmt, args);
//         va_end(args);
//         return -1;
// }

// Token *new_token(enum TokenKind kind) {
//         Token *tok = calloc(1, sizeof(Token));
//         tok->kind = kind;
//         return tok;
// }

// uint64_t declspec(Token *token) {
//         uint64_t t, lcnt, end;
//         t = lcnt = end = 0;
//         while (token->kind != TK_EOF) {
//                 switch (token->kind) {
//                         case TK_VOID:
//                                 if (t && (t & (~(TYPE_EXTERN | TYPE_CONST)))) {
//                                 bterror:
//                                         return error("too many basic types\n");
//                                 }
//                                 t |= TYPE_VOID;
//                                 break;
//                         case TK_CHAR:
//                                 if (t & TYPE_BMASK) goto bterror;
//                                 t |= TYPE_CHAR;
//                                 break;
//                         case TK_INT:
//                                 if (t & TYPE_BMASK) goto bterror;
//                                 t |= TYPE_INT;
//                                 break;
//                         case TK_FLOAT:
//                                 if (t & TYPE_BMASK) goto bterror;
//                                 t |= TYPE_FLOAT;
//                                 break;
//                         case TK_DOUBLE:
//                                 if (t & TYPE_BMASK) goto bterror;
//                                 t |= TYPE_DOUBLE;
//                                 break;
//                         /* ingored for now */
//                         case TK_STRUCT:
//                         case TK_UNION:
//                         case TK_ENUM: break;
//                         case TK_SHORT:
//                                 if (t & 0x30) {
//                                 sterror:
//                                         return error("invalid specifiers\n");
//                                 }
//                                 t |= TYPE_SHORT;
//                                 break;
//                         case TK_LONG:
//                                 if (t & TYPE_SHORT) goto sterror;
//                                 if ((t & TYPE_LONG) && (lcnt >= 2)) goto sterror;
//                                 t |= TYPE_LONG;
//                                 ++lcnt;
//                                 break;
//                         case TK_UNSIGNED:
//                                 if (t & 0xc0) goto sterror;
//                                 t |= TYPE_UNSIGNED;
//                                 break;
//                         case TK_SIGNED:
//                                 if (t & 0xc0) goto sterror;
//                                 t |= TYPE_SIGNED;
//                                 break;
//                         case TK_TYPEDEF:  // IMPLEMENTME
//                                 if (t & TYPE_SGMASK) {
//                                 sgerror:
//                                         // FIXME: we can count bits in the SGMARK region after the loop?
//                                         return error("more than one storage-class specifier\n");
//                                 }
//                                 t |= TYPE_TYPEDEF;
//                                 break;
//                         case TK_EXTERN:
//                                 if (t & TYPE_SGMASK) goto sgerror;
//                                 t |= TYPE_EXTERN;
//                                 break;
//                         case TK_STATIC:  // IMPLEMENTME
//                                 if (t & TYPE_SGMASK) goto sgerror;
//                                 t |= TYPE_STATIC;
//                                 break;
//                         case TK_CONST: t |= TYPE_CONST; break;
//                         case TK_INLINE:
//                                 t |= TYPE_INLINE;
//                                 break;  // IMPLEMENTME
//                         /* ignored */
//                         case TK_AUTO:
//                         case TK_REGISTER:
//                         case TK_RESTRICT:
//                         case TK_VOLATILE: break;
//                         default: end = 1; break;
//                 }
//                 if (end) {
//                         if (token->kind == TK_IDENT || token->kind == TK_STAR ||
//                             token->kind == TK_OPAR) {
//                                 break;
//                         }
//                         error("unknown declaration specifiers: %d\n", token->kind);
//                 }
//                 token = token->next;
//         }

//         uint64_t basety = (t & TYPE_BMASK);
//         if (!basety) {
//                 basety = TYPE_INT;
//                 t |= TYPE_INT;
//         }

//         // void
//         if (basety == TYPE_VOID) {
//                 if (t & TYPE_SMASK) return error("invalid void type\n");
//                 return t;
//         }

//         // no 'signed/unsigned' specifiers
//         if (!(t & 0xc0)) {
//                 if (basety == TYPE_CHAR) {
//                         // char             // unsigned
//                         if (t & 0x30) return error("invalid short/long char\n");
//                         return (t | TYPE_UNSIGNED);
//                 } else if (basety == TYPE_INT) {
//                         // (short int | int | long int | long long int) // signed
//                         return (t | TYPE_SIGNED);
//                 } else if (basety == TYPE_DOUBLE) {
//                         // (double | long double)       // signed
//                         if (t & TYPE_SHORT) return error("invalid short double\n");
//                         return (t | TYPE_SIGNED);
//                 } else if (basety == TYPE_FLOAT) {
//                         // float                        // signed
//                         if (t & 0x30) return error("invalid short/long float\n");
//                         return (t | TYPE_SIGNED);
//                 } else {
//                         return error("invalid no signed/unsiged combination\n");
//                 }
//         }
//         if (basety == TYPE_CHAR || basety == TYPE_INT) {
//                 if (basety == TYPE_CHAR && t & 0x30) return error("invalid short/long char\n");
//                 // signed/unsigned (char | int | short int | long int | long int int)
//                 return t;
//         }
//         return error("unknown type\n");
// }

// /*

//     declaration ::=
//         // declaration-specifiers init-declarator-list ;
//         // declaration-specifiers declarator-list ;
//         declaration-specifiers direct-declarator ;

//     // init-declarator-list ::=
//     //     init-declarator
//     //     init-declarator-list , init-declarator

//     // init-declarator ::=
//         declarator
//         // declarator = initializer

//     declarator
//         // pointer? direct-declarator
//         direct-declarator

//     direct-declarator:
//         identifier
//         ( declarator )
//         direct-declarator [ type-qualifier-list? assignment-expressionopt ]
//         direct-declarator [ static type-qualifier-list? assignment-expression ]
//         direct-declarator [ type-qualifier-list static assignment-expression ]
//         direct-declarator [ type-qualifier-list? * ]
//         direct-declarator ( parameter-type-list )
//         direct-declarator ( identifier-listopt )            // ignored: old style func defition
// */

// ExcDecl *decl(uint64_t type, Token *token) {
//         // pointers
//         if (token->kind == TK_IDENT) {
//         } else if (token->kind == TK_OPAR) {
//                 // abstract
//                 // function
//         } else if (token->kind == TK_OBR) {
//         } else {
//                 error("invalid declaration\n");
//         }

//         /*
//             pointers
//             if ident
//                  array
//                     [expression]    - fixed size array
//                     []              - only usable in certain scopes
//                     [*]             - only usable in func prototype scope
//                     ------------
//                     float fa[11]
//                     int a[n][6][m + 2]

//                  func
//                     (params)
//                     (params, ...)
//                     ------------
//                     int func(void)
//                     int sum(int a, int b)
//                     int product(int, int)
//                     int error(int cnt, ...)
//                     void addscalar(int n, int m, double a[n][n*m+300], double x)
//                     double maximum(int n, int m, double a[n][m]);
//                     double maximum(int n, int m, double a[*][*]);
//                     double maximum(int n, int m, double a[ ][*]);
//                     double maximum(int n, int m, double a[ ][m]);

//                  ident
//                     ------------
//                     int num, char ch, float x, double y
//             abstract
//         */
//         return NULL;
// }

// void parse(void) {
//         //
//         uint64_t type = declspec(token);
//         ExcDecl *declaration = decl(type, token);
// }

// void test_declspec(void);

// int main(void) {
//         char *program = "int main() { int var = 23; return 0; }";
//         test_declspec();
//         return 0;
// }

// Token *gentokens(int size, ...) {
//         va_list args;
//         va_start(args, size);
//         Token token = {};
//         Token *cur = &token;
//         for (int i = 0; i < size; i++) {
//                 cur = cur->next = new_token(va_arg(args, enum TokenKind));
//         }
//         // cur = cur->next = new_token(TK_IDENT);

//         va_end(args);
//         return token.next;
// }

// void test_declspec(void) {
//         // void
//         Token *token = gentokens(2, TK_VOID, TK_IDENT);
//         assert(declspec(token) == TYPE_VOID);
//         token = gentokens(3, TK_VOID, TK_INT, TK_IDENT);
//         assert(declspec(token) == -1);
//         token = gentokens(3, TK_CHAR, TK_VOID, TK_IDENT);
//         assert(declspec(token) == -1);
//         token = gentokens(3, TK_VOID, TK_SIGNED, TK_IDENT);
//         assert(declspec(token) == -1);
//         token = gentokens(3, TK_SIGNED, TK_VOID, TK_IDENT);
//         assert(declspec(token) == -1);
//         token = gentokens(3, TK_SHORT, TK_VOID, TK_IDENT);
//         assert(declspec(token) == -1);
//         token = gentokens(3, TK_SIGNED, TK_VOID, TK_IDENT);
//         assert(declspec(token) == -1);
//         token = gentokens(3, TK_LONG, TK_VOID, TK_IDENT);
//         assert(declspec(token) == -1);

//         // char
//         token = gentokens(2, TK_CHAR, TK_IDENT);
//         assert(declspec(token) == (TYPE_CHAR | TYPE_UNSIGNED));
//         token = gentokens(3, TK_SIGNED, TK_CHAR, TK_IDENT);
//         assert(declspec(token) == (TYPE_CHAR | TYPE_SIGNED));
//         token = gentokens(3, TK_UNSIGNED, TK_CHAR, TK_IDENT);
//         assert(declspec(token) == (TYPE_CHAR | TYPE_UNSIGNED));
//         token = gentokens(3, TK_CHAR, TK_UNSIGNED, TK_IDENT);
//         assert(declspec(token) == (TYPE_CHAR | TYPE_UNSIGNED));
//         token = gentokens(3, TK_CHAR, TK_INT, TK_IDENT);
//         assert(declspec(token) == -1);
//         token = gentokens(3, TK_CHAR, TK_DOUBLE, TK_IDENT);
//         assert(declspec(token) == -1);
//         token = gentokens(4, TK_INT, TK_CHAR, TK_FLOAT, TK_IDENT);
//         assert(declspec(token) == -1);

//         // short
//         token = gentokens(2, TK_SHORT, TK_IDENT);
//         assert(declspec(token) == (TYPE_SHORT | TYPE_INT | TYPE_SIGNED));
//         token = gentokens(3, TK_SIGNED, TK_SHORT, TK_IDENT);
//         assert(declspec(token) == (TYPE_SHORT | TYPE_INT | TYPE_SIGNED));
//         token = gentokens(3, TK_SHORT, TK_INT, TK_IDENT);
//         assert(declspec(token) == (TYPE_SHORT | TYPE_INT | TYPE_SIGNED));
//         token = gentokens(4, TK_SIGNED, TK_SHORT, TK_INT, TK_IDENT);
//         assert(declspec(token) == (TYPE_SHORT | TYPE_INT | TYPE_SIGNED));
//         token = gentokens(3, TK_UNSIGNED, TK_SHORT, TK_IDENT);
//         assert(declspec(token) == (TYPE_SHORT | TYPE_UNSIGNED | TYPE_INT));
//         token = gentokens(4, TK_UNSIGNED, TK_SHORT, TK_INT, TK_IDENT);
//         assert(declspec(token) == (TYPE_UNSIGNED | TYPE_SHORT | TYPE_INT));
//         token = gentokens(4, TK_SHORT, TK_LONG, TK_INT, TK_IDENT);
//         assert(declspec(token) == -1);
//         token = gentokens(3, TK_SHORT, TK_DOUBLE, TK_IDENT);
//         assert(declspec(token) == -1);

//         // signed
//         token = gentokens(2, TK_INT, TK_IDENT);
//         assert(declspec(token) == (TYPE_INT | TYPE_SIGNED));
//         token = gentokens(2, TK_SIGNED, TK_IDENT);
//         assert(declspec(token) == (TYPE_INT | TYPE_SIGNED));
//         token = gentokens(3, TK_SIGNED, TK_INT, TK_IDENT);
//         assert(declspec(token) == (TYPE_INT | TYPE_SIGNED));
//         token = gentokens(2, TK_UNSIGNED, TK_IDENT);
//         assert(declspec(token) == (TYPE_INT | TYPE_UNSIGNED));
//         token = gentokens(3, TK_UNSIGNED, TK_INT, TK_IDENT);
//         assert(declspec(token) == (TYPE_INT | TYPE_UNSIGNED));
//         token = gentokens(4, TK_UNSIGNED, TK_SIGNED, TK_INT, TK_IDENT);
//         assert(declspec(token) == -1);
//         token = gentokens(3, TK_UNSIGNED, TK_VOID, TK_IDENT);
//         assert(declspec(token) == -1);
//         token = gentokens(3, TK_UNSIGNED, TK_FLOAT, TK_IDENT);
//         assert(declspec(token) == -1);
//         token = gentokens(3, TK_SIGNED, TK_DOUBLE, TK_IDENT);
//         assert(declspec(token) == -1);

//         // long
//         token = gentokens(2, TK_LONG, TK_IDENT);
//         assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_SIGNED));
//         token = gentokens(3, TK_SIGNED, TK_LONG, TK_IDENT);
//         assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_SIGNED));
//         token = gentokens(3, TK_LONG, TK_INT, TK_IDENT);
//         assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_SIGNED));
//         token = gentokens(4, TK_SIGNED, TK_LONG, TK_INT, TK_IDENT);
//         assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_SIGNED));
//         token = gentokens(3, TK_UNSIGNED, TK_LONG, TK_IDENT);
//         assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_UNSIGNED));
//         token = gentokens(4, TK_UNSIGNED, TK_LONG, TK_INT, TK_IDENT);
//         assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_UNSIGNED));
//         token = gentokens(4, TK_UNSIGNED, TK_LONG, TK_CHAR, TK_IDENT);
//         assert(declspec(token) == -1);
//         token = gentokens(3, TK_LONG, TK_FLOAT, TK_IDENT);
//         assert(declspec(token) == -1);

//         // long long
//         token = gentokens(3, TK_LONG, TK_LONG, TK_IDENT);
//         assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_SIGNED));
//         token = gentokens(4, TK_SIGNED, TK_LONG, TK_LONG, TK_IDENT);
//         assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_SIGNED));
//         token = gentokens(4, TK_LONG, TK_LONG, TK_INT, TK_IDENT);
//         assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_SIGNED));
//         token = gentokens(5, TK_SIGNED, TK_LONG, TK_LONG, TK_INT, TK_IDENT);
//         assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_SIGNED));
//         token = gentokens(4, TK_UNSIGNED, TK_LONG, TK_LONG, TK_IDENT);
//         assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_UNSIGNED));
//         token = gentokens(5, TK_UNSIGNED, TK_LONG, TK_LONG, TK_INT, TK_IDENT);
//         assert(declspec(token) == (TYPE_INT | TYPE_LONG | TYPE_UNSIGNED));
//         token = gentokens(5, TK_LONG, TK_LONG, TK_LONG, TK_INT, TK_IDENT);
//         assert(declspec(token) == -1);

//         // float
//         token = gentokens(2, TK_FLOAT, TK_IDENT);
//         assert(declspec(token) == (TYPE_FLOAT | TYPE_SIGNED));
//         token = gentokens(3, TK_FLOAT, TK_SIGNED, TK_IDENT);
//         assert(declspec(token) == -1);
//         token = gentokens(3, TK_FLOAT, TK_UNSIGNED, TK_IDENT);
//         assert(declspec(token) == -1);
//         token = gentokens(3, TK_SHORT, TK_FLOAT, TK_IDENT);
//         assert(declspec(token) == -1);
//         token = gentokens(3, TK_LONG, TK_FLOAT, TK_IDENT);
//         assert(declspec(token) == -1);

//         // double
//         token = gentokens(2, TK_DOUBLE, TK_IDENT);
//         assert(declspec(token) == (TYPE_DOUBLE | TYPE_SIGNED));
//         token = gentokens(3, TK_LONG, TK_DOUBLE, TK_IDENT);
//         assert(declspec(token) == (TYPE_LONG | TYPE_DOUBLE | TYPE_SIGNED));
//         token = gentokens(3, TK_SHORT, TK_DOUBLE, TK_IDENT);
//         assert(declspec(token) == -1);

//         // const
//         token = gentokens(3, TK_CONST, TK_VOID, TK_IDENT);
//         assert(declspec(token) == (TYPE_CONST | TYPE_VOID));
//         token = gentokens(3, TK_CONST, TK_CHAR, TK_IDENT);
//         assert(declspec(token) == (TYPE_CONST | TYPE_UNSIGNED | TYPE_CHAR));
//         token = gentokens(3, TK_CONST, TK_INT, TK_IDENT);
//         assert(declspec(token) == (TYPE_CONST | TYPE_SIGNED | TYPE_INT));
//         token = gentokens(3, TK_CONST, TK_FLOAT, TK_IDENT);
//         assert(declspec(token) == (TYPE_CONST | TYPE_SIGNED | TYPE_FLOAT));

//         // extern
//         token = gentokens(3, TK_EXTERN, TK_VOID, TK_IDENT);
//         assert(declspec(token) == (TYPE_EXTERN | TYPE_VOID));
//         token = gentokens(3, TK_EXTERN, TK_STATIC, TK_VOID, TK_IDENT);
//         assert(declspec(token) == -1);
//         token = gentokens(3, TK_EXTERN, TK_CHAR, TK_IDENT);
//         assert(declspec(token) == (TYPE_EXTERN | TYPE_CHAR | TYPE_UNSIGNED));
//         token = gentokens(3, TK_EXTERN, TK_INT, TK_IDENT);
//         assert(declspec(token) == (TYPE_EXTERN | TYPE_INT | TYPE_SIGNED));
//         token = gentokens(3, TK_EXTERN, TK_FLOAT, TK_IDENT);
//         assert(declspec(token) == (TYPE_EXTERN | TYPE_FLOAT | TYPE_SIGNED));
//         token = gentokens(3, TK_EXTERN, TK_DOUBLE, TK_IDENT);
//         assert(declspec(token) == (TYPE_EXTERN | TYPE_DOUBLE | TYPE_SIGNED));

//         // typedef
//         // static
//         // inline
// }

#include "ganymede.h"

/* given a uint64_t token, exctract its respective field */
#define TGETKIND(num) ((num) & 0xFF)
#define TGETCOL(num) (((num) >> 8) & 0xFF)
#define TGETROW(num) (((num) >> 16) & 0xFFFF)
#define TGETIFN(num) (((num) >> 32) & 0xFFFF)
#define TSET(kind, col, row, ifn)                                 \
        ((uint64_t)kind & 0xFF) | (((uint64_t)col & 0xFF) << 8) | \
                (((uint64_t)row & 0xFFFF) << 16) | (((uint64_t)ifn & 0xFFFF) << 32);

void test(uint64_t kind, uint64_t col, uint64_t row, uint64_t ifn) {
        uint64_t token = TSET(kind, col, row, ifn);
        assert(TGETKIND(token) == (kind & 0xFF));
        assert(TGETCOL(token) == (col & 0xFF));
        assert(TGETROW(token) == (row & 0xFFFF));
        assert(TGETIFN(token) == (ifn & 0xFFFF));
}

void test1(uint64_t kind, uint64_t col, uint64_t row, uint64_t ifn) {
        uint64_t token = TSET(kind, 0, 0, 0);
        token |= TSET(0, col, 0, 0);
        token |= TSET(0, 0, row, 0);
        token |= TSET(0, 0, 0, ifn);
        assert(TGETKIND(token) == (kind & 0xFF));
        assert(TGETCOL(token) == (col & 0xFF));
        assert(TGETROW(token) == (row & 0xFFFF));
        assert(TGETIFN(token) == (ifn & 0xFFFF));
}

int main(void) {
        // Test 1: Basic Test Case
        test(1, 2, 3, 4);
        test1(1, 2, 3, 4);

        // Test 2: Test with Maximum Values
        test(255, 255, 65535, 65535);
        test(257, 257, 65537, 65537);
        test1(255, 255, 65535, 65535);
        test1(257, 257, 65537, 65537);

        // Test 3: Test with Minimum Values
        test(0, 0, 0, 0);
        test1(0, 0, 0, 0);

        // Test 4: Test with Different Kind
        test(42, 7, 128, 500);
        test1(42, 7, 128, 500);

        // Test 5: Test with Random Values
        srand(time(NULL));
        uint64_t random_kind = rand() % 256;
        uint64_t random_col = rand() % 256;
        uint64_t random_row = rand() % 65536;
        uint64_t random_ifn = rand() % 65536;
        test(random_kind, random_col, random_row, random_ifn);
        test1(random_kind, random_col, random_row, random_ifn);
        return 0;
}
