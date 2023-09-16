#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Token Token;
typedef struct ExcDecl ExcDecl;

// clang-format off
// https://stackoverflow.com/questions/111928/is-there-a-printf-converter-to-print-in-binary-format/25108449#25108449
#define BB_P8 "%c%c%c%c,%c%c%c%c" // byte to binary pattern
#define BB8(byte)                  \
  ((byte) & 0x80 ? '1' : '0'),     \
      ((byte) & 0x40 ? '1' : '0'), \
      ((byte) & 0x20 ? '1' : '0'), \
      ((byte) & 0x10 ? '1' : '0'), \
      ((byte) & 0x08 ? '1' : '0'), \
      ((byte) & 0x04 ? '1' : '0'), \
      ((byte) & 0x02 ? '1' : '0'), \
      ((byte) & 0x01 ? '1' : '0')

#define BB_P16 \
  BB_P8 "," BB_P8
#define BB16(i) \
  BB8((i) >> 8), BB8(i)
#define BB_P32 \
  BB_P16 "," BB_P16
#define BB32(i) \
  BB16((i) >> 16), BB16(i)
#define BB_P64 \
  BB_P32 "," BB_P32
#define BB64(i) \
  BB32((i) >> 32), BB32(i)

                      // 0xff,ff,ff,ff,ff,ff,ff,ff 
                      // 0x00,00,00,00,00,00,00,00 
// basic types
#define TYPE_BMASK        0x000000000000000f  // 1111
// #define TYPE_UNUSED       0x0000000000000000	// 0000 
#define TYPE_VOID         0x0000000000000001	// 0001
#define TYPE_CHAR         0x0000000000000002	// 0010
#define TYPE_INT          0x0000000000000003	// 0011
#define TYPE_FLOAT        0x0000000000000004	// 0100
#define TYPE_DOUBLE       0x0000000000000005	// 0101
#define TYPE_STRUCT       0x0000000000000006	// 0110	
#define TYPE_UNION        0x0000000000000007	// 0111		
#define TYPE_ENUM         0x0000000000000008	// 1000		
// #define TYPE_UNUSED       0x0000000000000009	// 1001	
// #define TYPE_UNUSED       0x000000000000000a	// 1010		
// #define TYPE_UNUSED       0x000000000000000b	// 1011		
// #define TYPE_UNUSED       0x000000000000000c	// 1100
// #define TYPE_UNUSED       0x000000000000000d	// 1101
// #define TYPE_UNUSED       0x000000000000000e	// 1110
// #define TYPE_UNUSED       0x000000000000000f	// 1111

#define TYPE_SMASK        0x00000000000000f0 // 1111,0000
#define TYPE_SHORT        0x0000000000000010 // 0001,0000
#define TYPE_LONG         0x0000000000000020 // 0010,0000 // long = long long
#define TYPE_UNSIGNED     0x0000000000000040 // 0100,0000
#define TYPE_SIGNED       0x0000000000000080 // 1000,0000

/* storage */
#define TYPE_SGMASK       0x0000000000000f00
#define TYPE_TYPEDEF      0x0000000000000100
#define TYPE_EXTERN       0x0000000000000200
#define TYPE_STATIC       0x0000000000000400
// #define TYPE_UNUSED       0x0000000000000800

#define TYPE_CONST        0x0000000000001000
#define TYPE_INLINE       0x0000000000002000
// #define TYPE_UNUSED       0x0000000000004000
// #define TYPE_UNUSED       0x0000000000008000

// clang-format on

/* ------------ scan ------------ */
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
        TK_TYPEDEF,
        TK_EXTERN,
        TK_STATIC,
        TK_CONST,
        TK_INLINE,
        TK_AUTO,
        TK_REGISTER,
        TK_RESTRICT,
        TK_VOLATILE,
        TK_STRUCT,
        TK_UNION,
        TK_ENUM,
        TK_IDENT,
        TK_SCOLON,
        TK_OPAR,
        TK_CPAR,
        TK_OBR,   // [
        TK_CBR,   // ]
        TK_OBRC,  // {
        TK_CBRC,  // }
        TK_STAR,  // *
        TK_EOF,
};

struct Token {
        enum TokenKind kind;
        Token *next;
};

/* ------------ parse ------------ */
struct ExtDecl {
        uint64_t type;
        uint64_t hash;  // index to symbol table
};
