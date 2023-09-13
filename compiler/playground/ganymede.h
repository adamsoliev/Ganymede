#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Token Token;

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


// basic types
#define TYPE_BMASK        0xf // 0000,1111
// #define TYPE_UNUSED       0x0	// 0000,0000 
#define TYPE_VOID         0x1	// 0000,0001
#define TYPE_CHAR         0x2	// 0000,0010
#define TYPE_INT          0x3	// 0000,0011
#define TYPE_FLOAT        0x4	// 0000,0100
#define TYPE_DOUBLE       0x5	// 0000,0101
#define TYPE_ARRAY        0x6	// 0000,0110	
#define TYPE_FUNC         0x7	// 0000,0111		
#define TYPE_PTR          0x8	// 0000,1000		
#define TYPE_STRUCT       0x9	// 0000,1001	
#define TYPE_UNION        0xa	// 0000,1010		
#define TYPE_ENUM         0xb	// 0000,1011		
// #define TYPE_UNUSED       0xc	// 0000,1100
// #define TYPE_UNUSED       0xd	// 0000,1101
// #define TYPE_UNUSED       0xe	// 0000,1110
// #define TYPE_UNUSED       0xf	// 0000,1111

#define TYPE_SHORT        0x10 // 0001,0000
#define TYPE_LONG         0x20 // 0010,0000 // long = long long
#define TYPE_UNSIGNED     0x40 // 0100,0000
#define TYPE_SIGNED       0x80 // 1000,0000

// 1000,0011

// clang-format on

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
