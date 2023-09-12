#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
#define TYPE_BMASK        0b1111
#define TYPE_VOID         0b0000
#define TYPE_CHAR         0b0001
#define TYPE_SHORT        0b0010
#define TYPE_INT          0b0011
#define TYPE_LONG         0b0100
#define TYPE_LLONG        0b0101
#define TYPE_FLOAT        0b0110
#define TYPE_DOUBLE       0b0111
#define TYPE_UNSIGNED     0b1000
#define TYPE_SIGNED       0b1001

/*
#define TYPE_UNUSED       0b1010
#define TYPE_UNUSED       0b1011
#define TYPE_UNUSED       0b1100
#define TYPE_UNUSED       0b1101
#define TYPE_UNUSED       0b1110
#define TYPE_UNUSED       0b1111
*/
/* Complex isn't supported */

#define TYPE_ARRAY        0b00010000		
#define TYPE_FUNC         0b00010001		
#define TYPE_PTR          0b00010010		
#define TYPE_STRUCT       0b00010011		
#define TYPE_UNION        0b00010100		
#define TYPE_ENUM         0b00010101		
/*
#define TYPE_UNUSED       0b00010110		
#define TYPE_UNUSED       0b00010111		
#define TYPE_UNUSED       0b00011000		
#define TYPE_UNUSED       0b00011001		
#define TYPE_UNUSED       0b00011010		
#define TYPE_UNUSED       0b00011011		
#define TYPE_UNUSED       0b00011100		
#define TYPE_UNUSED       0b00011101		
#define TYPE_UNUSED       0b00011110		
#define TYPE_UNUSED       0b00011111		
*/
// clang-format on
