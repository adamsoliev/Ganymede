#include "ganymede.h"

enum {
        BLANK = 01,
        NEWLINE = 02,
        LETTER = 04,
        DIGIT = 010,
        HEX = 020,
        OTHER = 040
};

static unsigned char map[256] = {
        /* 000 nul */ 0,
        /* 001 soh */ 0,
        /* 002 stx */ 0,
        /* 003 etx */ 0,
        /* 004 eot */ 0,
        /* 005 enq */ 0,
        /* 006 ack */ 0,
        /* 007 bel */ 0,
        /* 010 bs  */ 0,
        /* 011 ht  */ BLANK,
        /* 012 nl  */ NEWLINE,
        /* 013 vt  */ BLANK,
        /* 014 ff  */ BLANK,
        /* 015 cr  */ 0,
        /* 016 so  */ 0,
        /* 017 si  */ 0,
        /* 020 dle */ 0,
        /* 021 dc1 */ 0,
        /* 022 dc2 */ 0,
        /* 023 dc3 */ 0,
        /* 024 dc4 */ 0,
        /* 025 nak */ 0,
        /* 026 syn */ 0,
        /* 027 etb */ 0,
        /* 030 can */ 0,
        /* 031 em  */ 0,
        /* 032 sub */ 0,
        /* 033 esc */ 0,
        /* 034 fs  */ 0,
        /* 035 gs  */ 0,
        /* 036 rs  */ 0,
        /* 037 us  */ 0,
        /* 040 sp  */ BLANK,
        /* 041 !   */ OTHER,
        /* 042 "   */ OTHER,
        /* 043 #   */ OTHER,
        /* 044 $   */ 0,
        /* 045 %   */ OTHER,
        /* 046 &   */ OTHER,
        /* 047 '   */ OTHER,
        /* 050 (   */ OTHER,
        /* 051 )   */ OTHER,
        /* 052 *   */ OTHER,
        /* 053 +   */ OTHER,
        /* 054 ,   */ OTHER,
        /* 055 -   */ OTHER,
        /* 056 .   */ OTHER,
        /* 057 /   */ OTHER,
        /* 060 0   */ DIGIT,
        /* 061 1   */ DIGIT,
        /* 062 2   */ DIGIT,
        /* 063 3   */ DIGIT,
        /* 064 4   */ DIGIT,
        /* 065 5   */ DIGIT,
        /* 066 6   */ DIGIT,
        /* 067 7   */ DIGIT,
        /* 070 8   */ DIGIT,
        /* 071 9   */ DIGIT,
        /* 072 :   */ OTHER,
        /* 073 ;   */ OTHER,
        /* 074 <   */ OTHER,
        /* 075 =   */ OTHER,
        /* 076 >   */ OTHER,
        /* 077 ?   */ OTHER,
        /* 100 @   */ 0,
        /* 101 A   */ LETTER | HEX,
        /* 102 B   */ LETTER | HEX,
        /* 103 C   */ LETTER | HEX,
        /* 104 D   */ LETTER | HEX,
        /* 105 E   */ LETTER | HEX,
        /* 106 F   */ LETTER | HEX,
        /* 107 G   */ LETTER,
        /* 110 H   */ LETTER,
        /* 111 I   */ LETTER,
        /* 112 J   */ LETTER,
        /* 113 K   */ LETTER,
        /* 114 L   */ LETTER,
        /* 115 M   */ LETTER,
        /* 116 N   */ LETTER,
        /* 117 O   */ LETTER,
        /* 120 P   */ LETTER,
        /* 121 Q   */ LETTER,
        /* 122 R   */ LETTER,
        /* 123 S   */ LETTER,
        /* 124 T   */ LETTER,
        /* 125 U   */ LETTER,
        /* 126 V   */ LETTER,
        /* 127 W   */ LETTER,
        /* 130 X   */ LETTER,
        /* 131 Y   */ LETTER,
        /* 132 Z   */ LETTER,
        /* 133 [   */ OTHER,
        /* 134 \   */ OTHER,
        /* 135 ]   */ OTHER,
        /* 136 ^   */ OTHER,
        /* 137 _   */ LETTER,
        /* 140 `   */ 0,
        /* 141 a   */ LETTER | HEX,
        /* 142 b   */ LETTER | HEX,
        /* 143 c   */ LETTER | HEX,
        /* 144 d   */ LETTER | HEX,
        /* 145 e   */ LETTER | HEX,
        /* 146 f   */ LETTER | HEX,
        /* 147 g   */ LETTER,
        /* 150 h   */ LETTER,
        /* 151 i   */ LETTER,
        /* 152 j   */ LETTER,
        /* 153 k   */ LETTER,
        /* 154 l   */ LETTER,
        /* 155 m   */ LETTER,
        /* 156 n   */ LETTER,
        /* 157 o   */ LETTER,
        /* 160 p   */ LETTER,
        /* 161 q   */ LETTER,
        /* 162 r   */ LETTER,
        /* 163 s   */ LETTER,
        /* 164 t   */ LETTER,
        /* 165 u   */ LETTER,
        /* 166 v   */ LETTER,
        /* 167 w   */ LETTER,
        /* 170 x   */ LETTER,
        /* 171 y   */ LETTER,
        /* 172 z   */ LETTER,
        /* 173 {   */ OTHER,
        /* 174 |   */ OTHER,
        /* 175 }   */ OTHER,
        /* 176 ~   */ OTHER,
};

static struct Token *new_token(enum TokenKind kind, char *start, int len) {
        struct Token *token = calloc(1, sizeof(struct Token));
        token->kind = kind;
        token->start = start;
        token->len = len;
        return token;
}

static struct Token *ck;

void error(char *fmt, ...);
void printTokens(struct Token *head, FILE *outfile);
struct Token *scan(char *cp);

static void printTokenKind(enum TokenKind kind, FILE *outfile);
static void floatconst(char **rcp);

struct Token *scan(char *cp) {
#define CHECK_PUNCTUATION(op, token, incr) \
        if (*rcp == op) {                  \
                HANDLE_TOKEN(token, incr); \
        }

#define HANDLE_TOKEN(kind, length)     \
        rcp += length;                 \
        ck = new_token(kind, NULL, 0); \
        cp = rcp;                      \
        goto next;

        struct Token head = {};
        struct Token *cur = &head;
        char *rcp = cp;
        for (;;) {
                while (map[*rcp] & BLANK) rcp++;
                switch (*rcp++) {
                        case '/':
                                if (*rcp == '*') {
                                        rcp++;
                                        while (rcp < limit) {
                                                if (*rcp == '*' &&
                                                    *(rcp + 1) == '/') {
                                                        rcp += 2;
                                                        break;
                                                }
                                                rcp++;
                                        }
                                        if (rcp >= limit) {
                                                error("Unterminated comment\n");
                                        }
                                        cp = rcp;
                                        continue;
                                }
                                if (*rcp == '\\') {
                                        rcp += 2;
                                }
                                if (*rcp == '/') {
                                        rcp++;
                                        while (*rcp != '\n') {
                                                if (rcp == limit) {
                                                        error("Unterminated "
                                                              "comment: %s\n",
                                                              rcp - 1);
                                                }
                                                if (*rcp == '\\')
                                                        rcp += 2;
                                                else
                                                        rcp++;
                                        }
                                        rcp++;
                                        cp = rcp;
                                        continue;
                                }
                                CHECK_PUNCTUATION('=', DIVASSIGN, 1)
                                HANDLE_TOKEN(DIV, 0);
                        case '<':
                                if (rcp[0] == '<' && rcp[1] == '=') {
                                        rcp += 2;
                                        ck = new_token(LSHIFTASSIGN, NULL, 0);
                                        goto next;
                                }
                                CHECK_PUNCTUATION('=', LEQ, 1)
                                CHECK_PUNCTUATION('<', LSHIFT, 1)
                                CHECK_PUNCTUATION(':', OBR, 1)
                                CHECK_PUNCTUATION('%', OPAR, 1)
                                ck = new_token(LT, NULL, 0);
                                goto next;
                        case '>':
                                if (rcp[0] == '>' && rcp[1] == '=') {
                                        rcp += 2;
                                        ck = new_token(RSHIFTASSIGN, NULL, 0);
                                        goto next;
                                }
                                CHECK_PUNCTUATION('=', GEQ, 1)
                                CHECK_PUNCTUATION('>', RSHIFT, 1)
                                ck = new_token(GT, NULL, 0);
                                goto next;
                        case '-':
                                CHECK_PUNCTUATION('>', DEREF, 1)
                                CHECK_PUNCTUATION('-', DECR, 1)
                                CHECK_PUNCTUATION('=', SUBASSIGN, 1)
                                ck = new_token(SUB, NULL, 0);
                                goto next;
                        case '=':
                                CHECK_PUNCTUATION('=', EQ, 1)
                                ck = new_token(ASSIGN, NULL, 0);
                                goto next;
                        case '!':
                                CHECK_PUNCTUATION('=', NEQ, 1)
                                ck = new_token(NOT, NULL, 0);
                                goto next;
                        case '|':
                                CHECK_PUNCTUATION('|', OROR, 1)
                                CHECK_PUNCTUATION('=', ORASSIGN, 1)
                                ck = new_token(OR, NULL, 0);
                                goto next;
                        case '&':
                                CHECK_PUNCTUATION('&', ANDAND, 1)
                                CHECK_PUNCTUATION('=', ANDASSIGN, 1)
                                ck = new_token(AND, NULL, 0);
                                goto next;
                        case '+':
                                CHECK_PUNCTUATION('+', INCR, 1)
                                CHECK_PUNCTUATION('=', ADDASSIGN, 1)
                                ck = new_token(ADD, NULL, 0);
                                goto next;
                        case ';': ck = new_token(SEMIC, NULL, 0); goto next;
                        case ',': ck = new_token(COMMA, NULL, 0); goto next;
                        case ':':
                                CHECK_PUNCTUATION('>', CBR, 1)
                                ck = new_token(COLON, NULL, 0);
                                goto next;
                        case '*':
                                CHECK_PUNCTUATION('=', MULASSIGN, 1)
                                ck = new_token(MUL, NULL, 0);
                                goto next;
                        case '~': ck = new_token(TILDA, NULL, 0); goto next;
                        case '%':
                                CHECK_PUNCTUATION('>', CPAR, 1)
                                CHECK_PUNCTUATION('=', MODASSIGN, 1)
                                CHECK_PUNCTUATION(':', STRGIZE, 1)
                                ck = new_token(MOD, NULL, 0);
                                goto next;
                        case '^':
                                CHECK_PUNCTUATION('=', XORASSIGN, 1)
                                ck = new_token(XOR, NULL, 0);
                                goto next;
                        case '?': ck = new_token(QMARK, NULL, 0); goto next;
                        case '[': ck = new_token(OBR, NULL, 0); goto next;
                        case ']': ck = new_token(CBR, NULL, 0); goto next;
                        case '{': ck = new_token(OCBR, NULL, 0); goto next;
                        case '}': ck = new_token(CCBR, NULL, 0); goto next;
                        case '(': ck = new_token(OPAR, NULL, 0); goto next;
                        case ')': ck = new_token(CPAR, NULL, 0); goto next;
                        case '\n':
                        case '\v':
                        case '\r':
                        case '\f': continue;
                        case '\0':
                                ck = new_token(EOI, NULL, 0);
                                cur = cur->next = ck;
                                goto exit_loop;
                        case 'i':
                                if (rcp[0] == 'f' &&
                                    !(map[rcp[1]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(IF, 1);
                                }
                                if (rcp[0] == 'n' && rcp[1] == 't' &&
                                    !(map[rcp[2]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(INT, 2);
                                }
                                goto id;
                        case 'h':
                        case 'j':
                        case 'k':
                        case 'm':
                        case 'n':
                        case 'o':
                        case 'p':
                        case 'q':
                        case 'x':
                        case 'y':
                        case 'z':
                        case 'A':
                        case 'B':
                        case 'C':
                        case 'D':
                        case 'E':
                        case 'F':
                        case 'G':
                        case 'H':
                        case 'I':
                        case 'J':
                        case 'K':
                        case 'M':
                        case 'N':
                        case 'O':
                        case 'P':
                        case 'Q':
                        case 'R':
                        case 'S':
                        case 'T':
                        case 'U':
                        case 'V':
                        case 'W':
                        case 'X':
                        case 'Y':
                        case 'Z':
                        id : {
                                char *start = rcp - 1;
                                while (map[*rcp] & (DIGIT | LETTER)) rcp++;
                                ck = new_token(IDENT, start, rcp - start);
                                cp = rcp;
                                goto next;
                        }
                        next : {
                                cur = cur->next = ck;
                                continue;
                        }
                        case '0':
                        case '1':
                        case '2':
                        case '3':
                        case '4':
                        case '5':
                        case '6':
                        case '7':
                        case '8':
                        case '9': {
                                unsigned long n = 0;
                                char *start = rcp - 1;
                                if (*start == '0' &&
                                    (*rcp == 'x' || *rcp == 'X')) {
                                        // hex
                                        // HANDLEME: overflow
                                        int d;
                                        while (*++rcp) {
                                                if (map[*rcp] & DIGIT)
                                                        d = *rcp - '0';
                                                else if (*rcp >= 'a' &&
                                                         *rcp <= 'f')
                                                        d = *rcp - 'a' + 10;
                                                else if (*rcp >= 'A' &&
                                                         *rcp <= 'F')
                                                        d = *rcp - 'A' + 10;
                                                else
                                                        break;
                                                n = (n << 4) | d;
                                        }
                                        if (*rcp == 'l' || *rcp == 'L') rcp++;
                                        ck = new_token(
                                                INTCONST, start, rcp - start);
                                        ck->value = n;
                                        cp = rcp;
                                        goto next;
                                } else if (*start == '0') {
                                        int err = 0;
                                        // octal
                                        // HANDLEME: overflow
                                        // HANDLEME: floating point
                                        for (; map[*rcp] & DIGIT; rcp++) {
                                                if (*rcp == '8' || *rcp == '9')
                                                        err = 1;
                                                n = (n << 3) + (*rcp - '0');
                                        }
                                        if (err)
                                                error("Invalid octal "
                                                      "constant: %.*s\n",
                                                      rcp - start,
                                                      start);

                                        if ((*rcp == 'u' || *rcp == 'U') &&
                                                    (rcp[1] == 'l' ||
                                                     rcp[1] == 'L') ||
                                            (*rcp == 'l' || *rcp == 'L') &&
                                                    (rcp[1] == 'u' ||
                                                     rcp[1] == 'U')) {
                                                rcp += 2;
                                        }
                                        if (*rcp == 'u' || *rcp == 'U') {
                                                rcp++;
                                        }
                                        if (*rcp == 'l' || *rcp == 'L') rcp++;
                                        ck = new_token(
                                                INTCONST, start, rcp - start);
                                        ck->value = n;
                                        cp = rcp;
                                        goto next;
                                } else {
                                        // decimal
                                        for (n = *start - '0';
                                             map[*rcp] & DIGIT;) {
                                                int d = *rcp++ - '0';
                                                n = n * 10 + d;
                                        }
                                        if (*rcp == '.' || *rcp == 'e' ||
                                            *rcp == 'E') {
                                                floatconst(&rcp);
                                                ck = new_token(FLOATCONST,
                                                               start,
                                                               rcp - start);
                                                ck->value = n;
                                                cp = rcp;
                                                goto next;
                                        }
                                        if ((*rcp == 'u' || *rcp == 'U') &&
                                                    (rcp[1] == 'l' ||
                                                     rcp[1] == 'L') ||
                                            (*rcp == 'l' || *rcp == 'L') &&
                                                    (rcp[1] == 'u' ||
                                                     rcp[1] == 'U')) {
                                                rcp += 2;
                                        }
                                        if (*rcp == 'u' || *rcp == 'U') {
                                                rcp++;
                                        }
                                        if (*rcp == 'l' || *rcp == 'L') {
                                                rcp++;
                                        }
                                        ck = new_token(
                                                INTCONST, start, rcp - start);
                                        ck->value = n;
                                        cp = rcp;
                                        goto next;
                                }
                        }
                        case '.':
                                if (rcp[0] == '.' && rcp[1] == '.') {
                                        ck = new_token(ELLIPSIS, NULL, 0);
                                        rcp += 2;
                                        cp = rcp;
                                        goto next;
                                }
                                // HANDLEME: floating point
                                if ((map[*rcp] & DIGIT)) {
                                        char *start = --rcp;
                                        floatconst(&rcp);
                                        ck = new_token(
                                                FLOATCONST, start, rcp - start);
                                        cp = rcp;
                                        goto next;
                                }
                                HANDLE_TOKEN(DOT, 0);
                        case 'L':
                                // HANDLEME: wide char int const
                                // HANDLEME: wide char string const
                                goto id;
                        case '\'': {
                                // HANDLEME: wide char int const
                                char *start = rcp - 1;
                                while (*rcp != '\'') {
                                        if (*rcp == '\\') {
                                                rcp++;
                                        }
                                        if (rcp == limit) {
                                                error("Unterminated char "
                                                      "constant: %s\n",
                                                      start);
                                        }
                                        rcp++;
                                }
                                rcp++;
                                ck = new_token(CHARCONST, start, rcp - start);
                                cp = rcp;
                                goto next;
                        }
                        case '"': {
                                char *start = rcp - 1;
                                while (*rcp != '"') {
                                        if (*rcp == '\\') {
                                                rcp++;
                                        }
                                        if (rcp == limit) {
                                                error("Unterminated string "
                                                      "constant: %s\n",
                                                      start);
                                        }
                                        rcp++;
                                }
                                rcp++;
                                ck = new_token(STRCONST, start, rcp - start);
                                cp = rcp;
                                goto next;
                        }
                        case 'a':
                                if (rcp[0] == 'u' && rcp[1] == 't' &&
                                    rcp[2] == 'o' &&
                                    !(map[rcp[3]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(AUTO, 3);
                                }
                                goto id;
                        case 'b':
                                if (rcp[0] == 'r' && rcp[1] == 'e' &&
                                    rcp[2] == 'a' && rcp[3] == 'k' &&
                                    !(map[rcp[4]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(BREAK, 4);
                                }
                                goto id;
                        case 'c':
                                if (rcp[0] == 'a' && rcp[1] == 's' &&
                                    rcp[2] == 'e' &&
                                    !(map[rcp[3]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(CASE, 3);
                                }
                                if (rcp[0] == 'h' && rcp[1] == 'a' &&
                                    rcp[2] == 'r' &&
                                    !(map[rcp[3]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(CHAR, 3);
                                }
                                if (rcp[0] == 'o' && rcp[1] == 'n' &&
                                    rcp[2] == 's' && rcp[3] == 't' &&
                                    !(map[rcp[4]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(CONST, 4);
                                }
                                if (rcp[0] == 'o' && rcp[1] == 'n' &&
                                    rcp[2] == 't' && rcp[3] == 'i' &&
                                    rcp[4] == 'n' && rcp[5] == 'u' &&
                                    rcp[6] == 'e' &&
                                    !(map[rcp[7]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(CONTINUE, 7);
                                }
                                goto id;
                        case 'd':
                                if (rcp[0] == 'e' && rcp[1] == 'f' &&
                                    rcp[2] == 'a' && rcp[3] == 'u' &&
                                    rcp[4] == 'l' && rcp[5] == 't' &&
                                    !(map[rcp[6]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(DEFAULT, 6);
                                }
                                if (rcp[0] == 'o' && rcp[1] == 'u' &&
                                    rcp[2] == 'b' && rcp[3] == 'l' &&
                                    rcp[4] == 'e' &&
                                    !(map[rcp[5]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(DOUBLE, 5);
                                }
                                if (rcp[0] == 'o' &&
                                    !(map[rcp[1]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(DO, 1);
                                }
                                goto id;
                        case 'e':
                                if (rcp[0] == 'l' && rcp[1] == 's' &&
                                    rcp[2] == 'e' &&
                                    !(map[rcp[3]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(ELSE, 3);
                                }
                                if (rcp[0] == 'n' && rcp[1] == 'u' &&
                                    rcp[2] == 'm' &&
                                    !(map[rcp[3]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(ENUM, 3);
                                }
                                if (rcp[0] == 'x' && rcp[1] == 't' &&
                                    rcp[2] == 'e' && rcp[3] == 'r' &&
                                    rcp[4] == 'n' &&
                                    !(map[rcp[5]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(EXTERN, 5);
                                }
                                goto id;
                        case 'f':
                                if (rcp[0] == 'l' && rcp[1] == 'o' &&
                                    rcp[2] == 'a' && rcp[3] == 't' &&
                                    !(map[rcp[4]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(FLOAT, 4);
                                }
                                if (rcp[0] == 'o' && rcp[1] == 'r' &&
                                    !(map[rcp[2]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(FOR, 2);
                                }
                                goto id;
                        case 'g':
                                if (rcp[0] == 'o' && rcp[1] == 't' &&
                                    rcp[2] == 'o' &&
                                    !(map[rcp[3]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(GOTO, 3);
                                }
                                goto id;
                        case 'l':
                                if (rcp[0] == 'o' && rcp[1] == 'n' &&
                                    rcp[2] == 'g' &&
                                    !(map[rcp[3]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(LONG, 3);
                                }
                                goto id;
                        case 'r':
                                if (rcp[0] == 'e' && rcp[1] == 'g' &&
                                    rcp[2] == 'i' && rcp[3] == 's' &&
                                    rcp[4] == 't' && rcp[5] == 'e' &&
                                    rcp[6] == 'r' &&
                                    !(map[rcp[7]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(REGISTER, 7);
                                }
                                if (rcp[0] == 'e' && rcp[1] == 't' &&
                                    rcp[2] == 'u' && rcp[3] == 'r' &&
                                    rcp[4] == 'n' &&
                                    !(map[rcp[5]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(RETURN, 5);
                                }
                                goto id;
                        case 's':
                                if (rcp[0] == 'h' && rcp[1] == 'o' &&
                                    rcp[2] == 'r' && rcp[3] == 't' &&
                                    !(map[rcp[4]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(SHORT, 4);
                                }
                                if (rcp[0] == 'i' && rcp[1] == 'g' &&
                                    rcp[2] == 'n' && rcp[3] == 'e' &&
                                    rcp[4] == 'd' &&
                                    !(map[rcp[5]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(SIGNED, 5);
                                }
                                if (rcp[0] == 'i' && rcp[1] == 'z' &&
                                    rcp[2] == 'e' && rcp[3] == 'o' &&
                                    rcp[4] == 'f' &&
                                    !(map[rcp[5]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(SIZEOF, 5);
                                }
                                if (rcp[0] == 't' && rcp[1] == 'a' &&
                                    rcp[2] == 't' && rcp[3] == 'i' &&
                                    rcp[4] == 'c' &&
                                    !(map[rcp[5]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(STATIC, 5);
                                }
                                if (rcp[0] == 't' && rcp[1] == 'r' &&
                                    rcp[2] == 'u' && rcp[3] == 'c' &&
                                    rcp[4] == 't' &&
                                    !(map[rcp[5]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(STRUCT, 5);
                                }
                                if (rcp[0] == 'w' && rcp[1] == 'i' &&
                                    rcp[2] == 't' && rcp[3] == 'c' &&
                                    rcp[4] == 'h' &&
                                    !(map[rcp[5]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(SWITCH, 5);
                                }
                                goto id;
                        case 't':
                                if (rcp[0] == 'y' && rcp[1] == 'p' &&
                                    rcp[2] == 'e' && rcp[3] == 'd' &&
                                    rcp[4] == 'e' && rcp[5] == 'f' &&
                                    !(map[rcp[6]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(TYPEDEF, 6);
                                }
                                goto id;
                        case 'u':
                                if (rcp[0] == 'n' && rcp[1] == 'i' &&
                                    rcp[2] == 'o' && rcp[3] == 'n' &&
                                    !(map[rcp[4]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(UNION, 4);
                                }
                                if (rcp[0] == 'n' && rcp[1] == 's' &&
                                    rcp[2] == 'i' && rcp[3] == 'g' &&
                                    rcp[4] == 'n' && rcp[5] == 'e' &&
                                    rcp[6] == 'd' &&
                                    !(map[rcp[7]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(UNSIGNED, 7);
                                }
                                goto id;
                        case 'v':
                                if (rcp[0] == 'o' && rcp[1] == 'i' &&
                                    rcp[2] == 'd' &&
                                    !(map[rcp[3]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(VOID, 3);
                                }
                                if (rcp[0] == 'o' && rcp[1] == 'l' &&
                                    rcp[2] == 'a' && rcp[3] == 't' &&
                                    rcp[4] == 'i' && rcp[5] == 'l' &&
                                    rcp[6] == 'e' &&
                                    !(map[rcp[7]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(VOLATILE, 7);
                                }
                                goto id;
                        case 'w':
                                if (rcp[0] == 'h' && rcp[1] == 'i' &&
                                    rcp[2] == 'l' && rcp[3] == 'e' &&
                                    !(map[rcp[4]] & (DIGIT | LETTER))) {
                                        HANDLE_TOKEN(WHILE, 4);
                                }
                                goto id;
                        case '_': goto id;
                        case '#':
                                if (rcp[0] == 'i' && rcp[1] == 'n' &&
                                    rcp[2] == 'c' && rcp[3] == 'l' &&
                                    rcp[4] == 'u' && rcp[5] == 'd' &&
                                    rcp[6] == 'e' && rcp[7] == ' ') {
                                        HANDLE_TOKEN(INCLUDE, 7);
                                }
                                if (rcp[0] == 'd' && rcp[1] == 'e' &&
                                    rcp[2] == 'f' && rcp[3] == 'i' &&
                                    rcp[4] == 'n' && rcp[5] == 'e' &&
                                    rcp[6] == ' ') {
                                        HANDLE_TOKEN(DEFINE, 6);
                                }
                                CHECK_PUNCTUATION('#', TKPASTE, 1)
                                error("Invalid preprocessor directive: %s\n",
                                      rcp - 1);
                        default: error("Unhandled character: %c\n", *(rcp - 1));
                }
        }
exit_loop:
        return head.next;
}

void floatconst(char **start) {
        if (**start == '.') {
                do (*start)++;
                while (map[**start] & DIGIT);
        }
        if (**start == 'e' || **start == 'E') {
                if (*(++(*start)) == '-' || **start == '+') (*start)++;
                if (map[**start] & DIGIT) {
                        do {
                                (*start)++;
                        } while (map[**start] & DIGIT);
                } else {
                        error("Invalid floating point constant: %s\n", *start);
                }
        }
        if (**start == 'f' || **start == 'F') {
                ++(*start);
        } else if (**start == 'l' || **start == 'L') {
                // long double
                (*start)++;
        } else {
                // double type
        }
};

void error(char *fmt, ...) {
        va_list args;
        va_start(args, fmt);
        vprintf(fmt, args);
        va_end(args);
        exit(1);
}

void printTokens(struct Token *head, FILE *outfile) {
        struct Token *current = head;
        while (current != NULL) {
                printTokenKind(current->kind, outfile);
                fprintf(outfile, "\n");
                // fprintf(outfile, ": %.*s\n", current->len, current->start);
                current = current->next;
        }
}

void printTokenKind(enum TokenKind kind, FILE *output) {
        const char *tokenStrs[] = {
                "LT",        "GT",           "LEQ",          "GEQ",
                "LSHIFT",    "RSHIFT",       "DEREF",        "DECR",
                "EQ",        "NEQ",          "ADD",          "SUB",
                "MUL",       "DIV",          "MOD",          "ADDASSIGN",
                "SUBASSIGN", "MULASSIGN",    "DIVASSIGN",    "MODASSIGN",
                "OROR",      "ANDAND",       "INCR",         "EOI",
                "IF",        "INT",          "OBR",          "CBR",
                "OCBR",      "CCBR",         "OPAR",         "CPAR",
                "SEMIC",     "COMMA",        "TILDA",        "AND",
                "OR",        "XOR",          "NOT",          "ANDASSIGN",
                "ORASSIGN",  "XORASSIGN",    "NOTASSIGN",    "STRGIZE",
                "TKPASTE",   "ASSIGN",       "QMARK",        "IDENT",
                "INTCONST",  "FLOATCONST",   "STRCONST",     "CHARCONST",
                "ELLIPSIS",  "AUTO",         "CASE",         "CHAR",
                "CONST",     "CONTINUE",     "DEFAULT",      "DO",
                "DOUBLE",    "ELSE",         "ENUM",         "EXTERN",
                "FLOAT",     "FOR",          "GOTO",         "LONG",
                "REGISTER",  "RETURN",       "SHORT",        "SIGNED",
                "SIZEOF",    "STATIC",       "STRUCT",       "SWITCH",
                "TYPEDEF",   "UNION",        "UNSIGNED",     "VOID",
                "VOLATILE",  "WHILE",        "DOT",          "BREAK",
                "COLON",     "RSHIFTASSIGN", "LSHIFTASSIGN", "INCLUDE",
                "DEFINE"};
        size_t numTokenStrs = sizeof(tokenStrs) / sizeof(tokenStrs[0]);

        if (kind >= 0 && kind < numTokenStrs) {
                fprintf(output, "%s", tokenStrs[kind]);
        } else {
                fprintf(output, "Unknown token");
        }
}
