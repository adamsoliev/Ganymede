#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

enum TokenKind {
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
};

struct Token {
        enum TokenKind kind;
        char *start;  // for IDENT
        int len;      // for IDENT
        struct Token *next;
        int value;  // for INTCONST
};

static struct Token *new_token(enum TokenKind kind, char *start, int len) {
        struct Token *token = calloc(1, sizeof(struct Token));
        token->kind = kind;
        token->start = start;
        token->len = len;
        return token;
}

static void error(char *fmt, ...) {
        va_list args;
        va_start(args, fmt);
        vprintf(fmt, args);
        va_end(args);
        exit(1);
}

char *limit;
static struct Token *ck;

void printTokens(struct Token *head);
void printTokenKind(enum TokenKind kind, FILE *output);

struct Token *scan(char *cp) {
#define CHECK_PUNCTUATION(op, token, incr)      \
        if (*rcp == op) {                       \
                cp += incr;                     \
                ck = new_token(token, NULL, 0); \
                goto next;                      \
        }

        struct Token head = {};
        struct Token *cur = &head;
        char *rcp = cp;
        for (;;) {
                while (map[*rcp] & BLANK) rcp++;
                switch (*rcp++) {
                        case '/':
                                if (*rcp++ == '*') {
                                        while (rcp < limit) {
                                                if (*rcp == '*' &&
                                                    *(rcp + 1) == '/') {
                                                        rcp += 2;
                                                        break;
                                                }
                                                *rcp++;
                                        }
                                        if (rcp >= limit) {
                                                error("Unterminated comment\n");
                                        }
                                        cp = rcp;
                                        continue;
                                }
                                error("Unknown character following '/'\n");
                        case '<':
                                CHECK_PUNCTUATION('=', LEQ, 1)
                                CHECK_PUNCTUATION('<', LSHIFT, 1)
                                CHECK_PUNCTUATION(':', OBR, 1)
                                CHECK_PUNCTUATION('%', OPAR, 1)
                                // HANDLEME: <<=
                                ck = new_token(LT, NULL, 0);
                                goto next;
                        case '>':
                                CHECK_PUNCTUATION('=', GEQ, 1)
                                CHECK_PUNCTUATION('>', RSHIFT, 1)
                                // HANDLEME: >>=
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
                        case ';': ck = new_token(SEMIC, NULL, 0); continue;
                        case ',': ck = new_token(COMMA, NULL, 0); continue;
                        case ':': CHECK_PUNCTUATION('>', CBR, 1)
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
                        case '\0':
                        case '\f':
                                if (rcp == limit) {
                                        ck = new_token(EOI, NULL, 0);
                                        cur = cur->next = ck;
                                        goto exit_loop;
                                }
                        case 'i':
                                if (rcp[0] == 'f' &&
                                    !(map[rcp[1]] & (DIGIT | LETTER))) {
                                        rcp += 1;
                                        ck = new_token(IF, NULL, 0);
                                        cp = rcp;
                                        goto next;
                                }
                                if (rcp[0] == 'n' && rcp[1] == 't' &&
                                    !(map[rcp[2]] & (DIGIT | LETTER))) {
                                        rcp += 2;
                                        ck = new_token(INT, NULL, 0);
                                        cp = rcp;
                                        goto next;
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
                                        if (map[*rcp] & LETTER)
                                                error("Invalid hex constant: "
                                                      "%.*s\n",
                                                      rcp - start + 1,
                                                      start);
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
                                        ck = new_token(
                                                INTCONST, start, rcp - start);
                                        ck->value = n;
                                        cp = rcp;
                                        goto next;
                                }
                        }
                        
                        next : {
                                cur = cur->next = ck;
                                continue;
                        }
                        default: error("Unhandled character: %c\n", *(rcp - 1));
                }
        }
exit_loop:
        return head.next;
}

int main() {
        char cp[] = "0x1234 29 198734 05125 0x231465 int main() 0755";
        limit = cp + sizeof(cp);
        struct Token *tokens = scan(cp);
        printTokens(tokens);
        return 0;
}

void printTokens(struct Token *head) {
        FILE *output = stdout;
        struct Token *current = head;
        while (current != NULL) {
                printTokenKind(current->kind, output);
                fprintf(output, "\t: %.*s\n", current->len, current->start);
                current = current->next;
        }
}

void printTokenKind(enum TokenKind kind, FILE *output) {
        switch (kind) {
                case LEQ: fprintf(output, "LEQ"); break;
                case GEQ: fprintf(output, "GEQ"); break;
                case LSHIFT: fprintf(output, "LSHIFT"); break;
                case RSHIFT: fprintf(output, "RSHIFT"); break;
                case DEREF: fprintf(output, "DEREF"); break;
                case DECR: fprintf(output, "DECR"); break;
                case EQ: fprintf(output, "EQ"); break;
                case NEQ: fprintf(output, "NEQ"); break;
                case ADD: fprintf(output, "ADD"); break;
                case SUB: fprintf(output, "SUB"); break;
                case MUL: fprintf(output, "MUL"); break;
                case DIV: fprintf(output, "DIV"); break;
                case ADDASSIGN: fprintf(output, "ADDASSIGN"); break;
                case SUBASSIGN: fprintf(output, "SUBASSIGN"); break;
                case MULASSIGN: fprintf(output, "MULASSIGN"); break;
                case DIVASSIGN: fprintf(output, "DIVASSIGN"); break;
                case OROR: fprintf(output, "OROR"); break;
                case ANDAND: fprintf(output, "ANDAND"); break;
                case INCR: fprintf(output, "INCR"); break;
                case EOI: fprintf(output, "EOI"); break;
                case IF: fprintf(output, "IF"); break;
                case INT: fprintf(output, "INT"); break;
                case OBR: fprintf(output, "OBR"); break;
                case CBR: fprintf(output, "CBR"); break;
                case OCBR: fprintf(output, "OCBR"); break;
                case CCBR: fprintf(output, "CCBR"); break;
                case OPAR: fprintf(output, "OPAR"); break;
                case CPAR: fprintf(output, "CPAR"); break;
                case SEMIC: fprintf(output, "SEMIC"); break;
                case COMMA: fprintf(output, "COMMA"); break;
                case LT: fprintf(output, "LT"); break;
                case GT: fprintf(output, "GT"); break;
                case MOD: fprintf(output, "MOD"); break;
                case MODASSIGN: fprintf(output, "MODASSIGN"); break;
                case TILDA: fprintf(output, "TILDA"); break;
                case AND: fprintf(output, "AND"); break;
                case OR: fprintf(output, "OR"); break;
                case XOR: fprintf(output, "XOR"); break;
                case NOT: fprintf(output, "NOT"); break;
                case ANDASSIGN: fprintf(output, "ANDASSIGN"); break;
                case ORASSIGN: fprintf(output, "ORASSIGN "); break;
                case XORASSIGN: fprintf(output, "XORASSIGN"); break;
                case NOTASSIGN: fprintf(output, "NOTASSIGN"); break;
                case STRGIZE: fprintf(output, "STRGIZE"); break;
                case TKPASTE: fprintf(output, "TKPASTE"); break;
                case ASSIGN: fprintf(output, "ASSIGN"); break;
                case QMARK: fprintf(output, "QMARK"); break;
                case IDENT: fprintf(output, "IDENT"); break;
                case INTCONST: fprintf(output, "INTCONST"); break;
                case FLOATCONST: fprintf(output, "FLOATCONS"); break;
                case STRCONST: fprintf(output, "STRCONST"); break;
                case CHARCONST: fprintf(output, "CHARCONST"); break;
                default: fprintf(output, "Unknown Token"); break;
        }
}