#include <stdio.h>
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
        ADDASSIGN,  // +=
        SUBASSIGN,  // -=
        MULASSIGN,  // *=
        DIVASSIGN,  // /=
        OROR,       // ||
        ANDAND,     // &&
        INCR,       // ++
        EOI,        // end of input
        IF,
        INT,
        OBR,    // [
        CBR,    // ]
        OCBR,   // {
        CCBR,   // }
        OPAR,   // (
        CPAR,   // )
        SEMIC,  // ;
        COMMA,  // ,
};

struct Token {
        enum TokenKind kind;
        char *start;
        int len;
        struct Token *next;
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
                        case '>':
                                CHECK_PUNCTUATION('=', GEQ, 1)
                                CHECK_PUNCTUATION('>', RSHIFT, 1)
                                // HANDLEME: >>=
                        case '-':
                                CHECK_PUNCTUATION('>', DEREF, 1)
                                CHECK_PUNCTUATION('-', DECR, 1)
                                CHECK_PUNCTUATION('=', SUBASSIGN, 1)
                                ck = new_token(SUB, NULL, 0);
                                goto next;
                        case '=': CHECK_PUNCTUATION('=', EQ, 1)
                        case '!': CHECK_PUNCTUATION('=', NEQ, 1)
                        case '|': CHECK_PUNCTUATION('|', OROR, 1)
                        case '&': CHECK_PUNCTUATION('&', ANDAND, 1)
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
                        case '~':
                        case '%': CHECK_PUNCTUATION('>', CPAR, 1)
                        case '^':
                        case '?':
                        case '[':
                        case ']':
                        case '{':
                        case '}':
                        case '(':
                        case ')':
                        case '\n':
                        case '\v':
                        case '\r':
                        case '\f':
                        case 'i':
                                if (rcp[0] == 'f' &&
                                    !(map[rcp[1]] & (DIGIT | LETTER))) {
                                        cp = rcp + 1;
                                        ck = new_token(IF, NULL, 0);
                                        goto next;
                                }
                                if (rcp[0] == 'n' && rcp[1] == 't' &&
                                    !(map[rcp[2]] & (DIGIT | LETTER))) {
                                        cp = rcp + 2;
                                        ck = new_token(INT, NULL, 0);
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
                        id:
                                continue;
                        next:
                                cur = cur->next = ck;
                                continue;
                        default: error("Unhandled character: %c\n", *(rcp - 1));
                }
        }
        return head.next;
}

int main() {
        char *cp = "int main() { return 0; }";
        struct Token *tokens = scan(cp);
        return 0;
}

void printTokens(struct Token *head) {
        FILE *output = stdout;
        struct Token *current = head;
        while (current != NULL) {
                fprintf(output, "Token: ");
                printTokenKind(current->kind, output);
                fprintf(output,
                        "\tStart: %.*s\n",
                        current->len,
                        current->start);
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
                default: fprintf(output, "Unknown Token"); break;
        }
}
