xx(NONE, 0, "NONE")
        /* storage-class specifiers */
        xx(TYPEDEF, 1, "TYPEDEF") xx(EXTERN, 2, "EXTERN") xx(STATIC, 3, "STATIC")
                xx(AUTO, 4, "AUTO") xx(REGISTER, 5, "REGISTER")
        /* type-qualifier */
        xx(CONST, 6, "CONST") xx(RESTRICT, 7, "RESTRICT") xx(VOLATILE, 8, "VOLATILE")
        /* func-specifier */
        xx(INLINE, 9, "INLINE")
        /* type-specifier */
        xx(VOID, 10, "VOID") xx(CHAR, 11, "CHAR") xx(SHORT, 12, "SHORT") xx(INT, 13, "INT")
                xx(LONG, 14, "LONG") xx(FLOAT, 15, "FLOAT") xx(DOUBLE, 16, "DOUBLE")
                        xx(SIGNED, 17, "SIGNED") xx(UNSIGNED, 18, "UNSIGNED")
                                xx(STRUCT, 19, "STRUCT") xx(UNION, 20, "UNION") xx(ENUM, 21, "ENUM")
        /* binary ops - increasing grouped precedence */
        xx(OROR, 22, "OROR")      // ||
        xx(ANDAND, 23, "ANDAND")  // &&
        xx(OR, 24, "OR")          // |
        xx(XOR, 25, "XOR")        // ^
        xx(AND, 26, "AND")        // &
        xx(EQ, 27, "EQ")          // ==
        xx(NEQ, 28, "NEQ")        // !=
        xx(LT, 29, "LT")          // <
        xx(GT, 30, "GT")          // >
        xx(LEQ, 31, "LEQ")        // <=
        xx(GEQ, 32, "GEQ")        // >=
        xx(LSHIFT, 33, "LSHIFT")  // <<
        xx(RSHIFT, 34, "RSHIFT")  // >>
        xx(ADD, 35, "ADD")        // +
        xx(SUB, 36, "SUB")        // -
        xx(MUL, 37, "MUL")        // *
        xx(DIV, 38, "DIV")        // /
        xx(MOD, 39, "MOD")        // %
        /* unary ops */
        xx(DECR, 40, "DECR") xx(INCR, 41, "INCR")  // ++
        /* postfix ops */
        xx(DEREF, 42, "DEREF") xx(DOT, 43, "DOT")
        /* assigns */
        xx(ASSIGN, 44, "ASSIGN") xx(ADDASSIGN, 45, "ADDASSIGN") xx(SUBASSIGN, 46, "SUBASSIGN")
                xx(MULASSIGN, 47, "MULASSIGN") xx(DIVASSIGN, 48, "DIVASSIGN")
                        xx(MODASSIGN, 49, "MODASSIGN") xx(ANDASSIGN, 50, "ANDASSIGN")
                                xx(ORASSIGN, 51, "ORASSIGN") xx(XORASSIGN, 52, "XORASSIGN")
                                        xx(NOTASSIGN, 53, "NOTASSIGN")
                                                xx(LSHASSIGN, 54, "LSHASSIGN")
                                                        xx(RSHASSIGN, 55, "RSHASSIGN")
        /* iter stmt */
        xx(FOR, 56, "FOR") xx(WHILE, 57, "WHILE") xx(DO, 58, "DO")
        /* select stmt */
        xx(IF, 59, "IF") xx(SWITCH, 60, "SWITCH") xx(ELSE, 61, "ELSE")
        /* jump stmt */
        xx(GOTO, 62, "GOTO") xx(CONTINUE, 63, "CONTINUE") xx(BREAK, 64, "BREAK")
                xx(RETURN, 65, "RETURN")
        /* label stmt */
        xx(CASE, 66, "CASE") xx(DEFAULT, 67, "DEFAULT")
        /* consts */
        xx(IDENT, 68, "IDENT") xx(ICON, 69, "ICON") xx(FCON, 70, "FCON")
                xx(DCON, 71, "DCON") xx(LDCON, 72, "LDCON")
                        xx(SCON, 73, "SCON") xx(CCON, 74, "CCON")
        /* preprocess */
        xx(INCLUDE, 75, "INCLUDE") xx(STRGIZE, 76, "STRGIZE")  // #
        xx(TKPASTE, 77, "TKPASTE")                             // ##
        xx(DEFINE, 78, "DEFINE")
        /* punct */
        xx(OBR, 79, "OBR")      // [
        xx(CBR, 80, "CBR")      // ]
        xx(OCBR, 81, "OCBR")    // {
        xx(CCBR, 82, "CCBR")    // }
        xx(OPAR, 83, "OPAR")    // (
        xx(CPAR, 84, "CPAR")    // )
        xx(SEMIC, 85, "SEMIC")  // ;
        xx(COMMA, 86, "COMMA")  // ,
        xx(TILDA, 87, "TILDA")  // ~
        xx(NOT, 88, "NOT")      // !
        xx(QMARK, 89, "QMARK")  // ?
        xx(COLON, 90, "COLON") xx(ELLIPSIS, 91, "ELLIPSIS") xx(BACKSLASH, 92, "BACKSLASH")

                xx(SIZEOF, 93, "SIZEOF") xx(EOI, 94, "EOI")
#undef xx
