#pragma clang diagnostic ignored "-Wgnu-empty-initializer"
#pragma clang diagnostic ignored "-Wstrict-prototypes"

#include "ganymede.h"

/* global variables */
uint64_t _CTK;
uint64_t _INDEX = 0;
int _COMPTYPELEVEL = 1;
/* if GLOBAL, first declarator is parsed upfront to diff funcdef */
enum { GLOBAL = 1, LOCAL, PARAM } _cdecllevel;

/* utility functions */
static inline void consume(const char *msg, enum Kind kind) {
        if (TGETKIND(_CTK) != kind) {
                uint64_t line = TGETROW(_CTK);
                error("%s: expected '%s', but got '%s' in line %d\n ",
                      msg,
                      token_names[kind],
                      token_names[TGETKIND(_CTK)],
                      line);
        }
        _CTK = tokens[++_INDEX];
}

/* ---------- RECURSIVE DESCENT PARSER ---------- */

// https://github.com/katef/kgt/blob/main/examples/c99-grammar.iso-ebnf

void jumpstmt();
void stmt();
void directdeclarator();
void pointer();
void funcspec();
void typequal();
void typespec(uint64_t *);
void sclass(uint64_t *);
void declorstmt();
void compstmt();
void declarator();
uint64_t declspec();
void declaration();
void funcdef();
void extdecl();
void initdeclaratorlist();
void selectstmt();
void iterstmt();
void labelstmt();
void initializer();
void structdeclarationlist();
void structorunionspec();
void specqual();
void specquallist();
void structdeclaration();
void structdeclarator();
void structdeclaratorlist();
void enumtor();
void enumtorlist();
void enumspec();
void designator();
void designtorlist();
void designation();
void designinitzer();
void paramdeclaration();
void paramlist();
void paramtypelist();

void constant();
void primaryexpr();
void postfixoperator();
void postfixexpr();
void unaryexpr();
void binaryexpr();
void condexpr();
void assignexpr();
void expr();
void exprstmt();
void constexpr();
void typequalifierlist();

// translation-unit = {external-declaration}
void parse(void) {
        _CTK = tokens[_INDEX];
        while (TGETKIND(_CTK) != EOI) {
                extdecl();
        }
}

// external-declaration = declaration
void extdecl() {
        _cdecllevel = GLOBAL;
        declaration();
}

// function-definition = compound-statement
void funcdef() { compstmt(); }

// declaration = declaration-specifiers declarator (function-definition | [init-declarator-list] ';')
//             | ';'
void declaration() {
        /* parse 1st declarator to diff function definition */
        uint64_t type = declspec();
        declarator(&type, 0);
        printf("Type: " BB_P64 "\n", BB64(type));
        /* TODO: resolve type info here since declarator collected it */
        if (_cdecllevel == GLOBAL && TGETKIND(_CTK) == OCBR) {
                /* TODO: funcdef if
                        0) level is global
                        1) 1st declarator specifies FUNCT
                        2) 1st declarator includes IDENT
                        3) next token is '{' (we don't support old stype funcdef)
                */
                funcdef();
                return;
        }
        initdeclaratorlist();
        consume("missing ';' of declaration", SEMIC);
}

// declaration-specifiers = declaration-specifier {declaration-specifier}
// declaration-specifier = storage-class-specifier
//                       | type-specifier
//                       | type-qualifier
//                       | function-specifier
uint64_t declspec() {
        uint64_t btype = 0;
        while (TGETKIND(_CTK) >= TYPEDEF && TGETKIND(_CTK) <= ENUM) {
                sclass(&btype);
                typespec(&btype);
                typequal(&btype);
                funcspec(&btype);
        }
        return btype;
}

// declarator = [pointer] direct-declarator {suffix-declarator}
void declarator(uint64_t *type, int level) {
        if (TGETKIND(_CTK) == MUL) pointer(type, &level);
        directdeclarator(type, level);
}

// declaration-list = declaration {declaration}

// compound-statement = '{' {declaration-or-statement} '}'
void compstmt() {
        consume("missing '{' of compound statement", OCBR);
        while (TGETKIND(_CTK) != CCBR && TGETKIND(_CTK) != EOI) {
                declorstmt();
        }
        consume("missing '}' of compound statement", CCBR);
}

// declaration-or-statement = declaration | statement
void declorstmt() {
        if (TGETKIND(_CTK) >= TYPEDEF && TGETKIND(_CTK) <= ENUM) {
                _cdecllevel = LOCAL;
                declaration();
        } else {
                stmt();
        }
}

// init-declarator-list = ['=' initializer] {',' declarator ['=' initializer]}
void initdeclaratorlist() {
        /* first declarator is already parsed. now see if it has initializer */
        if (TGETKIND(_CTK) == ASSIGN) {
                consume("", ASSIGN);
                initializer();
        }
        /* rest of the declarators */
        while (TGETKIND(_CTK) == COMMA && TGETKIND(_CTK) != EOI) {
                consume("", COMMA);
                /* TODO: you can't have params here, so ensure that path is never taken in this call path */
                uint64_t type = 0;  // dummy
                declarator(&type, 1);
                if (TGETKIND(_CTK) == ASSIGN) {
                        consume("", ASSIGN);
                        initializer();
                }
        }
}

// storage-class-specifier = 'typedef'
//                         | 'extern'
//                         | 'static'
//                         | 'auto'
//                         | 'register'
void sclass(uint64_t *type) {
        enum Kind ctk = TGETKIND(_CTK);
        if (ctk >= TYPEDEF && ctk <= REGISTER) {
                if (((*type) & TYPE_TYPEDEF) || ((*type) & TYPE_EXTERN) ||
                    ((*type) & TYPE_STATIC)) {
                        int line = TGETROW(_CTK);
                        error("More than one storage-class specifier in line %d\n", line);
                }
                /* TODO: 6.7.1.5 */

                if (ctk == EXTERN)
                        (*type) |= TYPE_EXTERN;
                else if (ctk == TYPEDEF)
                        (*type) |= TYPE_TYPEDEF;
                else if (ctk == TYPEDEF)
                        (*type) |= TYPE_STATIC;
                consume("", ctk);
        }
}

// type-specifier = 'void'
//                | 'char'
//                | 'short'
//                | 'int'
//                | 'long'
//                | 'float'
//                | 'double'
//                | 'signed'
//                | 'unsigned'
//                | struct-or-union-specifier
//                | enum-specifier
//                | typedef-name
void typespec(uint64_t *type) {
        enum Kind ctk = TGETKIND(_CTK);
        if (ctk >= VOID && ctk <= ENUM) {
                if (ctk == STRUCT || ctk == UNION) {
                        if (ctk == STRUCT)
                                (*type) |= TYPE_STRUCT;
                        else
                                (*type) |= TYPE_UNION;
                        structorunionspec();
                } else if (ctk == ENUM) {
                        (*type) |= TYPE_ENUM;
                        enumspec();
                } else if (ctk >= VOID && ctk <= UNSIGNED) {
                        bool foundBase = (*type) & TYPE_BMASK;
                        bool foundModifier = (*type) & TYPE_MMASK;
                        bool s = (*type) & TYPE_SHORT;
                        bool l = (*type) & TYPE_LONG;
                        bool sg = (*type) & TYPE_SIGNED;
                        bool usg = (*type) & TYPE_UNSIGNED;
                        switch (ctk) {
                                case VOID:
                                        (*type) |= TYPE_VOID;
                                        if (foundBase || foundModifier) {
                                        tserror:
                                                error("Too many type-specifiers in line %d\n",
                                                      TGETROW(_CTK));
                                                break;
                                        }
                                        break;
                                case CHAR:
                                        if (foundBase) goto tserror;
                                        if (foundModifier && (s || l)) goto tserror;
                                        (*type) |= TYPE_CHAR;
                                        break;
                                case INT:
                                        if (foundBase) goto tserror;
                                        if (foundModifier) {
                                                if ((s && l) || (sg && usg)) goto tserror;
                                        }
                                        (*type) |= TYPE_INT;
                                        break;
                                case FLOAT:
                                        if (foundBase || foundModifier) goto tserror;
                                        (*type) |= TYPE_FLOAT;
                                        break;
                                case DOUBLE:
                                        if (foundBase || (foundModifier && !l)) goto tserror;
                                        (*type) |= TYPE_DOUBLE;
                                        break;
                                case SHORT:
                                        if (foundBase && (((*type) & TYPE_BMASK) != TYPE_INT))
                                                goto tserror;
                                        if (foundModifier && l) goto tserror;
                                        (*type) |= TYPE_SHORT;
                                        break;
                                case LONG:
                                        if (foundBase &&
                                            !(((*type) & TYPE_BMASK) & (TYPE_INT | TYPE_DOUBLE)))
                                                goto tserror;
                                        if (foundModifier && s) goto tserror;
                                        (*type) |= TYPE_LONG;
                                        break;
                                case SIGNED:
                                        if (foundBase &&
                                            !(((*type) & TYPE_BMASK) & (TYPE_INT | TYPE_CHAR)))
                                                goto tserror;
                                        if (foundModifier && usg) goto tserror;
                                        (*type) |= TYPE_SIGNED;
                                        break;
                                case UNSIGNED:
                                        if (foundBase &&
                                            !(((*type) & TYPE_BMASK) & (TYPE_INT | TYPE_CHAR)))
                                                goto tserror;
                                        if (foundModifier && sg) goto tserror;
                                        (*type) |= TYPE_UNSIGNED;
                                        break;
                                default: assert(0);
                        }
                        consume("", ctk);
                }
        }
}

// (* NOTE: Define typedef-name as result of 'typedef'. *)
// typedef-name = identifier;

// type-qualifier = 'const'
//                | 'restrict'
//                | 'volatile'
void typequal(uint64_t *type) {
        enum Kind ctk = TGETKIND(_CTK);
        if (ctk >= CONST && ctk <= VOLATILE) {
                if (ctk == CONST) (*type) |= TYPE_CONST;
                consume("", ctk);
        }
}

// function-specifier = 'inline'
void funcspec(uint64_t *type) {
        if (TGETKIND(_CTK) == INLINE) consume("", TGETKIND(_CTK));
}

// pointer = '*' [type-qualifier-list] [pointer]
void pointer(uint64_t *type, int *level) {
        consume("", MUL);
        /* buid compound type */
        (*type) |= (uint64_t)(TYPE_PTR << (24 + (*level) * 2));
        (*level)++;
        if (TGETKIND(_CTK) >= CONST && TGETKIND(_CTK) <= VOLATILE) typequalifierlist();
        if (TGETKIND(_CTK) == MUL) {
                pointer(type, level);
        }
}

/*
   abstract-declarators allow you to operate on types, without referencing an object.
   (e.g., in function prototypes, cast expressions, and sizeof arguments)
*/

// direct-declarator = identifier
//                   | '(' declarator ')'
// suffic-declarator = '[' ['*'] ']'
//                   | '[' 'static' [type-qualifier-list] assignment-expression ']'
//                   | '[' type-qualifier-list ['*'] ']'
//                   | '[' type-qualifier-list ['static'] assignment-expression ']'
//                   | '[' assignment-expression ']'
//                   | '(' parameter-type-list ')'
//                   | '(' ')'
void directdeclarator(uint64_t *type, int level) {
        if (TGETKIND(_CTK) == IDENT)
                consume("", IDENT);
        else if (TGETKIND(_CTK) == OPAR) {
                // abstract
                consume("", OPAR);
                declarator(type, level + 1);
                consume("missing ')' of direct declarator", CPAR);
        } else if (TGETKIND(_CTK) == OBR)
                ;
        else
                return;

        while ((TGETKIND(_CTK) == OPAR || TGETKIND(_CTK) == OBR) && TGETKIND(_CTK) != EOI) {
                if (TGETKIND(_CTK) == OPAR) {
                        // concrete function
                        consume("", OPAR);

                        /* buid compound type */
                        (*type) |= (uint64_t)(TYPE_FUNC << (24 + level * 2));
                        level++;

                        // params
                        if (TGETKIND(_CTK) >= VOID && TGETKIND(_CTK) <= ENUM) {
                                if (TGETKIND(_CTK) == VOID)
                                        consume("", VOID);
                                else
                                        paramtypelist();
                        }
                        consume("missing ')' of function definition/declaration", CPAR);
                } else {
                        // array
                        consume("", OBR);

                        /* buid compound type */
                        (*type) |= (uint64_t)(TYPE_ARRAY << (24 + level * 2));
                        level++;

                        if (TGETKIND(_CTK) == MUL) {
                                consume("", MUL);
                        } else if (TGETKIND(_CTK) == STATIC) {
                                consume("", STATIC);
                                if (TGETKIND(_CTK) >= CONST && TGETKIND(_CTK) <= VOLATILE)
                                        typequalifierlist();
                                assignexpr();
                        } else if (TGETKIND(_CTK) >= CONST && TGETKIND(_CTK) <= VOLATILE) {
                                typequalifierlist();
                                if (TGETKIND(_CTK) == MUL)
                                        consume("", MUL);
                                else if (TGETKIND(_CTK) != CBR) {
                                        if (TGETKIND(_CTK) == STATIC) consume("", STATIC);
                                        assignexpr();
                                }
                        } else if (TGETKIND(_CTK) != CBR) {
                                assignexpr();
                        }
                        consume("missing '[' of array declaration", CBR);
                }
        }
}

// initializer-list = designative-initializer {',' designative-initializer}
void initializerlist() {
        designinitzer();
        while (TGETKIND(_CTK) == COMMA) {
                consume("", COMMA);
                /*
                  check to handle trailing comma in an initializer-list
                  int y[4][3] = {1, 3, 5,};
                                        ^
                  ideally, it should be handled in initializer where you have [','],
                  suggesting that this 'while' loop way might be wrong approach
                  to parse initializerlist.
                */
                if (TGETKIND(_CTK) != CCBR) designinitzer();
        }
}

// designative-initializer = [designation] initializer
void designinitzer() {
        if (TGETKIND(_CTK) == OBR || TGETKIND(_CTK) == DOT) designation();
        initializer();
}

// initializer = '{' initializer-list [','] '}'
//             | assignment-expression;
void initializer() {
        if (TGETKIND(_CTK) == OCBR) {
                consume("", OCBR);
                initializerlist();
                if (TGETKIND(_CTK) == COMMA) consume("", COMMA);
                consume("missing '}' of initializer", CCBR);
        } else {
                assignexpr();
        }
}

// constant-expression = conditional-expression  (* with constraints *)
void constexpr() {
        /* TODO: add constraints */
        condexpr();
}

// struct-or-union-specifier = struct-or-union '{' struct-declaration-list '}'
//                           | struct-or-union identifier ['{' struct-declaration-list '}']
//                           | struct-or-union identifier
void structorunionspec() {
        if (TGETKIND(_CTK) == STRUCT || TGETKIND(_CTK) == UNION) {
                consume("", TGETKIND(_CTK));
                if (TGETKIND(_CTK) == IDENT) {
                        consume("", IDENT);
                }
                if (TGETKIND(_CTK) == OCBR) {
                        consume("", OCBR);
                        structdeclarationlist();
                        consume("missing '}' of struct\n", CCBR);
                }
        }
}

// struct-or-union = 'struct'
//                 | 'union'

// struct-declaration-list = struct-declaration { struct-declaration }
void structdeclarationlist() {
        while (TGETKIND(_CTK) != CCBR && TGETKIND(_CTK) != EOI) {
                structdeclaration();
        }
}

// struct-declaration = specifier-qualifier-list struct-declarator-list ';'
void structdeclaration() {
        specquallist();
        if (TGETKIND(_CTK) == SEMIC)
                consume("", SEMIC);
        else {
                structdeclaratorlist();
                consume("missing ';' of struct member\n", SEMIC);
        }
}

// enum-specifier = 'enum' '{' enumerator-list [','] '}'
//                | 'enum' identifier ['{' enumerator-list [','] '}']
void enumspec() {
        consume("", ENUM);
        if (TGETKIND(_CTK) == IDENT) consume("", IDENT);
        if (TGETKIND(_CTK) == OCBR) {
                consume("", OCBR);
                enumtorlist();
                while (TGETKIND(_CTK) == COMMA && TGETKIND(_CTK) != EOI) enumtorlist();
                consume("missing '}' of enum\n", CCBR);
        }
}

// enumerator-list = enumerator {',' enumerator}
void enumtorlist() {
        enumtor();
        while (TGETKIND(_CTK) == COMMA && TGETKIND(_CTK) != EOI) {
                consume("", COMMA);
                enumtor();
        }
}

// (* NOTE: Please define enumeration-constant for identifier inside enum { ... }. *)
// enumerator = identifier ['=' constant-expression];
void enumtor() {
        consume("", IDENT);
        if (TGETKIND(_CTK) == ASSIGN) {
                consume("", ASSIGN);
                constexpr();
        }
}

// enumeration-constant = identifier

// type-name = specifier-qualifier-list [abstract-declarator]
void typename() {
        specquallist();
        enum Kind ctk = TGETKIND(_CTK);
        if (ctk == MUL || ctk == OPAR || ctk == OBR) {
                uint64_t type = 0;  // dummy
                declarator(&type, 1);
        }
}

// specifier-qualifier-list = specifier-qualifier {specifier-qualifier}
void specquallist() {
        while (TGETKIND(_CTK) >= CONST && TGETKIND(_CTK) <= ENUM && TGETKIND(_CTK) != INLINE &&
               TGETKIND(_CTK) != EOI) {
                specqual();
        }
}

// specifier-qualifier = type-specifier | type-qualifier
void specqual() {
        uint64_t type = 0;  // dummy
        if (TGETKIND(_CTK) >= VOID && TGETKIND(_CTK) <= ENUM) {
                typespec(&type);
        } else if (TGETKIND(_CTK) >= CONST && TGETKIND(_CTK) <= VOLATILE) {
                typequal(&type);
        }
}

// struct-declarator-list = struct-declarator {',' struct-declarator}
void structdeclaratorlist() {
        structdeclarator();
        while (TGETKIND(_CTK) == COMMA && TGETKIND(_CTK) != EOI) {
                consume("", COMMA);
                structdeclarator();
        }
}

// type-qualifier-list = type-qualifier {type-qualifier}
void typequalifierlist() {
        uint64_t type = 0;  // dummy
        typequal(&type);
        while (TGETKIND(_CTK) >= CONST && TGETKIND(_CTK) <= VOLATILE) consume("", TGETKIND(_CTK));
}

// parameter-type-list = parameter-list [',' '...']
void paramtypelist() {
        paramlist();
        if (TGETKIND(_CTK) == COMMA) {
                consume("", COMMA);
                consume("missing '...' of parameter-type-list\n", ELLIPSIS);
        }
}

// parameter-list = parameter-declaration {',' parameter-declaration}
void paramlist() {
        paramdeclaration();
        while (TGETKIND(_CTK) == COMMA && TGETKIND(_CTK) != EOI &&
               TGETKIND(tokens[_INDEX + 1]) >= TYPEDEF && TGETKIND(tokens[_INDEX + 1]) <= ENUM) {
                consume("", COMMA);
                paramdeclaration();
        }
}

// parameter-declaration = declaration-specifiers [declarator | abstract-declarator]
void paramdeclaration() {
        uint64_t type = declspec();
        /* NOTE: directdeclarator() parses abstract-declarator */
        declarator(&type, 1);
}

// struct-declarator = ':' constant-expression
//                   | declarator [':' constant-expression]
void structdeclarator() {
        if (TGETKIND(_CTK) == COLON) {
                consume("", COLON);
                constexpr();
                return;
        }
        uint64_t type = 0;
        declarator(&type, 1);
        if (TGETKIND(_CTK) == COLON) {
                consume("", COLON);
                constexpr();
        }
}

// expression = assignment-expression {',' assignment-expression}
void expr() {
        assignexpr();
        while (TGETKIND(_CTK) == COMMA) {
                consume("", COMMA);
                assignexpr();
        }
}

// assignment-expression = conditional-expression
//                       | unary-expression assignment-operator assignment-expression
void assignexpr() {
        condexpr();
        if (TGETKIND(_CTK) >= ASSIGN && TGETKIND(_CTK) <= RSHASSIGN) {
                consume("", TGETKIND(_CTK));  // assignment-operator
                assignexpr();
        }
}

// assignment-operator = '=' | '*=' | '/=' | '%=' | '+=' | '-=' | '<<=' | '>>=' | '&=' | '^=' | '|='

// conditional-expression = binary-expression [ '?' expression ':' conditional-expression ]
void condexpr() {
        binaryexpr();
        if (TGETKIND(_CTK) == QMARK) {
                consume("", QMARK);
                expr();
                consume("missing ':' of conditional expr", COLON);
                condexpr();
        }
}

// binary-expression = unary-expression { binary-operator unary-expression }
void binaryexpr() {
        unaryexpr();
        while ((TGETKIND(_CTK) >= OROR && TGETKIND(_CTK) <= MOD)) {
                consume("", TGETKIND(_CTK));
                unaryexpr();
        }
}

// binary-operator = '||' | '&&' | '|' | '^' | '&' | '==' | '!=' | '<' | '>' | '<=' | '>=' | '<<'
//                 | '>>' | '+' | '-' | '*' | '/' | '%'

// unary-expression = postfix-expression
//                  | unary-operator unary-expression
//                  | '(' type-name ')' unary-expression      /* cast or compound literal */
//                  | 'sizeof' unary-expression
//                  | 'sizeof' '(' type-name ')'
void unaryexpr() {
        enum Kind ctk = TGETKIND(_CTK);
        if (ctk == INCR || ctk == DECR || ctk == AND || ctk == MUL || ctk == ADD || ctk == SUB ||
            ctk == TILDA || ctk == NOT) {
                consume("", ctk);
                unaryexpr();
        } else if (ctk == OPAR) {
                if (TGETKIND(tokens[_INDEX + 1]) >= CONST && TGETKIND(tokens[_INDEX + 1]) <= ENUM) {
                        consume("", OPAR);
                        typename();
                        consume("missing ')' of unary expr", CPAR);
                        unaryexpr();
                } else {
                        postfixexpr();
                }
        } else if (ctk == SIZEOF) {
                consume("", ctk);
                if (TGETKIND(_CTK) == OPAR) {
                        consume("", OPAR);
                        typename();
                        consume("missing ')' of unary expr", CPAR);
                        return;
                }
                unaryexpr();
        } else {
                postfixexpr();
        }
}

// unary-operator = '++' | '--' | '&' | '*' | '+' | '-' | '~' | '!'

// postfix-expression = primary-expression { postfix-operator }
//                    | '{' initializer-list [','] '}'          /* compound literal */
void postfixexpr() {
        if (TGETKIND(_CTK) == OCBR) {
                consume("", OCBR);
                initializerlist();
                if (TGETKIND(_CTK) == COMMA) consume("", COMMA);
                consume("missing '{' of postfix expr", CCBR);
                return;
        }
        primaryexpr();
        enum Kind ctk = TGETKIND(_CTK);
        while (ctk == OBR || ctk == OPAR || ctk == DOT || ctk == DEREF || ctk == INCR ||
               ctk == DECR || ctk == OCBR) {
                postfixoperator();
                ctk = TGETKIND(_CTK);
        }
}

// postfix-operator = '[' expression ']'
//                  | '(' [assignment-expression {',' assignment-expression}] ')'
//                  | ('.' | '->') identifier
//                  | ('++' | '--')
void postfixoperator() {
        switch (TGETKIND(_CTK)) {
                case OBR:
                        consume("", OBR);
                        expr();
                        consume("missing '[' of postfix operator", CBR);
                        break;
                case OPAR:
                        consume("", OPAR);
                        if (TGETKIND(_CTK) != CPAR) {
                                assignexpr();
                                while (TGETKIND(_CTK) == COMMA) {
                                        consume("", COMMA);
                                        assignexpr();
                                }
                        }
                        consume("missing ')' of postfix operator", CPAR);
                        break;
                case DOT:
                case DEREF:
                        consume("", TGETKIND(_CTK));
                        consume("missing 'identity' of postfix operator", IDENT);
                        break;
                case INCR:
                case DECR: consume("", TGETKIND(_CTK)); break;
                default: error("unknown postfix operator: %s\n", token_names[TGETKIND(_CTK)]);
        }
}

// primary-expression = identifier
//                    | constant
//                    | string
//                    | '(' expression ')'
void primaryexpr() {
        switch (TGETKIND(_CTK)) {
                case IDENT:
                case ICON:
                case SCON:
                case CCON:
                case FCON:
                case DCON:
                case LDCON: consume("", TGETKIND(_CTK)); break;
                case OPAR:
                        consume("", OPAR);
                        expr();
                        consume("missing ')' of primary expr", CPAR);
                        break;
                default: error("unknown primary expr: %s\n", token_names[TGETKIND(_CTK)]);
        }
}

// constant = integer-constant
//          | character-constant
//          | floating-constant
//          | enumeration-constant;
void constant() {
        switch (TGETKIND(_CTK)) {
                case ICON:
                case CCON:
                case FCON: consume("", TGETKIND(_CTK)); break;
                default: error("unknown constant: %s\n", token_names[TGETKIND(_CTK)]);
        }
}

// string = string-literal
//        | '__func__';

// designation = designator-list '='
void designation() {
        designtorlist();
        if (TGETKIND(_CTK) == ASSIGN) consume("", ASSIGN);
}

// designator-list = designator {designator}
void designtorlist() {
        designator();
        while (TGETKIND(_CTK) == OBR || TGETKIND(_CTK) == DOT) designator();
}

// designator = '[' constant-expression ']'
//            | '.' identifier
void designator() {
        if (TGETKIND(_CTK) == OBR) {
                consume("", OBR);
                constexpr();
                consume("missing '[' of designator", CBR);
        } else {
                assert(TGETKIND(_CTK) == DOT);
                consume("", DOT);
                consume("missing 'identity' of designator", IDENT);
        }
}

// statement = labeled-statement
//           | compound-statement
//           | expression-statement
//           | selection-statement
//           | iteration-statement
//           | jump-statement
void stmt() {
        /*
          TODO: refactor given every stmt has a clear indicator except the exprstmt,
          which should be in the else part.
        */
        enum Kind ctk = TGETKIND(_CTK);
        if (ctk == IDENT || ctk == CASE || ctk == DEFAULT || ctk == SEMIC || ctk == OPAR ||
            ctk == MUL) {
                if ((ctk == IDENT && TGETKIND(tokens[_INDEX + 1]) == COLON) || ctk == CASE ||
                    ctk == DEFAULT)
                        labelstmt();
                else
                        exprstmt();
        } else if (ctk >= GOTO && ctk <= RETURN) {
                jumpstmt();
        } else if (ctk >= FOR && ctk <= DO) {
                iterstmt();
        } else if (ctk == IF || ctk == SWITCH) {
                selectstmt();
        } else {
                compstmt();
        }
}

// labeled-statement = identifier ':' statement
//                   | 'case' constant-expression ':' statement
//                   | 'default' ':' statement
void labelstmt() {
        enum Kind ctk = TGETKIND(_CTK);
        if (ctk == IDENT) {
                consume("", IDENT);
                consume("missing ':' of label stmt", COLON);
                stmt();
        } else if (ctk == CASE) {
                consume("", CASE);
                constexpr();
                consume("missing ':' of label stmt", COLON);
                stmt();
        } else {
                assert(ctk == DEFAULT);
                consume("", DEFAULT);
                consume("missing ':' of label stmt", COLON);
                stmt();
        }
}

// expression-statement = [expression] ';'
void exprstmt() {
        if (TGETKIND(_CTK) != SEMIC) expr();
        consume("missing ';' of expr stmt", SEMIC);
}

// selection-statement = 'if' '(' expression ')' statement 'else' statement
//                     | 'if' '(' expression ')' statement
//                     | 'switch' '(' expression ')' statement
void selectstmt() {
        enum Kind ctk = TGETKIND(_CTK);
        if (ctk == IF) {
                consume("", IF);
                consume("missing '(' of if-stmt", OPAR);
                expr();
                consume("missing ')' of if-stmt", CPAR);
                stmt();
                if (TGETKIND(_CTK) == ELSE) {
                        consume("", ELSE);
                        stmt();
                }
        } else {
                assert(ctk == SWITCH);
                consume("", SWITCH);
                consume("missing '(' of switch-stmt", OPAR);
                expr();
                consume("missing ')' of switch-stmt", CPAR);
                stmt();
        }
}

//  iteration-statement = 'while' '(' expression ')' statement
//                      | 'do' statement 'while' '(' expression ')' ';'
//                      | 'for' '(' [expression] ';' [expression] ';' [expression] ')' statement
//                      | 'for' '(' declaration [expression] ';' [expression] ')' statement
void iterstmt() {
        enum Kind ctk = TGETKIND(_CTK);
        if (ctk == FOR) {
                consume("", FOR);
                consume("missing '(' of for-stmt", OPAR);
                if (TGETKIND(_CTK) >= VOID && TGETKIND(_CTK) <= ENUM) {
                        declaration();
                } else {
                        if (TGETKIND(_CTK) != SEMIC) expr();
                        consume("missing ';' of for-stmt", SEMIC);
                }
                if (TGETKIND(_CTK) != SEMIC) expr();
                consume("missing ';' of for-stmt", SEMIC);
                if (TGETKIND(_CTK) != CPAR) expr();
                consume("missing ')' of for-stmt", CPAR);
                stmt();
        } else if (ctk == DO) {
                consume("", DO);
                stmt();
                consume("missing 'while' of do-stmt", WHILE);
                consume("missing '(' of do-stmt", OPAR);
                expr();
                consume("missing ')' of do-stmt", CPAR);
                consume("missing ';' of do-stmt", SEMIC);
        } else {
                assert(ctk == WHILE);
                consume("", WHILE);
                consume("missing '(' of while-stmt", OPAR);
                expr();
                consume("missing ')' of while-stmt", CPAR);
                stmt();
        }
}

// jump-statement = 'goto' identifier ';'
//                | 'continue' ';'
//                | 'break' ';'
//                | 'return' [expression] ';'
void jumpstmt() {
        enum Kind ctk = TGETKIND(_CTK);
        if (ctk == RETURN) {
                consume("", ctk);
                if (TGETKIND(_CTK) != SEMIC) expr();
        } else if (ctk == BREAK) {
                consume("", ctk);
        } else if (ctk == CONTINUE) {
                consume("", ctk);
        } else if (ctk == GOTO) {
                consume("", ctk);
                consume("missing 'identity' of goto-stmt", IDENT);
        } else {
                error("invalid jump statement: %s\n", token_names[ctk]);
        }
        consume("missing ';' of jump stmt", SEMIC);
}
