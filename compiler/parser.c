#pragma clang diagnostic ignored "-Wgnu-empty-initializer"
#pragma clang diagnostic ignored "-Wstrict-prototypes"

#include "ganymede.h"

/* global variables */
static struct Token *_ct;
/* if GLOBAL, first declarator is parsed upfront to diff funcdef */
enum { GLOBAL = 1, LOCAL, PARAM } _cdecllevel;

/* utility functions */
static inline void consume(const char *msg, enum Kind kind) {
        if (_ct->kind != kind) {
                error("%s: expected %s, got %s\n", msg, token_names[kind], token_names[_ct->kind]);
        }
        _ct = _ct->next;
}

/* ---------- RECURSIVE DESCENT PARSER ---------- */

// https://github.com/katef/kgt/blob/main/examples/c99-grammar.iso-ebnf

void jumpstmt();
void stmt();
void directdeclarator();
void pointer();
void funcspec();
void typequal();
void typespec();
void sclass();
void declorstmt();
void compstmt();
void declarator();
void declspec();
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

void parse(struct Token *token);

// translation-unit = {external-declaration}
void parse(struct Token *token) {
        _ct = token;
        while (_ct->kind != EOI) {
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
        declspec();
        declarator();
        /* TODO: resolve type info here since declarator collected it */
        if (_cdecllevel == GLOBAL && _ct->kind == OCBR) {
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
void declspec() {
        while (_ct->kind >= TYPEDEF && _ct->kind <= ENUM) {
                sclass();
                typespec();
                typequal();
                funcspec();
        }
}

// declarator = [pointer] direct-declarator {suffix-declarator}
void declarator() {
        if (_ct->kind == MUL) pointer();
        directdeclarator();
}

// declaration-list = declaration {declaration}

// compound-statement = '{' {declaration-or-statement} '}'
void compstmt() {
        consume("missing '{' of compound statement", OCBR);
        while (_ct->kind != CCBR && _ct->kind != EOI) {
                declorstmt();
        }
        consume("missing '}' of compound statement", CCBR);
}

// declaration-or-statement = declaration | statement
void declorstmt() {
        if (_ct->kind >= TYPEDEF && _ct->kind <= ENUM) {
                _cdecllevel = LOCAL;
                declaration();
        } else {
                stmt();
        }
}

// init-declarator-list = ['=' initializer] {',' declarator ['=' initializer]}
void initdeclaratorlist() {
        /* first declarator is already parsed. now see if it has initializer */
        if (_ct->kind == ASSIGN) {
                consume("", ASSIGN);
                initializer();
        }
        /* rest of the declarators */
        while (_ct->kind == COMMA && _ct->kind != EOI) {
                consume("", COMMA);
                /* TODO: you can't have params here, so ensure that path is never taken in this call path */
                declarator();
                if (_ct->kind == ASSIGN) {
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
void sclass() {
        enum Kind ctk = _ct->kind;
        if (ctk >= TYPEDEF && ctk <= REGISTER) consume("", ctk);
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
void typespec() {
        enum Kind ctk = _ct->kind;
        if (ctk >= VOID && ctk <= ENUM) {
                if (ctk == STRUCT || ctk == UNION) {
                        structorunionspec();
                } else if (ctk == ENUM) {
                        enumspec();
                } else {
                        consume("", ctk);
                }
        }
}

// (* NOTE: Please define typedef-name as result of 'typedef'. *)
// typedef-name = identifier;

// type-qualifier = 'const'
//                | 'restrict'
//                | 'volatile'
void typequal() {
        enum Kind ctk = _ct->kind;
        if (ctk >= CONST && ctk <= VOLATILE) consume("", ctk);
}

// function-specifier = 'inline'
void funcspec() {
        if (_ct->kind == INLINE) consume("", _ct->kind);
}

// pointer = '*' [type-qualifier-list] [pointer]
void pointer() {
        consume("", MUL);
        if (_ct->kind >= CONST && _ct->kind <= VOLATILE) typequalifierlist();
        if (_ct->kind == MUL) pointer();
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
void directdeclarator() {
        if (_ct->kind == IDENT)
                consume("", IDENT);
        else if (_ct->kind == OPAR) {
                // abstract
                consume("", OPAR);
                declarator();
                consume("", CPAR);
        } else if (_ct->kind == OBR)
                ;
        else
                return;

        while ((_ct->kind == OPAR || _ct->kind == OBR) && _ct->kind != EOI) {
                if (_ct->kind == OPAR) {
                        // concrete function
                        consume("", OPAR);
                        // params
                        if (_ct->kind >= VOID && _ct->kind <= ENUM) {
                                if (_ct->kind == VOID)
                                        consume("", VOID);
                                else
                                        paramtypelist();
                        }
                        consume("missing ')' of function definition/declaration", CPAR);
                } else if (_ct->kind == OBR) {
                        // array
                        consume("", OBR);
                        if (_ct->kind == MUL) {
                                consume("", MUL);
                        } else if (_ct->kind == STATIC) {
                                consume("", STATIC);
                                if (_ct->kind >= CONST && _ct->kind <= VOLATILE)
                                        typequalifierlist();
                                assignexpr();
                        } else if (_ct->kind >= CONST && _ct->kind <= VOLATILE) {
                                typequalifierlist();
                                if (_ct->kind == MUL)
                                        consume("", MUL);
                                else {
                                        if (_ct->kind == STATIC) consume("", STATIC);
                                        assignexpr();
                                }
                        } else if (_ct->kind != CBR) {
                                assignexpr();
                        }
                        consume("missing '[' of array declaration", CBR);
                } else
                        assert(0);
        }
}

// initializer-list = designative-initializer {',' designative-initializer}
void initializerlist() {
        //
        designinitzer();
        while (_ct->kind == COMMA) {
                consume("", COMMA);
                /* 
                  check to handle trailing comma in an initializer-list
                  int y[4][3] = {1, 3, 5,};
                                        ^
                  ideally, it should be handled in initializer where you have [','],
                  suggesting that this 'while' loop way might be wrong approach 
                  to parse initializerlist. 
                */
                if (_ct->kind != CCBR) designinitzer();
        }
}

// designative-initializer = [designation] initializer
void designinitzer() {
        if (_ct->kind == OBR || _ct->kind == DOT) designation();
        initializer();
}

// initializer = '{' initializer-list [','] '}'
//             | assignment-expression;
void initializer() {
        if (_ct->kind == OCBR) {
                consume("", OCBR);
                initializerlist();
                if (_ct->kind == COMMA) consume("", COMMA);
                consume("", CCBR);
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
        if (_ct->kind == STRUCT || _ct->kind == UNION) {
                consume("", _ct->kind);
                if (_ct->kind == IDENT) {
                        consume("", IDENT);
                }
                if (_ct->kind == OCBR) {
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
        while (_ct->kind != CCBR && _ct->kind != EOI) {
                structdeclaration();
        }
}

// struct-declaration = specifier-qualifier-list struct-declarator-list ';'
void structdeclaration() {
        specquallist();
        if (_ct->kind == SEMIC)
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
        if (_ct->kind == IDENT) consume("", IDENT);
        if (_ct->kind == OCBR) {
                consume("", OCBR);
                enumtorlist();
                while (_ct->kind == COMMA && _ct->kind != EOI) enumtorlist();
                consume("missing '}' of enum\n", CCBR);
        }
}

// enumerator-list = enumerator {',' enumerator}
void enumtorlist() {
        enumtor();
        while (_ct->kind == COMMA && _ct->kind != EOI) {
                consume("", COMMA);
                enumtor();
        }
}

// (* NOTE: Please define enumeration-constant for identifier inside enum { ... }. *)
// enumerator = identifier ['=' constant-expression];
void enumtor() {
        consume("", IDENT);
        if (_ct->kind == ASSIGN) {
                consume("", ASSIGN);
                constexpr();
        }
}

// enumeration-constant = identifier

// type-name = specifier-qualifier-list [abstract-declarator]
void typename() {
        //
        specquallist();
        enum Kind ctk = _ct->kind;
        if (ctk == MUL || ctk == OPAR || ctk == OBR) declarator();
}

// specifier-qualifier-list = specifier-qualifier {specifier-qualifier}
void specquallist() {
        //
        while (_ct->kind >= CONST && _ct->kind <= ENUM && _ct->kind != INLINE && _ct->kind != EOI) {
                specqual();
        }
}

// specifier-qualifier = type-specifier | type-qualifier
void specqual() {
        if (_ct->kind >= VOID && _ct->kind <= ENUM) {
                typespec();
        } else if (_ct->kind >= CONST && _ct->kind <= VOLATILE) {
                typequal();
        }
}

// struct-declarator-list = struct-declarator {',' struct-declarator}
void structdeclaratorlist() {
        structdeclarator();
        while (_ct->kind == COMMA && _ct->kind != EOI) {
                consume("", COMMA);
                structdeclarator();
        }
}

// type-qualifier-list = type-qualifier {type-qualifier}
void typequalifierlist() {
        typequal();
        while (_ct->kind >= CONST && _ct->kind <= VOLATILE) consume("", _ct->kind);
}

// parameter-type-list = parameter-list [',' '...']
void paramtypelist() {
        paramlist();
        if (_ct->kind == COMMA) {
                consume("", COMMA);
                consume("missing '...' of parameter-type-list\n", ELLIPSIS);
        }
}

// parameter-list = parameter-declaration {',' parameter-declaration}
void paramlist() {
        paramdeclaration();
        while (_ct->kind == COMMA && _ct->kind != EOI) {
                consume("", COMMA);
                paramdeclaration();
        }
}

// parameter-declaration = declaration-specifiers [declarator | abstract-declarator]
void paramdeclaration() {
        declspec();
        /* NOTE: directdeclarator() parses abstract-declarator */
        declarator();
}

// struct-declarator = ':' constant-expression
//                   | declarator [':' constant-expression]
void structdeclarator() {
        //
        if (_ct->kind == COLON) {
                consume("", COLON);
                constexpr();
                return;
        }
        declarator();
        if (_ct->kind == COLON) {
                consume("", COLON);
                consume("", INTCONST);
        }
}

// expression = assignment-expression {',' assignment-expression}

void expr() {
        assignexpr();
        while (_ct->kind == COMMA) {
                consume("", COMMA);
                assignexpr();
        }
}

// assignment-expression = conditional-expression
//                       | unary-expression assignment-operator assignment-expression
void assignexpr() {
        condexpr();
        if (_ct->kind >= ASSIGN && _ct->kind <= RSHASSIGN) {
                consume("", _ct->kind);  // assignment-operator
                assignexpr();
        }
}

// assignment-operator = '=' | '*=' | '/=' | '%=' | '+=' | '-=' | '<<=' | '>>=' | '&=' | '^=' | '|='

// conditional-expression = binary-expression [ '?' expression ':' conditional-expression ]
void condexpr() {
        binaryexpr();
        if (_ct->kind == QMARK) {
                consume("", QMARK);
                expr();
                consume("", COLON);
                condexpr();
        }
}

// binary-expression = unary-expression { binary-operator unary-expression }
void binaryexpr() {
        unaryexpr();
        while ((_ct->kind >= OROR && _ct->kind <= MOD)) {
                consume("", _ct->kind);
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
        enum Kind ctk = _ct->kind;
        if (ctk == INCR || ctk == DECR || ctk == AND || ctk == MUL || ctk == ADD || ctk == SUB ||
            ctk == TILDA || ctk == NOT) {
                consume("", ctk);
                unaryexpr();
        } else if (ctk == OPAR) {
                /* 
                   Note: stylistically, we should've consumed '(' and called 'typename()';
                   we don't follow it here because type-name has abstract-declarator, which
                   needs '(' to be recognized in the 'ddeclarator()', called in 'typename()') 
                */
                if (_ct->next->kind >= CONST && _ct->next->kind <= ENUM) {
                        consume("", OPAR);
                        typename();
                        consume("", CPAR);
                        unaryexpr();
                } else {
                        postfixexpr();
                }
        } else if (ctk == SIZEOF) {
                consume("", ctk);
                if (_ct->kind == OPAR) {
                        consume("", OPAR);
                        typename();
                        consume("", CPAR);
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
        if (_ct->kind == OCBR) {
                consume("", OCBR);
                initializerlist();
                if (_ct->kind == COMMA) consume("", COMMA);
                consume("", CCBR);
                return;
        }
        primaryexpr();
        enum Kind ctk = _ct->kind;
        while (ctk == OBR || ctk == OPAR || ctk == DOT || ctk == DEREF || ctk == INCR ||
               ctk == DECR || ctk == OCBR) {
                postfixoperator();
                ctk = _ct->kind;
        }
}

// postfix-operator = '[' expression ']'
//                  | '(' [assignment-expression {',' assignment-expression}] ')'
//                  | ('.' | '->') identifier
//                  | ('++' | '--')
void postfixoperator() {
        switch (_ct->kind) {
                case OBR:
                        consume("", OBR);
                        expr();
                        consume("", CBR);
                        break;
                case OPAR:
                        consume("", OPAR);
                        if (_ct->kind != CPAR) {
                                assignexpr();
                                while (_ct->kind == COMMA) {
                                        consume("", COMMA);
                                        assignexpr();
                                }
                        }
                        consume("", CPAR);
                        break;
                case DOT:
                case DEREF:
                        consume("", _ct->kind);
                        consume("", IDENT);
                        break;
                case INCR:
                case DECR: consume("", _ct->kind); break;
                default: assert(0);
        }
}

// primary-expression = identifier
//                    | constant
//                    | string
//                    | '(' expression ')'
void primaryexpr() {
        switch (_ct->kind) {
                case IDENT:
                case INTCONST:
                case STRCONST:
                case CHARCONST:
                case FLOATCONST:
                case DOUBLECONST: consume("", _ct->kind); break;
                case OPAR:
                        consume("", OPAR);
                        expr();
                        consume("", CPAR);
                        break;
                default: assert(0);
        }
}

// constant = integer-constant
//          | character-constant
//          | floating-constant
//          | enumeration-constant;
void constant() {
        switch (_ct->kind) {
                case INTCONST:
                case CHARCONST:
                case FLOATCONST: consume("", _ct->kind); break;
                default: assert(0);
        }
}

// string = string-literal
//        | '__func__';

// designation = designator-list '='
void designation() {
        designtorlist();
        if (_ct->kind == ASSIGN) consume("", ASSIGN);
}

// designator-list = designator {designator}
void designtorlist() {
        designator();
        while (_ct->kind == OBR || _ct->kind == DOT) designator();
}

// designator = '[' constant-expression ']'
//            | '.' identifier
void designator() {
        if (_ct->kind == OBR) {
                consume("", OBR);
                constexpr();
                consume("", CBR);
        } else {
                assert(_ct->kind == DOT);
                consume("", DOT);
                consume("", IDENT);
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
        enum Kind ctk = _ct->kind;
        if (ctk == IDENT || ctk == CASE || ctk == DEFAULT || ctk == SEMIC || ctk == OPAR ||
            ctk == MUL) {
                if ((ctk == IDENT && _ct->next->kind == COLON) || ctk == CASE || ctk == DEFAULT)
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
        enum Kind ctk = _ct->kind;
        if (ctk == IDENT) {
                consume("", IDENT);
                consume("", COLON);
                stmt();
        } else if (ctk == CASE) {
                consume("", CASE);
                constexpr();
                consume("", COLON);
                stmt();
        } else {
                assert(ctk == DEFAULT);
                consume("", DEFAULT);
                consume("", COLON);
                stmt();
        }
}

// expression-statement = [expression] ';'
void exprstmt() {
        if (_ct->kind != SEMIC) expr();
        consume("", SEMIC);
}

// selection-statement = 'if' '(' expression ')' statement 'else' statement
//                     | 'if' '(' expression ')' statement
//                     | 'switch' '(' expression ')' statement
void selectstmt() {
        enum Kind ctk = _ct->kind;
        if (ctk == IF) {
                consume("", IF);
                consume("", OPAR);
                expr();
                consume("", CPAR);
                stmt();
                if (_ct->kind == ELSE) {
                        consume("", ELSE);
                        stmt();
                }
        } else {
                assert(ctk == SWITCH);
                consume("", SWITCH);
                consume("", OPAR);
                expr();
                consume("", CPAR);
                stmt();
        }
}

//  iteration-statement = 'while' '(' expression ')' statement
//                      | 'do' statement 'while' '(' expression ')' ';'
//                      | 'for' '(' [expression] ';' [expression] ';' [expression] ')' statement
//                      | 'for' '(' declaration [expression] ';' [expression] ')' statement
void iterstmt() {
        enum Kind ctk = _ct->kind;
        if (ctk == FOR) {
                consume("", FOR);
                consume("", OPAR);
                if (_ct->kind >= VOID && _ct->kind <= ENUM) {
                        declaration();
                } else {
                        if (_ct->kind != SEMIC) expr();
                        consume("", SEMIC);
                }
                if (_ct->kind != SEMIC) expr();
                consume("", SEMIC);
                if (_ct->kind != CPAR) expr();
                consume("", CPAR);
                stmt();
        } else if (ctk == DO) {
                consume("", DO);
                stmt();
                consume("", WHILE);
                consume("", OPAR);
                expr();
                consume("", CPAR);
                consume("", SEMIC);
        } else {
                assert(ctk == WHILE);
                consume("", WHILE);
                consume("", OPAR);
                expr();
                consume("", CPAR);
                stmt();
        }
}

// jump-statement = 'goto' identifier ';'
//                | 'continue' ';'
//                | 'break' ';'
//                | 'return' [expression] ';'
void jumpstmt() {
        enum Kind ctk = _ct->kind;
        if (ctk == RETURN) {
                consume("", ctk);
                if (_ct->kind != SEMIC) expr();
        } else if (ctk == BREAK) {
                consume("", ctk);
        } else if (ctk == CONTINUE) {
                consume("", ctk);
        } else if (ctk == GOTO) {
                consume("", ctk);
                consume("", IDENT);
        } else {
                error("invalid jump statement\n");
        }
        consume("missing ';' of jump stmt", SEMIC);
}
