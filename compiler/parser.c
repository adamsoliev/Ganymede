#pragma clang diagnostic ignored "-Wgnu-empty-initializer"
#pragma clang diagnostic ignored "-Wstrict-prototypes"

#include "ganymede.h"

/* global variables */
static struct Token *_ct;
/* if GLOBAL, first declarator is parsed upfront to diff funcdef */
enum { GLOBAL = 1, LOCAL, PARAM } _cdecllevel;

/* utility functions */
static void consume(const char *msg, enum Kind kind) {
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

void parse(struct Token *token);

// translation-unit = {external-declaration};
void parse(struct Token *token) {
        _ct = token;
        while (_ct->kind != EOI) {
                extdecl();
        }
}

// external-declaration = {function-definition | declaration}
void extdecl() {
        _cdecllevel = GLOBAL;
        declaration();
}

// function-definition = compound-statement;
void funcdef() { compstmt(); }

// declaration = declaration-specifiers declarator [declaration-specifiers] [init-declarator-list] ';'
//             | ';';
void declaration() {
        /* parse 1st declarator to diff function definition */
        declspec();
        declarator();
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

// declaration-specifiers = declaration-specifier {declaration-specifier};
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

// declarator = [pointer] direct-declarator
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
        if (_ct->kind >= VOID && _ct->kind <= ENUM) {
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
//                         | 'register';
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
//                | typedef-name;
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

// pointer = '*', [type-qualifier-list], [pointer];
void pointer() {
        while (_ct->kind == MUL && _ct->kind != EOI) consume("", MUL);
}

// direct-declarator = identifier
//                   | '(' declarator ')'
//                   | direct-declarator '[' ['*'] ']'
//                   | direct-declarator '[' 'static' [type-qualifier-list] assignment-expression ']'
//                   | direct-declarator '[' type-qualifier-list ['*'] ']'
//                   | direct-declarator '[' type-qualifier-list ['static'] assignment-expression ']'
//                   | direct-declarator '[' assignment-expression ']'
//                   | direct-declarator '(' parameter-type-list ')'
//                   | direct-declarator '(' ')'
void directdeclarator() {
        if (_ct->kind == IDENT) consume("", IDENT);
        if (_ct->kind == OPAR) {
                // function
                consume("", OPAR);
                if (_ct->kind == VOID) consume("", VOID);
                consume("missing ')' of function definition/declaration", CPAR);
                return;
        } else if (_ct->kind == OBR) {
                // array
                while (_ct->kind == OBR && _ct->kind != EOI) {
                        consume("", OBR);
                        if (_ct->kind == INTCONST) consume("", INTCONST);
                        consume("missing '[' of array declaration", CBR);
                }
        }
}

// initializer-list = designative-initializer {',' designative-initializer}
void initializerlist() {
        //
        designinitzer();
        while (_ct->kind == COMMA) {
                consume("", COMMA);
                designinitzer();
        }
}

// designative-initializer = [designation], initializer;
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
                return;
        }
        if (_ct->kind == SUB) consume("", SUB); /* negative intconst */
        consume("", INTCONST);
}

// constant-expression = conditional-expression;  (* with constraints *)

// struct-or-union-specifier = struct-or-union '{' struct-declaration-list '}'
//                           | struct-or-union identifier ['{' struct-declaration-list '}'];
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
//                 | 'union';

// struct-declaration-list = struct-declaration {struct-declaration};
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
//                | 'enum' identifier ['{' enumerator-list [','] '}'];
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

// enumerator-list = enumerator {',' enumerator};
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
                consume("", INTCONST);
        }
}

// enumeration-constant = identifier;

// type-name = specifier-qualifier-list, [abstract-declarator];

// specifier-qualifier-list = specifier-qualifier {specifier-qualifier}
void specquallist() {
        //
        while (_ct->kind >= CONST && _ct->kind <= ENUM && _ct->kind != INLINE && _ct->kind != EOI) {
                specqual();
        }
}

// specifier-qualifier = type-specifier | type-qualifier;
void specqual() {
        if (_ct->kind >= VOID && _ct->kind <= ENUM) {
                typespec();
        }
        if (_ct->kind >= CONST && _ct->kind == VOLATILE) {
                typequal();
        }
}

// abstract-declarator = pointer, [direct-abstract-declarator]
//                     | direct-abstract-declarator;

// direct-abstract-declarator = '(', abstract-declarator, ')'
//                            | '(', parameter-type-list, ')'
//                            | '(', ')'
//                            | '[', ['*'], ']'
//                            | '[', 'static', [type-qualifier-list], assignment-expression, ']'
//                            | '[', type-qualifier-list, [['static'], assignment-expression], ']'
//                            | '[', assignment-expression, ']'
//                            | direct-abstract-declarator, '[', ['*'], ']'
//                            | direct-abstract-declarator, '[', 'static', [type-qualifier-list], assignment-expression, ']'
//                            | direct-abstract-declarator, '[', type-qualifier-list, [['static'], assignment-expression], ']'
//                            | direct-abstract-declarator, '[', assignment-expression, ']'
//                            | direct-abstract-declarator, '(', parameter-type-list, ')'
//                            | direct-abstract-declarator, '(', ')';

// struct-declarator-list = struct-declarator {',' struct-declarator}
void structdeclaratorlist() {
        structdeclarator();
        while (_ct->kind == COMMA && _ct->kind != EOI) {
                consume("", COMMA);
                structdeclarator();
        }
}

// type-qualifier-list = type-qualifier, {type-qualifier};

// parameter-type-list = parameter-list, [',', '...'];

// struct-declarator = ':' constant-expression
//                   | declarator [':' constant-expression]
void structdeclarator() {
        //
        if (_ct->kind == COLON) {
                consume("", COLON);
                consume("", INTCONST);
                return;
        }
        declarator();
        if (_ct->kind == COLON) {
                consume("", COLON);
                consume("", INTCONST);
        }
}

// assignment-operator = '='
//                     | '*='
//                     | '/='
//                     | '%='
//                     | '+='
//                     | '-='
//                     | '<<='
//                     | '>>='
//                     | '&='
//                     | '^='
//                     | '|=';

// parameter-list = parameter-declaration, {',', parameter-declaration};

// parameter-declaration = declaration-specifiers, [declarator | abstract-declarator];

// expression = assignment-expression, {',', assignment-expression};

// assignment-expression = conditional-expression
//                       | unary-expression, assignment-operator, assignment-expression;

// conditional-expression = logical-or-expression, ['?', expression, ':', conditional-expression];

// logical-or-expression = logical-and-expression, {'||', logical-and-expression};

// logical-and-expression = inclusive-or-expression, {'&&', inclusive-or-expression};

// inclusive-or-expression = exclusive-or-expression, {'|', exclusive-or-expression};

// exclusive-or-expression = and-expression, {'^', and-expression};

// and-expression = equality-expression, {'&', equality-expression};

// equality-expression = relational-expression, {('==' | '!='), relational-expression};

// relational-expression = shift-expression, {('<' | '>' | '<=' | '>='), shift-expression};

// shift-expression = additive-expression, {('<<' | '>>'), additive-expression};

// additive-expression = multiplicative-expression, {('+' | '-'), multiplicative-expression};

// multiplicative-expression = cast-expression, {('*' | '/' | '%'), cast-expression};

// cast-expression = unary-expression
//                 | '(', type-name, ')', cast-expression;

// unary-expression = postfix-expression
//                  | ('++' | '--'), unary-expression
//                  | unary-operator, cast-expression
//                  | 'sizeof', unary-expression
//                  | 'sizeof', '(', type-name, ')'
//                  | '_Alignof', '(', type-name, ')';

// postfix-expression = primary-expression
//                    | postfix-expression, '[', expression, ']'
//                    | postfix-expression, '(', [argument-expression-list], ')'
//                    | postfix-expression, ('.' | '->'), identifier
//                    | postfix-expression, ('++' | '--')
//                    | '(', type-name, ')', '{', initializer-list, [','], '}';

// unary-operator = '&'
//                | '*'
//                | '+'
//                | '-'
//                | '~'
//                | '!';

// primary-expression = identifier
//                    | constant
//                    | string
//                    | '(', expression, ')'
//                    | generic-selection;

// argument-expression-list = assignment-expression, {',', assignment-expression};

// constant = integer-constant
//          | character-constant
//          | floating-constant
//          | enumeration-constant;

// string = string-literal
//        | '__func__';

// generic-selection = '_Generic', '(', assignment-expression, ',', generic-assoc-list, ')';

// generic-assoc-list = generic-association, {',', generic-association};

// generic-association = type-name, ':', assignment-expression
//                     | 'default', ':', assignment-expression;

// designation = designator-list '=';
void designation() {
        designtorlist();
        if (_ct->kind == ASSIGN) consume("", ASSIGN);
}

// designator-list = designator {designator};
void designtorlist() {
        //
        designator();
}

// designator = '[' constant-expression ']'
//            | '.' identifier
void designator() {
        if (_ct->kind == OBR) {
                consume("", OBR);
                consume("", INTCONST);
                consume("", CBR);
        } else {
                assert(_ct->kind == DOT);
                consume("", IDENT);
        }
}

// statement = labeled-statement
//           | compound-statement
//           | expression-statement
//           | selection-statement
//           | iteration-statement
//           | jump-statement;
void stmt() {
        enum Kind ctk = _ct->kind;
        if (ctk == IDENT) {
                labelstmt();
        } else if (ctk >= GOTO && ctk <= RETURN) {
                jumpstmt();
        } else if (ctk == FOR) {
                iterstmt();
        } else if (ctk == IF) {
                selectstmt();
        } else {
                compstmt();
        }
}

// labeled-statement = identifier, ':', statement
//                   | 'case', constant-expression, ':', statement
//                   | 'default', ':', statement;
void labelstmt() {
        enum Kind ctk = _ct->kind;
        if (ctk == IDENT) {
                consume("", IDENT);
                consume("", COLON);
                stmt();
        }
}

// expression-statement = [expression], ';';

// selection-statement = 'if', '(', expression, ')', statement, 'else', statement
//                     | 'if', '(', expression, ')', statement
//                     | 'switch', '(', expression, ')', statement;
void selectstmt() {
        enum Kind ctk = _ct->kind;
        if (ctk == IF) {
                consume("", IF);
                consume("", OPAR);
                consume("", IDENT);
                consume("", EQ);
                consume("", INTCONST);
                consume("", CPAR);
                stmt();
        }
}

//  iteration-statement = 'while', '(', expression, ')', statement
//                      | 'do', statement, 'while', '(', expression, ')', ';'
//                      | 'for', '(', [expression], ';', [expression], ';', [expression], ')', statement
//                      | 'for', '(', declaration, [expression], ';', [expression], ')', statement;
void iterstmt() {
        enum Kind ctk = _ct->kind;
        if (ctk == FOR) {
                consume("", FOR);
                consume("", OPAR);
                consume("", IDENT);
                consume("", ASSIGN);
                consume("", INTCONST);
                consume("", SEMIC);
                consume("", IDENT);
                consume("", LT);
                consume("", INTCONST);
                consume("", SEMIC);
                consume("", IDENT);
                consume("", INCR);
                consume("", CPAR);
                stmt();
        }
}

// jump-statement = 'goto' identifier ';'
//                | 'continue' ';'
//                | 'break' ';'
//                | 'return' [expression] ';';
void jumpstmt() {
        enum Kind ctk = _ct->kind;
        if (ctk == RETURN) {
                consume("", ctk);
                consume("", INTCONST);
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
