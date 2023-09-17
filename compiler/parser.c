#pragma clang diagnostic ignored "-Wgnu-empty-initializer"
#pragma clang diagnostic ignored "-Wstrict-prototypes"

#include "ganymede.h"

/* global variables */
static struct Token *_ct;
enum { FUNC = 1, DECL } _cextdecl;
enum { GLOBAL = 1, LOCAL, PARAM } _cdecl;

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
void ddecltor();
void pointer();
void funcspec();
void typequal();
void typespc();
void sclass();
void declorstmt();
void compstmt();
void decltor();
void declspec();
void decl();
void funcdef();
void extdecl();
void initdecllist();
void selectstmt();
void iterstmt();
void labelstmt();
void initializer();

void parse(struct Token *token);

// translation-unit = {external-declaration};
void parse(struct Token *token) {
        _ct = token;
        while (_ct->kind != EOI) {
                extdecl();
        }
}

// external-declaration = declaration-specifiers declarator {function-definition | declaration}
void extdecl() {
        declspec();
        decltor();
        if (_cextdecl == FUNC) {
                funcdef();
        } else {
                _cdecl = GLOBAL;
                decl();
        }
}

// function-definition = compound-statement;
void funcdef() { compstmt(); }

// declaration = [declaration-specifiers] [init-declarator-list] ';'
//             | static-assert-declaration
//             | ';';
void decl() {
        if (_cdecl != GLOBAL) {
                declspec();
                decltor();
        }
        initdecllist();
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
                typespc();
                typequal();
                funcspec();
        }
}

// declarator = [pointer] direct-declarator
void decltor() {
        if (_ct->kind == MUL) pointer();
        ddecltor();
}

// declaration-list = declaration, {declaration};

// compound-statement = '{', {declaration-or-statement}, '}';
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
                _cdecl = LOCAL;
                decl();
        } else {
                stmt();
        }
}

// init-declarator-list = ['=', initializer] {',' declarator ['=' initializer]};
void initdecllist() {
        /* first declarator is already parsed. now see if it has initializer */
        if (_ct->kind == ASSIGN) {
                consume("", ASSIGN);
                initializer();
        }
        /* rest of the declarators */
        while (_ct->kind == COMMA && _ct->kind != EOI) {
                consume("", COMMA);
                decltor();
                if (_ct->kind == ASSIGN) {
                        consume("", ASSIGN);
                        initializer();
                }
        }
}

// static-assert-declaration = '_Static_assert', '(', constant-expression, ',', string-literal, ')', ';';

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
void typespc() {
        enum Kind ctk = _ct->kind;
        if (ctk >= VOID && ctk <= ENUM) consume("", ctk);
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
        if (_ct->kind == MUL) consume("", MUL);
}

// direct-declarator = identifier
//                   | '(', declarator, ')'
//                   | direct-declarator, '[', ['*'], ']'
//                   | direct-declarator, '[', 'static', [type-qualifier-list], assignment-expression, ']'
//                   | direct-declarator, '[', type-qualifier-list, ['*'], ']'
//                   | direct-declarator, '[', type-qualifier-list, ['static'], assignment-expression, ']'
//                   | direct-declarator, '[', assignment-expression, ']'
//                   | direct-declarator, '(', parameter-type-list, ')'
//                   | direct-declarator, '(', ')';
void ddecltor() {
        if (_ct->kind == IDENT) consume("", IDENT);
        if (_ct->kind == OPAR) {
                _cextdecl = FUNC; /* global var */

                // function
                consume("", OPAR);
                consume("", VOID);
                consume("missing ')' of function definition", CPAR);
                return;
        } else if (_ct->kind == OBR) {
                // array
                consume("", OBR);
                consume("", INT);
                consume("missing '[' of array declaration", CBR);
        }
        _cextdecl = DECL; /* global var */
}

// initializer-list = designative-initializer, {',', designative-initializer};

// designative-initializer = [designation], initializer;

// initializer = '{', initializer-list, [','], '}'
//             | assignment-expression;
void initializer() {
        //
        if (_ct->kind == INTCONST) consume("", INTCONST);
}

// constant-expression = conditional-expression;  (* with constraints *)

// struct-or-union-specifier = struct-or-union, '{', struct-declaration-list, '}'
//                           | struct-or-union, identifier, ['{', struct-declaration-list, '}'];

// struct-or-union = 'struct'
//                 | 'union';

// struct-declaration-list = struct-declaration, {struct-declaration};

// struct-declaration = specifier-qualifier-list, ';'     (* for anonymous struct/union *)
//                    | specifier-qualifier-list, struct-declarator-list, ';'
//                    | static-assert-declaration;

// enum-specifier = 'enum', '{', enumerator-list, [','], '}'
//                | 'enum', identifier, ['{', enumerator-list, [','], '}'];

// enumerator-list = enumerator, {',', enumerator};

// (* NOTE: Please define enumeration-constant for identifier inside enum { ... }. *)
// enumerator = enumeration-constant, ['=', constant-expression];

// enumeration-constant = identifier;

// type-name = specifier-qualifier-list, [abstract-declarator];

// specifier-qualifier-list = specifier-qualifier, {specifier-qualifier};

// specifier-qualifier = type-specifier | type-qualifier;

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

// struct-declarator-list = struct-declarator, {',', struct-declarator};

// type-qualifier-list = type-qualifier, {type-qualifier};

// parameter-type-list = parameter-list, [',', '...'];

// struct-declarator = ':', constant-expression
//                   | declarator, [':', constant-expression];

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

// designation = designator-list, '=';

// designator-list = designator, {designator};

// designator = '[', constant-expression, ']'
//            | '.', identifier;

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
