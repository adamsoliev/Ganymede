#include "ganymede.h"

/*
References
https://github.com/rui314/chibicc/blob/main/parse.c
https://github.com/sheisc/ucc162.3/blob/10658ca08af36aee9737b6087df27fda580e2c75/ucc/ucl/decl.c
https://github.com/aligrudi/neatcc/blob/3472e4b5aca9e4543f0149015babb329943197db/ncc.c
https://github.com/bobrippling/ucc-c-compiler/blob/d0283673261bbd5d872057ee169378fdf349a6fa/src/cpp2/expr.c
https://github.com/hikalium/compilium/blob/v2/parser.c
https://github.com/xorvoid/sectorc/blob/5e3fb46bdc6b1190531a68424bdc81d1c19a8dae/lint/lint.c
https://github.com/andrewchambers/c/blob/master/src/cc/parse.c
https://github.com/jserv/shecc/blob/master/src/cfront.c
https://github.com/larmel/lacc/tree/30839843daaff9d87574b5854854c9ee4610cdcd/src/parser
https://github.com/jserv/MazuCC/blob/master/parser.c
https://github.com/ptsolg/scc/blob/799afe627a76e9eccfdb7f7a076d2b36592b16fd/include/scc/syntax/parser.h
*/

// Forward declarations
static struct decl *create_decl();
static struct stmt *create_stmt();
static struct expr *create_expr();
static struct type *create_type();
static struct param_list *create_param_list();
static struct decl *function_definition(struct Token **rest,
                                        struct Token *token);
static struct stmt *declaration(struct Token **rest, struct Token *token);
static struct type *declaration_specifiers(struct Token **rest,
                                           struct Token *token);
static struct type *declaration_specifier(struct Token **rest,
                                          struct Token *token);
static struct type *declarator(struct Token **rest, struct Token *token);
static struct stmt *compound_statement(struct Token **rest,
                                       struct Token *token);
static struct stmt *declaration_or_statement(struct Token **rest,
                                             struct Token *token);
static struct expr *init_declarator_list(struct Token **rest,
                                         struct Token *token);
static struct expr *init_declarator(struct Token **rest, struct Token *token);
static struct type *type_specifier(struct Token **rest, struct Token *token);
static struct type *direct_declarator(struct Token **rest, struct Token *token);
static struct expr *initializer(struct Token **rest, struct Token *token);
static struct param_list *parameter_list(struct Token **rest,
                                         struct Token *token);
struct param_list *parameter_type_list(struct Token **rest,
                                       struct Token *token);
static struct param_list *parameter_declaration(struct Token **rest,
                                                struct Token *token);
static struct stmt *statement(struct Token **rest, struct Token *token);
static struct stmt *expression_statement(struct Token **rest,
                                         struct Token *token);
static struct stmt *selection_statement(struct Token **rest,
                                        struct Token *token);
static struct stmt *jump_statement(struct Token **rest, struct Token *token);

static struct expr *expression(struct Token **rest, struct Token *token);
static struct expr *assignment_expression(struct Token **rest,
                                          struct Token *token);
static struct expr *conditional_expression(struct Token **rest,
                                           struct Token *token);
static struct expr *logical_or_expression(struct Token **rest,
                                          struct Token *token);
static struct expr *logical_and_expression(struct Token **rest,
                                           struct Token *token);
static struct expr *inclusive_or_expression(struct Token **rest,
                                            struct Token *token);
static struct expr *exclusive_or_expression(struct Token **rest,
                                            struct Token *token);
static struct expr *and_expression(struct Token **rest, struct Token *token);
static struct expr *equality_expression(struct Token **rest,
                                        struct Token *token);
static struct expr *relational_expression(struct Token **rest,
                                          struct Token *token);
static struct expr *shift_expression(struct Token **rest, struct Token *token);
static struct expr *additive_expression(struct Token **rest,
                                        struct Token *token);
static struct expr *multiplicative_expression(struct Token **rest,
                                              struct Token *token);
static struct expr *cast_expression(struct Token **rest, struct Token *token);
static struct expr *unary_expression(struct Token **rest, struct Token *token);
static struct expr *postfix_expression(struct Token **rest,
                                       struct Token *token);
static struct expr *primary_expression(struct Token **rest,
                                       struct Token *token);
static struct expr *argument_expression_list(struct Token **rest,
                                             struct Token *token);
const char *type2str(struct type *type);
static struct stmt *iteration_statement(struct Token **rest,
                                        struct Token *token);

void print_params(struct param_list *param_list, int level);

// utils
static struct decl *create_decl(
    // char *name, struct type *type, struct expr *value,
    //                      struct stmt *code, struct decl *next
) {
    struct decl *decl = calloc(sizeof(struct decl), 1);
    // decl->name = name;
    // decl->type = type;
    // decl->value = value;
    // decl->code = code;
    // decl->next = next;
    return decl;
}

static struct stmt *create_stmt(enum stmt_t kind
                                //  struct decl *decl,
                                //  struct expr *init_expr, struct expr *expr,
                                //  struct expr *next_expr, struct stmt *body,
                                //  struct stmt *else_body, struct stmt *next
) {
    struct stmt *stmt = calloc(sizeof(struct stmt), 1);
    stmt->kind = kind;
    // stmt->decl = decl;
    // stmt->init_expr = init_expr;
    // stmt->expr = expr;
    // stmt->next_expr = next_expr;
    // stmt->body = body;
    // stmt->else_body = else_body;
    // stmt->next = next;
    return stmt;
}

static struct expr *create_expr(enum expr_t kind, struct expr *left,
                                struct expr *right, const char *name,
                                int integer_value, const char *string_literal) {
    struct expr *expr = calloc(sizeof(struct expr), 1);
    expr->kind = kind;
    expr->left = left;
    expr->right = right;
    expr->name = name;
    expr->integer_value = integer_value;
    expr->string_literal = string_literal;
    return expr;
}

static struct type *create_type(enum type_t kind, struct type *subtype,
                                struct param_list *params) {
    struct type *type = calloc(sizeof(struct type), 1);
    type->kind = kind;
    type->subtype = subtype;
    type->params = params;
    return type;
}

static struct param_list *create_param_list(char *name, struct type *type,
                                            struct param_list *next) {
    struct param_list *param_list = calloc(sizeof(struct param_list), 1);
    param_list->name = name;
    param_list->type = type;
    param_list->next = next;
    return param_list;
}

// Grammar source:
// https://github.com/katef/kgt/blob/main/examples/c99-grammar.iso-ebnf

// translation-unit = {external-declaration};

// external-declaration = function-definition
//                      | declaration;

// function-definition = declaration-specifiers, declarator, [declaration-list], compound-statement;
static struct decl *function_definition(struct Token **rest,
                                        struct Token *token) {
    struct decl *decl = create_decl();

    // declaration-specifiers
    struct type *subtype = declaration_specifiers(&token, token);
    struct type *type = create_type(TYPE_FUNCTION, subtype, NULL);
    decl->type = type;

    // declarator
    // decl->name = declarator(&token, token);
    struct type *temp = declarator(&token, token);  // BADDESIGN
    decl->name = temp->name;
    decl->type->params = temp->params;

    // FIXME: [declaration-list]

    // compound-statement;
    decl->code = compound_statement(&token, token);
    *rest = token;
    return decl;
};

// declaration = declaration-specifiers [pointer] [init-declarator-list] ';'
//             | static-assert-declaration
//             | ';';
static struct stmt *declaration(struct Token **rest, struct Token *token) {
    struct decl *decl = create_decl();

    // declaration-specifiers
    struct type *decl_specs = declaration_specifiers(&token, token);
    decl->type = decl_specs;

    // pointer
    if (token->kind == TK_PUNCT && equal(token, "*")) {
        token = token->next;
        struct type *type = create_type(TYPE_POINTER, decl_specs, NULL);
        decl->type = type;
    }

    // init-declarator-list
    struct expr *expr = init_declarator_list(&token, token);

    // array
    if (expr->array_size > 0) {
        struct type *type = create_type(TYPE_ARRAY, decl_specs, NULL);
        decl->type = type;
    }

    decl->name = calloc(sizeof(char), sizeof(expr->name));
    strcpy(decl->name, expr->name);

    decl->value = expr;

    struct stmt *stmt = create_stmt(STMT_DECL);
    stmt->decl = decl;

    *rest = skip(token, ";");
    return stmt;
};

// declaration-specifiers = declaration-specifier, {declaration-specifier};
static struct type *declaration_specifiers(struct Token **rest,
                                           struct Token *token) {
    //
    return declaration_specifier(rest, token);
};

// declaration-specifier = storage-class-specifier
//                       | type-specifier
//                       | type-qualifier
//                       | function-specifier
//                       | alignment-specifier;
static struct type *declaration_specifier(struct Token **rest,
                                          struct Token *token) {
    return type_specifier(rest, token);
};

// declarator = direct-declarator;
static struct type *declarator(struct Token **rest, struct Token *token) {
    return direct_declarator(rest, token);
};

// declaration-list = declaration, {declaration};

// compound-statement = '{', {declaration-or-statement}, '}';
static struct stmt *compound_statement(struct Token **rest,
                                       struct Token *token) {
    struct stmt *stmt = create_stmt(STMT_BLOCK);
    struct stmt head = {};
    struct stmt *cur = &head;

    token = skip(token, "{");
    while (!equal(token, "}")) {
        struct stmt *s = declaration_or_statement(&token, token);
        cur = cur->next = s;
    }
    *rest = skip(token, "}");
    stmt->body = head.next;
    return stmt;
};

// declaration-or-statement = declaration | statement;
static struct stmt *declaration_or_statement(struct Token **rest,
                                             struct Token *token) {
    // FIXME: generalize for all types
    if (equal(token, "int")) {
        struct stmt *stmt = declaration(&token, token);
        *rest = token;
        return stmt;
    }
    struct stmt *stmt = statement(&token, token);
    *rest = token;
    return stmt;
};

// init-declarator-list = init-declarator, {',', init-declarator};
static struct expr *init_declarator_list(struct Token **rest,
                                         struct Token *token) {
    //
    return init_declarator(rest, token);
};

// init-declarator = declarator, ['=', initializer];
static struct expr *init_declarator(struct Token **rest, struct Token *token) {
    //
    struct type *temp = declarator(&token, token);
    if (token->kind == TK_PUNCT && equal(token, "=")) {
        token = token->next;
        struct expr *expr = initializer(&token, token);
        expr->name = temp->name;
        if (temp->kind == TYPE_ARRAY)
            expr->array_size = temp->expr->integer_value;
        *rest = token;
        return expr;
    }
    struct expr *expr = create_expr(EXPR_NAME, NULL, NULL, NULL, 0, NULL);
    expr->name = temp->name;
    if (temp->kind == TYPE_ARRAY) expr->array_size = temp->expr->integer_value;
    *rest = token;
    return expr;
};

// static-assert-declaration = '_Static_assert', '(', constant-expression, ',', string-literal, ')', ';';

// storage-class-specifier = 'typedef'
//                         | 'extern'
//                         | 'static'
//                         | '_Thread_local'
//                         | 'auto'
//                         | 'register';

// type-specifier = 'void'
//                | 'char'
//                | 'short'
//                | 'int'
//                | 'long'
//                | 'float'
//                | 'double'
//                | 'signed'
//                | 'unsigned'
//                | '_Bool'
//                | '_Complex'
//                | '_Imaginary'       (* non-mandated extension *)
//                | atomic-type-specifier
//                | struct-or-union-specifier
//                | enum-specifier
//                | typedef-name;
static struct type *type_specifier(struct Token **rest, struct Token *token) {
    if (equal(token, "int")) {
        *rest = token->next;
        struct type *type = create_type(TYPE_INTEGER, NULL, NULL);
        return type;
    }
    error(
        true, "Unknown or unimplemented type_specifier for token: %s\n", token);
};

// (* NOTE: Please define typedef-name as result of 'typedef'. *)
// typedef-name = identifier;

// type-qualifier = 'const'
//                | 'restrict'
//                | 'volatile'
//                | '_Atomic';

// function-specifier = 'inline'
//                    | '_Noreturn';

// alignment-specifier = '_Alignas', '(', type-name, ')'
//                     | '_Alignas', '(', constant-expression, ')';

// pointer = '*', [type-qualifier-list], [pointer];
//

// direct-declarator = identifier
//                   | '(', declarator, ')'
//                   | direct-declarator, '[', ['*'], ']'
//                   | direct-declarator, '[', 'static', [type-qualifier-list], assignment-expression, ']'
//                   | direct-declarator, '[', type-qualifier-list, ['*'], ']'
//                   | direct-declarator, '[', type-qualifier-list, ['static'], assignment-expression, ']'
//                   | direct-declarator, '[', assignment-expression, ']'
//                   | direct-declarator, '(', parameter-type-list, ')'
//                   | direct-declarator, '(', identifier-list, ')'
//                   | direct-declarator, '(', ')';
// static char *direct_declarator(struct Token **rest, struct Token *token) {
static struct type *direct_declarator(struct Token **rest,
                                      struct Token *token) {
    struct type *type = NULL;
    if (token->next->kind == TK_PUNCT && equal(token->next, "(")) {
        // funcall
        type = create_type(TYPE_FUNCTION, NULL, NULL);
        type->name = calloc(sizeof(char), token->len + 1);
        memcpy(type->name, token->buffer, token->len);
        token = token->next;

        token = skip(token, "(");
        if (token->kind == TK_KEYWORD && equal(token, "int")) {
            type->params = parameter_type_list(&token, token);
        }
        token = skip(token, ")");
    } else if (token->next->kind == TK_PUNCT && equal(token->next, "[")) {
        // array
        type = create_type(TYPE_ARRAY, NULL, NULL);
        type->name = calloc(sizeof(char), token->len + 1);
        memcpy(type->name, token->buffer, token->len);
        token = token->next;

        token = skip(token, "[");
        struct expr *expr = assignment_expression(&token, token);
        type->expr = expr;
        token = skip(token, "]");
    } else {
        type = create_type(TYPE_VOID, NULL, NULL);
        type->name = calloc(sizeof(char), token->len + 1);
        memcpy(type->name, token->buffer, token->len);
        token = token->next;
    }

    *rest = token;
    return type;
};

// identifier-list = identifier, {',', identifier};

// initializer-list = designative-initializer, {',', designative-initializer};

// designative-initializer = [designation], initializer;

// initializer = '{', initializer-list, [','], '}'
//             | assignment-expression;
static struct expr *initializer(struct Token **rest, struct Token *token) {
    //
    return assignment_expression(rest, token);
};

// constant-expression = conditional-expression;  (* with constraints *)

// atomic-type-specifier = '_Atomic', '(', type-name, ')';

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
struct param_list *parameter_type_list(struct Token **rest,
                                       struct Token *token) {
    return parameter_list(rest, token);
};

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
static struct param_list *parameter_list(struct Token **rest,
                                         struct Token *token) {
    struct param_list *param_list = parameter_declaration(&token, token);
    if (token->kind == TK_PUNCT && equal(token, ",")) {
        token = token->next;
        param_list->next = parameter_list(&token, token);
    }
    *rest = token;
    return param_list;
};

// parameter-declaration = declaration-specifiers, [declarator | abstract-declarator];
static struct param_list *parameter_declaration(struct Token **rest,
                                                struct Token *token) {
    struct param_list *param_list = create_param_list(NULL, NULL, NULL);

    // declaration-specifiers
    struct type *type = declaration_specifiers(&token, token);
    param_list->type = type;

    // declarator | abstract-declarator
    if (token->kind == TK_PUNCT && equal(token, "*")) {
        token = token->next;
        struct type *type = create_type(TYPE_POINTER, type, NULL);
        param_list->type = type;
    }

    struct type *temp = declarator(&token, token);  // BADDESIGN
    param_list->name = temp->name;

    *rest = token;
    return param_list;
};

// expression = assignment-expression, {',', assignment-expression};
static struct expr *expression(struct Token **rest, struct Token *token) {
    //
    return assignment_expression(rest, token);
};

// assignment-expression = conditional-expression
//                       | unary-expression, assignment-operator, assignment-expression;
static struct expr *assignment_expression(struct Token **rest,
                                          struct Token *token) {
    //
    struct expr *expr = conditional_expression(&token, token);
    if (equal(token, "=")) {
        token = token->next;
        struct expr *right = assignment_expression(&token, token);
        expr = create_expr(EXPR_ASSIGN, expr, right, NULL, 0, NULL);
        *rest = token;
        return expr;
    }
    if (equal(token, "+=")) {
        token = token->next;
        struct expr *right = assignment_expression(&token, token);
        expr = create_expr(EXPR_ADD, expr, right, NULL, 0, NULL);
        *rest = token;
        return expr;
    }
    if (equal(token, "-=")) {
        token = token->next;
        struct expr *right = assignment_expression(&token, token);
        expr = create_expr(EXPR_SUB, expr, right, NULL, 0, NULL);
        *rest = token;
        return expr;
    }
    if (equal(token, "*=")) {
        token = token->next;
        struct expr *right = assignment_expression(&token, token);
        expr = create_expr(EXPR_MUL, expr, right, NULL, 0, NULL);
        *rest = token;
        return expr;
    }
    if (equal(token, "/=")) {
        token = token->next;
        struct expr *right = assignment_expression(&token, token);
        expr = create_expr(EXPR_DIV, expr, right, NULL, 0, NULL);
        *rest = token;
        return expr;
    }
    *rest = token;
    return expr;
};

// conditional-expression = logical-or-expression, ['?', expression, ':', conditional-expression];
static struct expr *conditional_expression(struct Token **rest,
                                           struct Token *token) {
    //
    return logical_or_expression(rest, token);
};

// logical-or-expression = logical-and-expression, {'||', logical-and-expression};
static struct expr *logical_or_expression(struct Token **rest,
                                          struct Token *token) {
    //
    return logical_and_expression(rest, token);
};

// logical-and-expression = inclusive-or-expression, {'&&', inclusive-or-expression};
static struct expr *logical_and_expression(struct Token **rest,
                                           struct Token *token) {
    //
    return inclusive_or_expression(rest, token);
};

// inclusive-or-expression = exclusive-or-expression, {'|', exclusive-or-expression};
static struct expr *inclusive_or_expression(struct Token **rest,
                                            struct Token *token) {
    //
    return exclusive_or_expression(rest, token);
};

// exclusive-or-expression = and-expression, {'^', and-expression};
static struct expr *exclusive_or_expression(struct Token **rest,
                                            struct Token *token) {
    //
    return and_expression(rest, token);
};

// and-expression = equality-expression, {'&', equality-expression};
static struct expr *and_expression(struct Token **rest, struct Token *token) {
    //
    return equality_expression(rest, token);
};

// equality-expression = relational-expression, {('==' | '!='), relational-expression};
static struct expr *equality_expression(struct Token **rest,
                                        struct Token *token) {
    //
    struct expr *expr = relational_expression(&token, token);
    if (token->kind == TK_PUNCT) {
        if (equal(token, "==")) {
            token = token->next;
            struct expr *right = relational_expression(&token, token);
            expr = create_expr(EXPR_EQ, expr, right, NULL, 0, NULL);
            *rest = token;
            return expr;
        }
        if (equal(token, "!=")) {
            token = token->next;
            struct expr *right = relational_expression(&token, token);
            expr = create_expr(EXPR_NE, expr, right, NULL, 0, NULL);
            *rest = token;
            return expr;
        }
    }
    *rest = token;
    return expr;
};

// relational-expression = shift-expression, {('<' | '>' | '<=' | '>='), shift-expression};
static struct expr *relational_expression(struct Token **rest,
                                          struct Token *token) {
    //
    struct expr *expr = shift_expression(&token, token);
    if (equal(token, "<")) {
        token = token->next;
        struct expr *right = shift_expression(&token, token);
        expr = create_expr(EXPR_LT, expr, right, NULL, 0, NULL);
        *rest = token;
        return expr;
    }
    if (equal(token, ">")) {
        token = token->next;
        struct expr *right = shift_expression(&token, token);
        expr = create_expr(EXPR_LT, right, expr, NULL, 0, NULL);
        *rest = token;
        return expr;
    }
    if (equal(token, "<=")) {
        token = token->next;
        struct expr *right = shift_expression(&token, token);
        expr = create_expr(EXPR_LE, expr, right, NULL, 0, NULL);
        *rest = token;
        return expr;
    }
    if (equal(token, ">=")) {
        token = token->next;
        struct expr *right = shift_expression(&token, token);
        expr = create_expr(EXPR_LE, right, expr, NULL, 0, NULL);
        *rest = token;
        return expr;
    }
    *rest = token;
    return expr;
};

// shift-expression = additive-expression, {('<<' | '>>'), additive-expression};
static struct expr *shift_expression(struct Token **rest, struct Token *token) {
    //
    return additive_expression(rest, token);
};

// additive-expression = multiplicative-expression, {('+' | '-'), multiplicative-expression};
static struct expr *additive_expression(struct Token **rest,
                                        struct Token *token) {
    //
    struct expr *expr = multiplicative_expression(&token, token);
    while (token->kind == TK_PUNCT) {
        if (equal(token, "+")) {
            token = token->next;
            struct expr *right = multiplicative_expression(&token, token);
            expr = create_expr(EXPR_ADD, expr, right, NULL, 0, NULL);
            continue;
        }
        if (equal(token, "-")) {
            token = token->next;
            struct expr *right = multiplicative_expression(&token, token);
            expr = create_expr(EXPR_SUB, expr, right, NULL, 0, NULL);
            continue;
        }
        *rest = token;
        return expr;
    }
};

// multiplicative-expression = cast-expression, {('*' | '/' | '%'), cast-expression};
static struct expr *multiplicative_expression(struct Token **rest,
                                              struct Token *token) {
    struct expr *expr = cast_expression(&token, token);
    while (token->kind == TK_PUNCT) {
        if (equal(token, "*")) {
            token = token->next;
            struct expr *right = cast_expression(&token, token);
            expr = create_expr(EXPR_MUL, expr, right, NULL, 0, NULL);
            continue;
        } else if (equal(token, "/")) {
            token = token->next;
            struct expr *right = cast_expression(&token, token);
            expr = create_expr(EXPR_DIV, expr, right, NULL, 0, NULL);
            continue;
        }
        *rest = token;
        return expr;
    }
};

// cast-expression = unary-expression
//                 | '(', type-name, ')', cast-expression;
static struct expr *cast_expression(struct Token **rest, struct Token *token) {
    //
    return unary_expression(rest, token);
};

// unary-expression = postfix-expression
//                  | ('++' | '--'), unary-expression
//                  | unary-operator, cast-expression
//                  | 'sizeof', unary-expression
//                  | 'sizeof', '(', type-name, ')'
//                  | '_Alignof', '(', type-name, ')';
static struct expr *unary_expression(struct Token **rest, struct Token *token) {
    //
    if (token->kind == TK_PUNCT) {
        if (equal(token, "+")) {
            token = token->next;
            struct expr *left =
                create_expr(EXPR_INTEGER_LITERAL, NULL, NULL, NULL, 0, NULL);
            struct expr *right = cast_expression(rest, token);
            return create_expr(EXPR_ADD, left, right, NULL, 0, NULL);
        }
        if (equal(token, "-")) {
            token = token->next;
            struct expr *left =
                create_expr(EXPR_INTEGER_LITERAL, NULL, NULL, NULL, 0, NULL);
            struct expr *right = cast_expression(rest, token);
            return create_expr(EXPR_SUB, left, right, NULL, 0, NULL);
        }
        if (equal(token, "~")) {
            token = token->next;
            struct expr *left = cast_expression(rest, token);
            return create_expr(EXPR_BITNOT, left, NULL, NULL, 0, NULL);
        }
        if (equal(token, "!")) {
            token = token->next;
            struct expr *left = cast_expression(rest, token);
            return create_expr(EXPR_NOT, left, NULL, NULL, 0, NULL);
        }
        if (equal(token, "*")) {
            token = token->next;
            struct expr *left = cast_expression(rest, token);
            return create_expr(EXPR_DEREF, left, NULL, NULL, 0, NULL);
        }
        if (equal(token, "&")) {
            token = token->next;
            struct expr *left = cast_expression(rest, token);
            return create_expr(EXPR_ADDR, left, NULL, NULL, 0, NULL);
        }
    }
    return postfix_expression(rest, token);
};

// postfix-expression = primary-expression
//                    | postfix-expression, '[', expression, ']'
//                    | postfix-expression, '(', [argument-expression-list], ')'
//                    | postfix-expression, ('.' | '->'), identifier
//                    | postfix-expression, ('++' | '--')
//                    | '(', type-name, ')', '{', initializer-list, [','], '}';
static struct expr *postfix_expression(struct Token **rest,
                                       struct Token *token) {
    //
    struct expr *expr = primary_expression(&token, token);
    if (token->kind == TK_PUNCT) {
        if (equal(token, "+") && equal(token->next, "+")) {
            token = token->next->next;
            struct expr *right =
                create_expr(EXPR_INTEGER_LITERAL, NULL, NULL, NULL, 1, NULL);
            expr = create_expr(EXPR_ADD, expr, right, NULL, 0, NULL);
            *rest = token;
            return expr;
        }
        if (equal(token, "-") && equal(token->next, "-")) {
            token = token->next->next;
            struct expr *right =
                create_expr(EXPR_INTEGER_LITERAL, NULL, NULL, NULL, 1, NULL);
            expr = create_expr(EXPR_SUB, expr, right, NULL, 0, NULL);
            *rest = token;
            return expr;
        }
        if (equal(token, "(")) {
            token = token->next;
            struct expr *args = NULL;
            if (!equal(token, ")"))
                args = argument_expression_list(&token, token);
            expr = create_expr(EXPR_FUNCALL, expr, args, NULL, 0, NULL);
            *rest = skip(token, ")");
            return expr;
        }
    }
    *rest = token;
    return expr;
};

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
static struct expr *primary_expression(struct Token **rest,
                                       struct Token *token) {
    if (token->kind == TK_NUM) {
        char *constant = calloc(sizeof(char), token->len + 1);
        memcpy(constant, token->buffer, token->len);
        struct expr *constant_expr = create_expr(EXPR_INTEGER_LITERAL,
                                                 NULL,
                                                 NULL,
                                                 NULL,
                                                 strtol(constant, NULL, 10),
                                                 NULL);

        *rest = token->next;
        return constant_expr;
    }
    if (token->kind == TK_IDENT) {
        char *ident = calloc(sizeof(char), token->len + 1);
        memcpy(ident, token->buffer, token->len);
        struct expr *ident_expr =
            create_expr(EXPR_NAME, NULL, NULL, ident, 0, NULL);
        *rest = token->next;
        return ident_expr;
    }
    error(true, "Unknown primary_expression for token: %s\n", token->buffer);
};

// argument-expression-list = assignment-expression, {',', assignment-expression};
static struct expr *argument_expression_list(struct Token **rest,
                                             struct Token *token) {
    //
    struct expr *expr = assignment_expression(&token, token);
    while (equal(token, ",")) {
        token = token->next;
        struct expr *right = argument_expression_list(&token, token);
        *rest = token;
        return create_expr(EXPR_ARGS, expr, right, NULL, 0, NULL);
    }
    *rest = token;
    return create_expr(EXPR_ARGS, expr, NULL, NULL, 0, NULL);
};

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
static struct stmt *statement(struct Token **rest, struct Token *token) {
    //
    if (token->kind == TK_PUNCT && equal(token, "{")) {
        return compound_statement(rest, token);
    }
    if (token->kind == TK_KEYWORD && equal(token, "if")) {
        return selection_statement(rest, token);
    }
    if (token->kind == TK_KEYWORD &&
        (equal(token, "for") || equal(token, "while"))) {
        return iteration_statement(rest, token);
    }
    if (token->kind == TK_KEYWORD && equal(token, "return")) {
        return jump_statement(rest, token);
    }
    return expression_statement(rest, token);
};

// labeled-statement = identifier, ':', statement
//                   | 'case', constant-expression, ':', statement
//                   | 'default', ':', statement;

// expression-statement = [expression], ';';
static struct stmt *expression_statement(struct Token **rest,
                                         struct Token *token) {
    struct stmt *stmt = create_stmt(STMT_EXPR);
    stmt->expr = expression(&token, token);
    *rest = skip(token, ";");
    return stmt;
}

// selection-statement = 'if', '(', expression, ')', statement, 'else', statement
//                     | 'if', '(', expression, ')', statement
//                     | 'switch', '(', expression, ')', statement;
static struct stmt *selection_statement(struct Token **rest,
                                        struct Token *token) {
    if (equal(token, "if")) {
        token = token->next;
        struct stmt *stmt = create_stmt(STMT_IF_ELSE);
        token = skip(token, "(");
        stmt->expr = expression(&token, token);
        token = skip(token, ")");
        stmt->body = statement(&token, token);
        if (equal(token, "else")) {
            token = token->next;
            stmt->else_body = statement(&token, token);
        }
        *rest = token;
        return stmt;
    }
    error(true, "Unknown selection_statement for token: %s\n", token->buffer);
};

//  iteration-statement = 'while', '(', expression, ')', statement
//                      | 'do', statement, 'while', '(', expression, ')', ';'
//                      | 'for', '(', [expression], ';', [expression], ';', [expression], ')', statement
//                      | 'for', '(', declaration, [expression], ';', [expression], ')', statement;
static struct stmt *iteration_statement(struct Token **rest,
                                        struct Token *token) {
    if (equal(token, "while")) {
        token = token->next;
        struct stmt *stmt = create_stmt(STMT_FOR);
        token = skip(token, "(");
        stmt->expr = expression(&token, token);
        token = skip(token, ")");
        stmt->body = statement(&token, token);
        *rest = token;
        return stmt;
    }
    if (equal(token, "for")) {
        token = token->next;
        struct stmt *stmt = create_stmt(STMT_FOR);
        token = skip(token, "(");
        if (equal(token, ";")) {
            token = token->next;
        } else {
            if (equal(token, "int")) {
                struct expr *expr =
                    create_expr(EXPR_DECL, NULL, NULL, NULL, 0, NULL);
                expr->decl = declaration(&token, token)->decl;  // FIXME: buggy?
                stmt->init_expr = expr;
            } else {
                stmt->init_expr = expression(&token, token);
                token = skip(token, ";");
            }
        }
        if (equal(token, ";")) {
            token = token->next;
        } else {
            stmt->expr = expression(&token, token);
            token = skip(token, ";");
        }
        if (equal(token, ")")) {
            token = token->next;
        } else {
            stmt->next_expr = expression(&token, token);
            token = skip(token, ")");
        }
        stmt->body = statement(&token, token);
        *rest = token;
        return stmt;
    }
    error(true, "Unknown iteration_statement for token: %s\n", token->buffer);
};

// jump-statement = 'goto', identifier, ';'
//                | 'continue', ';'
//                | 'break', ';'
//                | 'return', [expression], ';';
static struct stmt *jump_statement(struct Token **rest, struct Token *token) {
    if (equal(token, "return")) {
        token = token->next;
        struct stmt *stmt = create_stmt(STMT_RETURN);
        if (consume(rest, token, ";")) return stmt;
        struct expr *expr = expression(&token, token);
        stmt->expr = expr;
        *rest = skip(token, ";");
        return stmt;
    }
    error(true, "Unknown jump_statement for token: %s\n", token->buffer);
};

struct decl *parse(struct Token *token) {
    struct decl head = {};
    struct decl *cur = &head;
    while (token->kind != TK_EOF) {
        cur = cur->next = function_definition(&token, token);
    }
    return head.next;
};

void print_decl(struct decl *decl, int level) {
    if (decl == NULL) return;
    switch (decl->type->kind) {
        case TYPE_FUNCTION: {
            char *subtypeName = type2str(decl->type->subtype);
            fprintf(outfile,
                    "%*sFunctionDecl %s '%s'\n",
                    level * 2,
                    "",
                    subtypeName,
                    decl->name);
            if (decl->type->params != NULL) {
                fprintf(outfile, "%*sParams\n", (level + 1) * 2, "");
                print_params(decl->type->params, level + 2);
            }
            print_stmt(decl->code, level + 1);
            break;
        }
        case TYPE_INTEGER: {
            fprintf(outfile,
                    "%*sVarDecl %s '%s'\n",
                    level * 2,
                    "",
                    type2str(decl->type),
                    decl->name);
            print_expr(decl->value, level + 1);
            break;
        }
        case TYPE_POINTER: {
            fprintf(outfile,
                    "%*sVarDecl %s %s '%s'\n",
                    level * 2,
                    "",
                    type2str(decl->type),
                    type2str(decl->type->subtype),
                    decl->name);
            print_expr(decl->value, level + 1);
            break;
        }
        case TYPE_ARRAY: {
            fprintf(outfile,
                    "%*sVarDecl %s[%d] '%s'\n",
                    level * 2,
                    type2str(decl->type),
                    type2str(decl->type->subtype),
                    decl->value->array_size,
                    decl->name);
            print_expr(decl->value, level + 1);
            break;
        }
    }
    print_decl(decl->next, level);
}

void print_stmt(struct stmt *stmt, int level) {
    if (stmt == NULL) return;
    switch (stmt->kind) {
        case STMT_RETURN:
            fprintf(outfile, "%*sReturnStmt\n", level * 2, "");
            print_expr(stmt->expr, level + 1);
            break;
        case STMT_BLOCK:
            fprintf(outfile, "%*sBlockStmt\n", level * 2, "");
            print_stmt(stmt->body, level + 1);
            break;
        case STMT_DECL:
            fprintf(outfile, "%*sDeclStmt\n", level * 2, "");
            print_decl(stmt->decl, level + 1);
            break;
        case STMT_IF_ELSE:
            fprintf(outfile, "%*sIfElseStmt\n", level * 2, "");
            print_expr(stmt->expr, level + 1);
            print_stmt(stmt->body, level + 1);
            print_stmt(stmt->else_body, level + 1);
            break;
        case STMT_FOR:
            fprintf(outfile, "%*sForStmt\n", level * 2, "");
            print_expr(stmt->init_expr, level + 1);
            print_expr(stmt->expr, level + 1);
            print_expr(stmt->next_expr, level + 1);
            print_stmt(stmt->body, level + 1);
            break;
        case STMT_EXPR:
            fprintf(outfile, "%*sExprStmt\n", level * 2, "");
            print_expr(stmt->expr, level + 1);
            break;
    }
    print_stmt(stmt->next, level);
}

void print_expr(struct expr *expr, int level) {
    if (expr == NULL) return;
    switch (expr->kind) {
        case EXPR_INTEGER_LITERAL:
            fprintf(outfile,
                    "%*sIntegerLiteral %d\n",
                    level * 2,
                    "",
                    expr->integer_value);
            break;
        case EXPR_ADD:
            fprintf(outfile, "%*sAddExpr\n", level * 2, "");
            print_expr(expr->left, level + 1);
            print_expr(expr->right, level + 1);
            break;
        case EXPR_SUB:
            fprintf(outfile, "%*sSubExpr\n", level * 2, "");
            print_expr(expr->left, level + 1);
            print_expr(expr->right, level + 1);
            break;
        case EXPR_MUL:
            fprintf(outfile, "%*sMulExpr\n", level * 2, "");
            print_expr(expr->left, level + 1);
            print_expr(expr->right, level + 1);
            break;
        case EXPR_DIV:
            fprintf(outfile, "%*sDivExpr\n", level * 2, "");
            print_expr(expr->left, level + 1);
            print_expr(expr->right, level + 1);
            break;
        case EXPR_DECL:
            fprintf(outfile, "%*sDeclExpr\n", level * 2, "");
            print_decl(expr->decl, level + 1);
            break;
        case EXPR_EQ:
            fprintf(outfile, "%*sEqExpr\n", level * 2, "");
            print_expr(expr->left, level + 1);
            print_expr(expr->right, level + 1);
            break;
        case EXPR_NE:
            fprintf(outfile, "%*sNeExpr\n", level * 2, "");
            print_expr(expr->left, level + 1);
            print_expr(expr->right, level + 1);
            break;
        case EXPR_NAME:
            fprintf(outfile, "%*sNameExpr '%s'\n", level * 2, "", expr->name);
            break;
        case EXPR_LT:
            fprintf(outfile, "%*sLtExpr\n", level * 2, "");
            print_expr(expr->left, level + 1);
            print_expr(expr->right, level + 1);
            break;
        case EXPR_LE:
            fprintf(outfile, "%*sLeExpr\n", level * 2, "");
            print_expr(expr->left, level + 1);
            print_expr(expr->right, level + 1);
            break;
        case EXPR_ASSIGN:
            fprintf(outfile, "%*sAssignExpr\n", level * 2, "");
            print_expr(expr->left, level + 1);
            print_expr(expr->right, level + 1);
            break;
        case EXPR_NOT:
            fprintf(outfile, "%*sNotExpr\n", level * 2, "");
            print_expr(expr->left, level + 1);
            break;
        case EXPR_BITNOT:
            fprintf(outfile, "%*sBitNotExpr\n", level * 2, "");
            print_expr(expr->left, level + 1);
            break;
        case EXPR_DEREF:
            fprintf(outfile, "%*sDerefExpr\n", level * 2, "");
            print_expr(expr->left, level + 1);
            break;
        case EXPR_ADDR:
            fprintf(outfile, "%*sAddrExpr\n", level * 2, "");
            print_expr(expr->left, level + 1);
            break;
        case EXPR_FUNCALL:
            fprintf(outfile, "%*sFuncallExpr\n", level * 2, "");
            print_expr(expr->left, level + 1);
            print_expr(expr->right, level + 1);
            break;
        case EXPR_ARGS:
            fprintf(outfile, "%*sArgsExpr\n", level * 2, "");
            print_expr(expr->left, level + 1);
            print_expr(expr->right, level + 1);
            break;
    }
}

void print_params(struct param_list *param_list, int level) {
    if (param_list == NULL) return;
    fprintf(outfile,
            "%*sParam %s '%s'\n",
            level * 2,
            "",
            type2str(param_list->type),
            param_list->name);
    print_params(param_list->next, level);
}

const char *type2str(struct type *type) {
    switch (type->kind) {
        case TYPE_VOID: return "void";
        case TYPE_BOOLEAN: return "bool";
        case TYPE_CHARACTER: return "char";
        case TYPE_INTEGER: return "int";
        case TYPE_STRING: return "string";
        case TYPE_ARRAY: return "";
        case TYPE_FUNCTION: return "function";
        case TYPE_POINTER: return "pointer";
    }
    return "unknown";
}