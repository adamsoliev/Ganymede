#include "ganymede.h"

static struct Token *ct;
static int INDENT = 4;

// either function or declaration
struct ExtDecl {
        struct ExtDecl *next;
        struct declspec *declspec;
        struct decltor *decltor;
        struct expr *expr;         // for declaration
        struct initializer *init;  // for array declaration
        struct stmt *compStmt;     // for function
};

struct declspec {
        enum Kind type;
};

struct decltor {
        //
        char *name;
        enum {
                FUNCTION,
                DECLARATION,
        } kind;
        struct params *params;  // function
        int row;                // array
        int col;                // array
};

struct params {
        struct params *next;
        struct declspec *declspec;
        struct decltor *decltor;
};

struct initializer {
        struct initializer *next;
        struct initializer *children;
        union {
                int ivalue;
        } value;
        enum Kind type;
};

struct expr {
        enum Kind kind;
        int value;
        char *strLit;

        struct expr *lhs;
        struct expr *rhs;
};

struct expr *new_expr(enum Kind kind, struct expr *lhs, struct expr *rhs) {
        struct expr *expr = calloc(1, sizeof(struct expr));
        expr->kind = kind;
        expr->lhs = lhs;
        expr->rhs = rhs;
        return expr;
}

// statement or declaration
struct block {
        struct block *next;
        struct ExtDecl *decl;
        struct stmt *stmt;
};

/*
label_stmt
    ident ':' stmt                              | value then
    'case' expr ':' stmt                        | cond then
    'default' ':' stmt                          | then

compound_stmt
    '{' block '}'                               | body

expression_stmt
    expr ';'                                    | value 

selection_stmt
    'if' '(' expr ')' stmt                      | cond then
    'if' '(' expr ')' stmt 'else' stmt          | cond then els
    'switch' '(' expr ')' stmt                  | cond then

iteration_stmt
    'while' '(' expr ')' stmt                   | cond then
    'do' stmt 'while' '(' expr ')' ';'          | then cond
    'for' '(' expr ';' expr ';' expr ')' stmt   | init cond inc then
    'for' '(' decl expr ';' expr ')' stmt       | init cond inc then

jump_stmt
    'goto' ident ';'                            | value
    'continue' ';'                              | 
    'break' ';'                                 | 
    'return' expr ';'                           | value
*/

struct stmt {
        enum Kind kind;
        struct expr *cond;
        struct stmt *then;
        struct stmt *els;
        union {
                struct expr *expr;
                struct ExtDecl *decl;
        } init;
        int init_kind;  // 0 for expr, 1 for decl
        struct expr *inc;
        struct expr *value;
        struct block *body;
};

void copystr(char **dest, char **src, int len);
void consume(enum Kind kind);
struct ExtDecl *function(struct declspec **declspec, struct decltor **decltor);
struct ExtDecl *declaration(struct declspec **declspec, struct decltor **decltor);
struct expr *expr();
struct declspec *declaration_specifiers();
struct decltor *declarator();
struct params *parameters();
void printBlock(struct block *block, int level);
void printStmt(struct stmt *stmt, int level);
void printExpr(struct expr *expr, int level);
void printParams(struct params *params, int level);
void printInitializer(struct initializer *initializer, int level);
struct expr *primary_expression();
struct expr *additive_expression();
struct expr *multiplicative_expression();
struct expr *shift_expression();
struct expr *relational_expression();
struct expr *equality_expression();
struct expr *equality_expression();
struct expr *and_expression();
struct expr *exc_or_expression();
struct expr *inc_or_expression();
struct expr *logic_and_expression();
struct expr *logic_or_expression();
struct expr *conditional_expression();
struct expr *unary_expression();
struct expr *assignment_expression();
struct expr *postfix_expression();
struct expr *arg_expr_list();
struct stmt *stmt();
struct stmt *compound_stmt();
struct initializer *initializer_list();

void consume(enum Kind kind) {
        if (ct->kind != kind) {
                error("Expected %s, got %s", token_names[kind], token_names[ct->kind]);
        }
        ct = ct->next;
}

// function-definition ::=
//      declarator compount-statement? ;
struct ExtDecl *function(struct declspec **declspec, struct decltor **decltor) {
        struct ExtDecl *func = calloc(1, sizeof(struct ExtDecl));
        func->declspec = *declspec;
        func->decltor = *decltor;
        func->compStmt = compound_stmt();
        return func;
};

struct stmt *stmt() {
        struct stmt *statement = calloc(1, sizeof(struct stmt));
        switch (ct->kind) {
                case IDENT:
                        if (ct->next->kind == COLON) {
                                statement->kind = IDENT;  // label statement
                                statement->value = primary_expression();
                                consume(COLON);
                                statement->then = stmt();
                                break;
                        }
                        goto stmt_expr;
                case CASE: {
                        consume(CASE);
                        statement->kind = CASE;
                        statement->cond = conditional_expression();  // constant-expression
                        consume(COLON);
                        statement->then = stmt();
                        break;
                }
                case DEFAULT: {
                        consume(DEFAULT);
                        statement->kind = DEFAULT;
                        consume(COLON);
                        statement->then = stmt();
                        break;
                }
                case IF:
                        statement->kind = IF;
                        consume(IF);
                        consume(OPAR);
                        statement->cond = expr();
                        consume(CPAR);
                        statement->then = stmt();
                        if (ct->kind == ELSE) {
                                consume(ELSE);
                                statement->els = stmt();
                        }
                        break;
                case SWITCH:
                        statement->kind = SWITCH;
                        consume(SWITCH);
                        consume(OPAR);
                        statement->cond = expr();
                        consume(CPAR);
                        statement->then = stmt();
                        break;
                case WHILE:
                        statement->kind = WHILE;
                        consume(WHILE);
                        consume(OPAR);
                        statement->cond = expr();
                        consume(CPAR);
                        statement->then = stmt();
                        break;
                case DO:
                        statement->kind = DO;
                        consume(DO);
                        statement->then = stmt();
                        consume(WHILE);
                        consume(OPAR);
                        statement->cond = expr();
                        consume(CPAR);
                        consume(SEMIC);
                        break;
                case FOR:
                        statement->kind = FOR;
                        consume(FOR);
                        consume(OPAR);
                        // init
                        if (ct->kind == INT) {
                                struct declspec *declspec = declaration_specifiers();
                                struct decltor *decltor = declarator();
                                statement->init.decl = calloc(1, sizeof(struct ExtDecl));
                                statement->init.decl = declaration(&declspec, &decltor);
                                statement->init_kind = 1;
                        } else {
                                statement->init.expr = expr();
                                statement->init_kind = 0;
                                consume(SEMIC);
                        }
                        // cond
                        statement->cond = expr();
                        consume(SEMIC);
                        // inc
                        statement->inc = expr();
                        consume(CPAR);
                        // stmt
                        statement->then = stmt();
                        break;
                case GOTO: {
                        statement->kind = GOTO;
                        consume(GOTO);
                        statement->value = primary_expression();  // identifier
                        consume(SEMIC);
                        break;
                }
                case CONTINUE: {
                        statement->kind = CONTINUE;
                        consume(CONTINUE);
                        consume(SEMIC);
                        break;
                }
                case BREAK: {
                        statement->kind = BREAK;
                        consume(BREAK);
                        consume(SEMIC);
                        break;
                }
                case RETURN:
                        statement->kind = RETURN;
                        consume(RETURN);
                        if (ct->kind != SEMIC) {
                                statement->value = expr();
                        }
                        consume(SEMIC);
                        break;
                case OCBR:  // compound statement
                        return compound_stmt();
                default:
                stmt_expr : {
                        // expression-statement
                        statement->kind = STMT_EXPR;
                        statement->value = expr();
                        consume(SEMIC);
                }
        }
        return statement;
}

struct stmt *compound_stmt() {
        struct stmt *comp_stmt = calloc(1, sizeof(struct stmt));
        comp_stmt->kind = STMT_COMPOUND;
        struct block head = {};
        struct block *cur = &head;
        if (ct->kind == OCBR) {
                consume(OCBR);
                while (ct->kind != CCBR) {
                        // declaration or statement
                        cur = cur->next = calloc(1, sizeof(struct block));
                        if (ct->kind == INT) {
                                struct declspec *declspec = declaration_specifiers();
                                struct decltor *decltor = declarator();
                                cur->decl = calloc(1, sizeof(struct ExtDecl));
                                cur->decl = declaration(&declspec, &decltor);
                        } else {
                                cur->stmt = stmt();
                        }
                }
                consume(CCBR);
        }
        comp_stmt->body = head.next;
        return comp_stmt;
}

// declaration ::=
// 	    declspec decltor ("=" expr)? ("," decltor ("=" expr)?)* ";"
struct ExtDecl *declaration(struct declspec **declspec, struct decltor **decltor) {
        struct ExtDecl head = {};
        struct ExtDecl *cur = &head;
        cur = cur->next = calloc(1, sizeof(struct ExtDecl));
        cur->declspec = *declspec;
        cur->decltor = *decltor;
        while (ct->kind != SEMIC) {
                if (ct->kind == ASSIGN) {
                        consume(ASSIGN);
                        if (ct->kind == OCBR) {
                                cur->init = calloc(1, sizeof(struct initializer));
                                cur->init->type = (*declspec)->type;
                                cur->init->children = calloc(1, sizeof(struct initializer));
                                consume(OCBR);
                                cur->init->children = initializer_list();
                                consume(CCBR);
                        } else {
                                cur->expr = expr();  // initializer
                        }
                }
                if (ct->kind == COMMA) {
                        consume(COMMA);
                        cur = cur->next = calloc(1, sizeof(struct ExtDecl));
                        cur->declspec = *declspec;
                        cur->decltor = declarator();
                }
        }
        consume(SEMIC);
        return head.next;
};

struct initializer *initializer_list() {
        struct initializer head = {};
        struct initializer *cur = &head;
        while (ct->kind != CCBR) {
                if (ct->kind == OCBR) {
                        consume(OCBR);
                        cur = cur->next = calloc(1, sizeof(struct initializer));
                        cur->type = INT;
                        cur->children = initializer_list();
                } else if (ct->kind == INTCONST) {
                        cur = cur->next = calloc(1, sizeof(struct initializer));
                        cur->type = INT;
                        cur->value.ivalue = ct->value;
                        consume(INTCONST);
                } else {
                        error("Initializer list not implemented\n");
                }
                if (ct->kind == COMMA) {
                        consume(COMMA);
                }
        }
        return head.next;
};

struct declspec *declaration_specifiers() {
        struct declspec *declspec = calloc(1, sizeof(struct declspec));
        if (ct->kind == INT) {
                consume(INT);
                declspec->type = INT;
                return declspec;
        }
        if (ct->kind == FLOAT) {
                consume(FLOAT);
                declspec->type = FLOAT;
                return declspec;
        }
        return declspec;
};

// declarator ::=
// 	    pointer? (identifier or "(" declarator ")")
struct decltor *declarator() {
        struct decltor *decltor = calloc(1, sizeof(struct decltor));
        if (ct->kind == IDENT) {
                copystr(&decltor->name, &ct->start, ct->len);
                consume(IDENT);
                if (ct->kind == OPAR) {
                        decltor->kind = FUNCTION;
                        consume(OPAR);
                        decltor->params = parameters();
                        consume(CPAR);
                        return decltor;
                } else if (ct->kind == OBR) {
                        decltor->kind = DECLARATION;
                        consume(OBR);
                        if (ct->kind == INTCONST) {
                                decltor->row = ct->value;
                                consume(INTCONST);
                        }
                        if (ct->kind == CBR && ct->next->kind == OBR &&
                            ct->next->next->kind == INTCONST) {
                                consume(CBR);
                                consume(OBR);
                                decltor->col = ct->value;
                                consume(INTCONST);
                        }
                        consume(CBR);
                        return decltor;
                } else {
                        decltor->kind = DECLARATION;
                        return decltor;
                }
        }
        return decltor;
};

struct params *parameters() {
        if (ct->kind == CPAR) return NULL;
        struct params head = {};
        struct params *cur = &head;
        int maxParams = 10;
        while (1) {
                if (maxParams == 0) {
                        error("Too many parameters\n");
                }

                cur = cur->next = calloc(1, sizeof(struct params));
                cur->declspec = declaration_specifiers();
                cur->decltor = declarator();
                if (ct->kind != COMMA) {
                        break;
                }
                consume(COMMA);

                maxParams--;
        }
        return head.next;
};

struct expr *expr() { return assignment_expression(); }

#define HANDLE_BINOP(opEnum, func)                  \
        if (ct->kind == opEnum) {                   \
                consume(opEnum);                    \
                struct expr *rhs = func;            \
                return new_expr(opEnum, expr, rhs); \
        }

struct expr *assignment_expression() {
#define HANDLE_OPASSIGN(opAssign, op)                                                     \
        if (ct->kind == opAssign) {                                                       \
                consume(opAssign);                                                        \
                struct expr *assign_expr = assignment_expression();                       \
                return new_expr(ASSIGN, cond_expr, new_expr(op, cond_expr, assign_expr)); \
        }

        struct expr *cond_expr = conditional_expression();
        if (ct->kind == ASSIGN) {
                consume(ASSIGN);
                struct expr *assign_expr = assignment_expression();
                return new_expr(ASSIGN, cond_expr, assign_expr);
        }
        HANDLE_OPASSIGN(MULASSIGN, MUL);
        HANDLE_OPASSIGN(DIVASSIGN, DIV);
        HANDLE_OPASSIGN(MODASSIGN, MOD);
        HANDLE_OPASSIGN(ADDASSIGN, ADD);
        HANDLE_OPASSIGN(SUBASSIGN, SUB);
        HANDLE_OPASSIGN(LSHIFTASSIGN, LSHIFT);
        HANDLE_OPASSIGN(RSHIFTASSIGN, RSHIFT);
        HANDLE_OPASSIGN(ANDASSIGN, AND);
        HANDLE_OPASSIGN(XORASSIGN, XOR);
        HANDLE_OPASSIGN(ORASSIGN, OR);
        return cond_expr;
}

struct expr *conditional_expression() {
        struct expr *cond_expr = logic_or_expression();
        if (ct->kind == QMARK) {
                consume(QMARK);
                struct expr *true_expr = expr();
                consume(COLON);
                struct expr *false_expr = conditional_expression();
                return new_expr(QMARK, cond_expr, new_expr(COLON, true_expr, false_expr));
        }
        return cond_expr;
}

struct expr *logic_or_expression() {
        struct expr *expr = logic_and_expression();
        HANDLE_BINOP(OROR, logic_or_expression());
        return expr;
}

struct expr *logic_and_expression() {
        struct expr *expr = inc_or_expression();
        HANDLE_BINOP(ANDAND, logic_and_expression());
        return expr;
}

struct expr *inc_or_expression() {
        struct expr *expr = exc_or_expression();
        HANDLE_BINOP(OR, inc_or_expression());
        return expr;
}

struct expr *exc_or_expression() {
        struct expr *expr = and_expression();
        HANDLE_BINOP(XOR, exc_or_expression());
        return expr;
}

struct expr *and_expression() {
        struct expr *expr = equality_expression();
        HANDLE_BINOP(AND, and_expression());
        return expr;
}

struct expr *equality_expression() {
        struct expr *expr = relational_expression();
        HANDLE_BINOP(EQ, equality_expression());
        HANDLE_BINOP(NEQ, equality_expression());
        return expr;
}

struct expr *relational_expression() {
        struct expr *expr = shift_expression();
        HANDLE_BINOP(LT, relational_expression());
        HANDLE_BINOP(GT, relational_expression());
        HANDLE_BINOP(LEQ, relational_expression());
        HANDLE_BINOP(GEQ, relational_expression());
        return expr;
}

struct expr *shift_expression() {
        struct expr *expr = additive_expression();
        HANDLE_BINOP(LSHIFT, shift_expression());
        HANDLE_BINOP(RSHIFT, shift_expression());
        return expr;
}

struct expr *additive_expression() {
        struct expr *expr = multiplicative_expression();
        HANDLE_BINOP(ADD, additive_expression());
        HANDLE_BINOP(SUB, additive_expression());
        return expr;
};

struct expr *multiplicative_expression() {
        struct expr *expr = unary_expression();
        HANDLE_BINOP(MUL, multiplicative_expression());
        HANDLE_BINOP(DIV, multiplicative_expression());
        return expr;
};

struct expr *unary_expression() {
        if (ct->kind == INCR) {
                consume(INCR);
                return new_expr(INCR, unary_expression(), NULL);
        }
        if (ct->kind == DECR) {
                consume(DECR);
                return new_expr(DECR, unary_expression(), NULL);
        }
        if (ct->kind == AND || ct->kind == MUL || ct->kind == ADD || ct->kind == SUB ||
            ct->kind == TILDA || ct->kind == NOT) {
                enum Kind kind = ct->kind;
                consume(kind);
                return new_expr(kind, unary_expression(), NULL);
        }
        if (ct->kind == SIZEOF) {
                if (ct->next->kind == OPAR) {
                        error("sizeof (type-name) not implemented\n");
                } else {
                        consume(SIZEOF);
                        struct expr *expr = unary_expression();
                        return new_expr(SIZEOF, expr, NULL);
                }
        }
        return postfix_expression();
}

// postfix-expression ::=
// 	    primary-expression
// 	    postfix-expression "[" expression "]"                               -- array
// 	    postfix-expression "(" argument-expression-list? ")"                -- function call
// 	    postfix-expression "." identifier                                   -- struct
// 	    postfix-expression "->" identifier                                  -- struct pointer
// 	    postfix-expression "++"                                             -- increment
// 	    postfix-expression "--"                                             -- decrement
// 	    "(" type-name ")" "{" initializer-list "}"                          -- compound literal
// 	    "(" type-name ")" "{" initializer-list "," "}"                      -- compound literal
struct expr *postfix_expression() {
        if (ct->kind == OPAR) {
                error("postfix_expression not implemented\n");
        }
        struct expr *prim_expr = primary_expression();
        if (ct->kind == OBR) {
                // array access
                consume(OBR);
                struct expr *index = expr();
                consume(CBR);
                return new_expr(OBR, prim_expr, index);
        } else if (ct->kind == OPAR) {
                // func call
                consume(OPAR);
                struct expr *arg_list = arg_expr_list();
                // arguments
                consume(CPAR);
                return new_expr(OPAR, prim_expr, arg_list);
        } else if (ct->kind == DOT) {
                // struct access
                consume(DOT);
                struct expr *field = primary_expression();
                return new_expr(DOT, prim_expr, field);
        } else if (ct->kind == DEREF) {
                consume(DEREF);
                struct expr *field = primary_expression();
                return new_expr(DEREF, prim_expr, field);
        } else if (ct->kind == INCR) {
                consume(INCR);
                return new_expr(INCR, prim_expr, NULL);
        } else if (ct->kind == DECR) {
                consume(DECR);
                return new_expr(DECR, prim_expr, NULL);
        }
        return prim_expr;
}

struct expr *primary_expression() {
        struct expr *expr = calloc(1, sizeof(struct expr));
        if (ct->kind == IDENT) {
                expr->kind = IDENT;
                copystr(&expr->strLit, &ct->start, ct->len);
                consume(IDENT);
                return expr;
        }
        if (ct->kind == INTCONST) {
                expr->kind = INT;
                expr->value = ct->value;
                consume(INTCONST);
                return expr;
        }
        return expr;
};

// argument-expression-list ::=
// 	    assignment-expression
// 	    argument-expression-list "," assignment-expression
struct expr *arg_expr_list() {
        if (ct->kind != CPAR) {
                struct expr *arg_list = expr();
                if (ct->kind == COMMA) {
                        consume(COMMA);
                        struct expr *next = arg_expr_list();
                        return new_expr(OPAR, arg_list, next);
                }
                return new_expr(OPAR, arg_list, NULL);
        }
        return NULL;
}

// direct-declarator ::=
// 	    "[" type-qualifier-list? assignment-expression? "]"
// 	    "[" "static" type-qualifier-list? assignment-expression "]"
// 	    "[" type-qualifier-list "static" assignment-expression "]"
// 	    "[" type-qualifier-list? "*" "]"
// 	    "(" parameter-type-list ")"
// 	    "(" identifier-list? ")"

// function-definition
// declaration
struct ExtDecl *parse(struct Token *tokens) {
        ct = tokens;
        struct ExtDecl head = {};
        struct ExtDecl *cur = &head;
        while (ct->kind != EOI) {
                struct declspec *declspec = declaration_specifiers();
                struct decltor *decltor = declarator();  // #1 declarator
                if (decltor->kind == FUNCTION) {
                        cur = cur->next = function(&declspec, &decltor);
                } else {
                        cur = cur->next = declaration(&declspec, &decltor);
                }
        }
        return head.next;
};

// UTILS
void copystr(char **dest, char **src, int len) {
        if (*dest == NULL) {
                *dest = calloc(len, sizeof(char));
        }
        strncpy(*dest, *src, len);
        (*dest)[len] = '\0';
};

// prints AST
void printExtDecl(struct ExtDecl *extDecl, int level) {
        if (extDecl == NULL) {
                return;
        }
        switch (extDecl->decltor->kind) {
                case FUNCTION: {
                        fprintf(outfile,
                                "%*sFunctionDecl %s '%s'\n",
                                level * INDENT,
                                "",
                                token_names[extDecl->declspec->type],
                                extDecl->decltor->name);
                        if (extDecl->decltor->params != NULL) {
                                fprintf(outfile, "%*sParams\n", (level + 1) * INDENT, "");
                                printParams(extDecl->decltor->params, level + 1);
                        }
                        printStmt(extDecl->compStmt, level + 1);
                        break;
                }
                case DECLARATION: {
                        if (extDecl->decltor->row == 0 && extDecl->decltor->col == 0) {
                                fprintf(outfile,
                                        "%*sVariableDecl %s '%s'\n",
                                        level * INDENT,
                                        "",
                                        token_names[extDecl->declspec->type],
                                        extDecl->decltor->name);
                                printExpr(extDecl->expr, level + 1);
                                break;
                        } else {
                                if (extDecl->decltor->col == 0 && extDecl->decltor->row == 0)
                                        fprintf(outfile,
                                                "%*sArrayDecl %s '%s'[]\n",
                                                level * INDENT,
                                                "",
                                                token_names[extDecl->declspec->type],
                                                extDecl->decltor->name);
                                else if (extDecl->decltor->col == 0 && extDecl->decltor->row != 0)
                                        fprintf(outfile,
                                                "%*sArrayDecl %s '%s'[%d]\n",
                                                level * INDENT,
                                                "",
                                                token_names[extDecl->declspec->type],
                                                extDecl->decltor->name,
                                                extDecl->decltor->row);
                                else if (extDecl->decltor->row != 0 && extDecl->decltor->col != 0)
                                        fprintf(outfile,
                                                "%*sArrayDecl %s '%s'[%d][%d]\n",
                                                level * INDENT,
                                                "",
                                                token_names[extDecl->declspec->type],
                                                extDecl->decltor->name,
                                                extDecl->decltor->row,
                                                extDecl->decltor->col);
                                if (extDecl->expr != NULL)
                                        printExpr(extDecl->expr, level + 1);
                                else if (extDecl->init != NULL) {
                                        fprintf(outfile,
                                                "%*sInitializer\n",
                                                (level + 1) * INDENT,
                                                "");
                                        printInitializer(extDecl->init, level + 2);
                                } else {
                                        // zero initialized array
                                }
                                break;
                        }
                }
        }
        printExtDecl(extDecl->next, level);
};

void printBlock(struct block *block, int level) {
        if (block == NULL) return;
        assert(block->stmt == NULL || block->decl == NULL);
        if (block->stmt != NULL) {
                printStmt(block->stmt, level);
        } else if (block->decl != NULL) {
                printExtDecl(block->decl, level);
        } else {
                error("Empty block\n");
        }
        printBlock(block->next, level);
};

void printStmt(struct stmt *stmt, int level) {
        if (stmt == NULL) return;
        switch (stmt->kind) {
                case IDENT: {
                        fprintf(outfile, "%*sLabelStmt\n", level * INDENT, "");
                        printExpr(stmt->value, level + 1);
                        printStmt(stmt->then, level + 1);
                        break;
                }
                case CONTINUE: {
                        fprintf(outfile, "%*sContinueStmt\n", level * INDENT, "");
                        break;
                }
                case GOTO: {
                        fprintf(outfile, "%*sGotoStmt\n", level * INDENT, "");
                        printExpr(stmt->value, level + 1);
                        break;
                }
                case BREAK: {
                        fprintf(outfile, "%*sBreakStmt\n", level * INDENT, "");
                        break;
                }
                case DEFAULT: {
                        fprintf(outfile, "%*sDefaultStmt\n", level * INDENT, "");
                        printStmt(stmt->then, level + 1);
                        break;
                }
                case CASE: {
                        fprintf(outfile, "%*sCaseStmt\n", level * INDENT, "");
                        printExpr(stmt->cond, level + 1);
                        printStmt(stmt->then, level + 1);
                        break;
                }
                case SWITCH: {
                        fprintf(outfile, "%*sSwitchStmt\n", level * INDENT, "");
                        printExpr(stmt->cond, level + 1);
                        printStmt(stmt->then, level + 1);
                        break;
                }
                case DO: {
                        fprintf(outfile, "%*sDoStmt\n", level * INDENT, "");
                        printStmt(stmt->then, level + 1);
                        printExpr(stmt->cond, level + 1);
                        break;
                }
                case STMT_COMPOUND: {
                        fprintf(outfile, "%*sCompoundStmt\n", level * INDENT, "");
                        printBlock(stmt->body, level + 1);
                        break;
                }
                case WHILE: {
                        fprintf(outfile, "%*sWhileStmt\n", level * INDENT, "");
                        printExpr(stmt->cond, level + 1);
                        printStmt(stmt->then, level + 1);
                        break;
                }
                case FOR: {
                        fprintf(outfile, "%*sForStmt\n", level * INDENT, "");
                        if (stmt->init_kind == 1) {
                                printExtDecl(stmt->init.decl, level + 1);
                        } else {
                                printExpr(stmt->init.expr, level + 1);
                        }
                        printExpr(stmt->cond, level + 1);
                        printExpr(stmt->inc, level + 1);
                        printStmt(stmt->then, level + 1);
                        break;
                }
                case IF: {
                        fprintf(outfile, "%*sIfStmt\n", level * INDENT, "");
                        printExpr(stmt->cond, level + 1);
                        printStmt(stmt->then, level + 1);
                        printStmt(stmt->els, level + 1);
                        break;
                }
                case STMT_EXPR: {
                        fprintf(outfile, "%*sExprStmt\n", level * INDENT, "");
                        printExpr(stmt->value, level + 1);
                        break;
                }
                case RETURN: {
                        fprintf(outfile, "%*sReturnStmt\n", level * INDENT, "");
                        printExpr(stmt->value, level + 1);
                        break;
                }
                default: error("Unknown statement kind\n");
        }
};

void printExpr(struct expr *expr, int level) {
        if (expr == NULL) return;
        switch (expr->kind) {
                case INT:
                        fprintf(outfile, "%*sIntegerLiteral %d\n", level * INDENT, "", expr->value);
                        break;
                case IDENT:
                        fprintf(outfile, "%*sIdentifier '%s'\n", level * INDENT, "", expr->strLit);
                        break;
                case ADD:
                case SUB:
                case MUL:
                case DIV: {
                        if (expr->rhs == NULL) {
                                fprintf(outfile,
                                        "%*sUnaryExpr %s\n",
                                        level * INDENT,
                                        "",
                                        token_names[expr->kind]);
                        } else {
                                fprintf(outfile,
                                        "%*sArithExpr %s\n",
                                        level * INDENT,
                                        "",
                                        token_names[expr->kind]);
                        }
                        printExpr(expr->lhs, level + 1);
                        printExpr(expr->rhs, level + 1);
                        break;
                }
                case LSHIFT:
                case RSHIFT: {
                        fprintf(outfile,
                                "%*sShiftExpr %s\n",
                                level * INDENT,
                                "",
                                token_names[expr->kind]);
                        printExpr(expr->lhs, level + 1);
                        printExpr(expr->rhs, level + 1);
                        break;
                }
                case LT:
                case GT:
                case LEQ:
                case GEQ:
                case EQ:
                case NEQ: {
                        fprintf(outfile,
                                "%*sRelatExpr %s\n",
                                level * INDENT,
                                "",
                                token_names[expr->kind]);
                        printExpr(expr->lhs, level + 1);
                        printExpr(expr->rhs, level + 1);
                        break;
                }
                case AND:
                case OR:
                case XOR: {
                        fprintf(outfile,
                                "%*sBitExpr %s\n",
                                level * INDENT,
                                "",
                                token_names[expr->kind]);
                        printExpr(expr->lhs, level + 1);
                        printExpr(expr->rhs, level + 1);
                        break;
                }
                case ANDAND:
                case OROR: {
                        fprintf(outfile,
                                "%*sLogicExpr %s\n",
                                level * INDENT,
                                "",
                                token_names[expr->kind]);
                        printExpr(expr->lhs, level + 1);
                        printExpr(expr->rhs, level + 1);
                        break;
                }
                case QMARK: {
                        fprintf(outfile, "%*sCondExpr\n", level * INDENT, "");
                        printExpr(expr->lhs, level + 1);
                        printExpr(expr->rhs->lhs, level + 1);
                        printExpr(expr->rhs->rhs, level + 1);
                        break;
                }
                case INCR:
                case DECR:
                case NOT:
                case TILDA:
                case SIZEOF: {
                        fprintf(outfile,
                                "%*sUnaryExpr %s\n",
                                level * INDENT,
                                "",
                                token_names[expr->kind]);
                        printExpr(expr->lhs, level + 1);
                        break;
                }
                case ASSIGN: {
                        fprintf(outfile, "%*sAssignExpr\n", level * INDENT, "");
                        printExpr(expr->lhs, level + 1);
                        printExpr(expr->rhs, level + 1);
                        break;
                }
                case OPAR: {  // func call
                        fprintf(outfile, "%*sFuncCallExpr\n", level * INDENT, "");
                        printExpr(expr->lhs, level + 1);
                        printExpr(expr->rhs, level + 1);
                        break;
                }
                case OBR: {  // array access
                        fprintf(outfile, "%*sArrayExpr\n", level * INDENT, "");
                        printExpr(expr->lhs, level + 1);
                        printExpr(expr->rhs, level + 1);
                        break;
                }
                case DOT: {
                        fprintf(outfile, "%*sStructExpr\n", level * INDENT, "");
                        printExpr(expr->lhs, level + 1);
                        printExpr(expr->rhs, level + 1);
                        break;
                }
                case DEREF: {
                        fprintf(outfile, "%*sDerefExpr\n", level * INDENT, "");
                        printExpr(expr->lhs, level + 1);
                        printExpr(expr->rhs, level + 1);
                        break;
                }
                default: error("Unknown expression kind\n");
        }
};

void printParams(struct params *params, int level) {
        if (params == NULL) return;
        fprintf(outfile,
                "%*s%s '%s'\n",
                (level + 1) * INDENT,
                "",
                token_names[params->declspec->type],
                params->decltor->name);
        printParams(params->next, level);
};

void printInitializer(struct initializer *initializer, int level) {
        if (initializer == NULL) return;
        struct initializer *head = initializer;
        initializer = initializer->children;
        fprintf(outfile, "%*s", level * INDENT, "");
        while (initializer != NULL) {
                fprintf(outfile,
                        "%d%s",
                        initializer->value.ivalue,
                        initializer->next == NULL ? "\n" : ", ");
                initializer = initializer->next;
        }
        printInitializer(head->next, level);
};