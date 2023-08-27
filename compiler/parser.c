#include "ganymede.h"

static struct Token *ct;
static int INDENT = 4;

// either function or declaration
struct ExtDecl {
        struct ExtDecl *next;
        struct declspec *declspec;
        struct decltor *decltor;
        struct expr *expr;    // for declaration
        struct block *block;  // for function
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

struct stmt {
        struct expr *expr;
        enum Kind kind;
};

void copystr(char **dest, char **src, int len);
void consume(enum Kind kind);
struct ExtDecl *function(struct declspec **declspec, struct decltor **decltor);
struct ExtDecl *declaration(struct declspec **declspec, struct decltor **decltor);
struct expr *expr();
struct declspec *declaration_specifiers();
struct decltor *declarator();
void printBlock(struct block *block, int level);
void printStmt(struct stmt *stmt, int level);
void printExpr(struct expr *expr, int level);
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

void consume(enum Kind kind) {
        if (ct->kind != kind) {
                error("Expected %s, got %s", token_names[kind], token_names[ct->kind]);
        }
        ct = ct->next;
}

// function-definition ::=
//      declarator ("{" declaration* or statement* "}")? ;
struct ExtDecl *function(struct declspec **declspec, struct decltor **decltor) {
        struct ExtDecl *func = calloc(1, sizeof(struct ExtDecl));
        func->declspec = *declspec;
        func->decltor = *decltor;

        struct block head = {};
        struct block *cur = &head;
        if (ct->kind == OCBR) {
                consume(OCBR);
                while (ct->kind != CCBR) {
                        // declaration or statement
                        switch (ct->kind) {
                                case IDENT:
                                case CASE:
                                case DEFAULT:
                                case IF:
                                case SWITCH:
                                case WHILE:
                                case DO:
                                case FOR:
                                case GOTO:
                                case CONTINUE:
                                case BREAK: break;
                                case RETURN:
                                        consume(RETURN);
                                        cur = cur->next = calloc(1, sizeof(struct block));
                                        cur->stmt = calloc(1, sizeof(struct stmt));
                                        cur->stmt->kind = RETURN;
                                        if (ct->kind != SEMIC) {
                                                cur->stmt->expr = expr();
                                        }
                                        consume(SEMIC);
                                        break;
                                default:
                                        // declaration
                                        {
                                                struct declspec *declspec =
                                                        declaration_specifiers();
                                                struct decltor *decltor = declarator();
                                                func = func->next =
                                                        declaration(&declspec, &decltor);
                                        }
                        }
                }
                consume(CCBR);
        }
        func->block = head.next;
        return func;
};

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
                        cur->expr = expr();
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

struct declspec *declaration_specifiers() {
        struct declspec *declspec = calloc(1, sizeof(struct declspec));
        if (ct->kind == INT) {
                consume(INT);
                declspec->type = INT;
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
                        consume(OPAR);
                        decltor->kind = FUNCTION;
                        consume(CPAR);
                        return decltor;
                } else {
                        decltor->kind = DECLARATION;
                        return decltor;
                }
        }
        return decltor;
};

struct expr *expr() { return conditional_expression(); }

#define HANDLE_BINOP(opEnum, func)                  \
        if (ct->kind == opEnum) {                   \
                consume(opEnum);                    \
                struct expr *rhs = func;            \
                return new_expr(opEnum, expr, rhs); \
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
        struct expr *expr = primary_expression();
        HANDLE_BINOP(MUL, multiplicative_expression());
        HANDLE_BINOP(DIV, multiplicative_expression());
        return expr;
};

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
                        printf("%*s%s FuncExcDecl '%s'\n",
                               level * INDENT,
                               "",
                               token_names[extDecl->declspec->type],
                               extDecl->decltor->name);
                        printBlock(extDecl->block, level + 1);
                        break;
                }
                case DECLARATION: {
                        printf("%*s%s DeclExlDecl '%s'\n",
                               level * INDENT,
                               "",
                               token_names[extDecl->declspec->type],
                               extDecl->decltor->name);
                        printExpr(extDecl->expr, level + 1);
                        break;
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
                case RETURN: {
                        printf("%*sReturnStmt\n", level * INDENT, "");
                        printExpr(stmt->expr, level + 1);
                        break;
                }
                default: error("Unknown statement kind\n");
        }
};

void printExpr(struct expr *expr, int level) {
        if (expr == NULL) return;
        switch (expr->kind) {
                case INT: printf("%*sIntExpr %d\n", level * INDENT, "", expr->value); break;
                case IDENT: printf("%*sIdentExpr '%s'\n", level * INDENT, "", expr->strLit); break;
                case ADD:
                case SUB:
                case MUL:
                case DIV: {
                        printf("%*sArithExpr %s\n", level * INDENT, "", token_names[expr->kind]);
                        printExpr(expr->lhs, level + 1);
                        printExpr(expr->rhs, level + 1);
                        break;
                }
                case LSHIFT:
                case RSHIFT: {
                        printf("%*sShiftExpr %s\n", level * INDENT, "", token_names[expr->kind]);
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
                        printf("%*sRelatExpr %s\n", level * INDENT, "", token_names[expr->kind]);
                        printExpr(expr->lhs, level + 1);
                        printExpr(expr->rhs, level + 1);
                        break;
                }
                case AND:
                case OR:
                case XOR: {
                        printf("%*sBitExpr %s\n", level * INDENT, "", token_names[expr->kind]);
                        printExpr(expr->lhs, level + 1);
                        printExpr(expr->rhs, level + 1);
                        break;
                }
                case ANDAND:
                case OROR: {
                        printf("%*sLogicExpr %s\n", level * INDENT, "", token_names[expr->kind]);
                        printExpr(expr->lhs, level + 1);
                        printExpr(expr->rhs, level + 1);
                        break;
                }
                case QMARK: {
                        printf("%*sCondExpr\n", level * INDENT, "");
                        printExpr(expr->lhs, level + 1);
                        printExpr(expr->rhs->lhs, level + 1);
                        printExpr(expr->rhs->rhs, level + 1);
                        break;
                }
                default: error("Unknown expression kind\n");
        }
};
