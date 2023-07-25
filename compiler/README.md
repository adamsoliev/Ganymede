# Baikal Compiler

## Design

![Compiler Design](./assets/compiler_stages.png)

Design principles
  1. Simplicity
  2. Testability

## Build
```bash
$ git clone https://github.com/adamsoliev/Baikal.git
$ cd Baikal/compiler 
$ make
```

## Scanner
It is a hand-coded scanner that generates the following types of tokens:
```
enum TokenKind {
    TK_IDENT,    
    TK_NUM,      
    TK_KEYWORD,  
    TK_STR,      // string literal 
    TK_CHAR,     // character literal
    TK_PUNCT,    
    TK_ERROR,    
    TK_EOF,      
};
```
See [baikalc.h](./baikalc.h) and [scanner tests](./tests/scanner/) for more details.

## Parser 
It is a hand-coded recursive descent parser. It generates an AST using 5 types of nodes: decl, stmt, expr, type and param_list. 
See [baikalc.h](./baikalc.h) for more details.
```
int main() {
  return 0;
}
```
For the above code, the parser will generate the following AST:
![First Example](./assets/first_example.png)

## Semantic Routines
- Name resolution 
- Type checking

## Intermediate Representation
[LLVM IR](https://llvm.org/docs/LangRef.html). It is a SSA based language that can represent 'all' modern high-level language constructs cleanly. It follows 'three address code' form and, hence, maps nicely to RISC-V assembly. 

## Optimizers
Implemented optimization passes (each takes in IR and returns optimized IR):
- Constant folding

## Code Generator
Ideally, I would like this part to consist of 3 separate stages: 
- instruction selection
- instruction scheduling 
- register allocation

Currently, however, the compiler directly maps IR to RISC-V assembly, essentially generating template-like code.


