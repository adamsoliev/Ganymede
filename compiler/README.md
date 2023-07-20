# Baikal Compiler

## Design

![Compiler Design](./assets/compiler_stages.png)

## Build
```bash
$ git clone https://github.com/adamsoliev/Baikal.git
$ cd Baikal/compiler 
$ make
```

## Scanner

```
enum TokenKind {
  KEYWORDS,
  IDENTIFIERS,
  NUMBERS,
  STRINGS,
  CHAR
}
```

```
Token {
    enum TokenKind kind;
}
```

## Parser 
It is a hand-coded recursive descent parser. It generates an AST using 5 types of nodes: decl, stmt, expr, type and param_list. Take a look at [baikalc.h](./baikalc.h) for more details.
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


