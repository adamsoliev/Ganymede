# Ganymede Compiler

## Design

![Compiler Design](./assets/compiler_stages.png)

Design principles
  1. Simplicity
  2. Modularity
  3. Testability

## Build
```bash
$ git clone https://github.com/adamsoliev/Ganymede.git
$ cd Ganymede/compiler 
$ make
```

## Data Structures
![Data Structures](./assets/data_structures_9_8_23.png)

## Scanner
It is a hand-coded scanner that generates tokens for every keyword and type of symbol in C99.
See [ganymede.h](./ganymede.h) and [scanner tests](./tests/scanner/) for more details.

## Parser 
It is a hand-coded recursive descent parser. It generates an AST (complete program), consisting of a sequence of ExcDecl structs. Each external declaration can either represent a variable declaration or function and is defined by its specifiers (struct declspec) and declarator (struct decltor). 

A simple variable declaration has its specifiers in declspec and its name in decltor. If initialized, the value is stored in expr. For more complex variables (arrays, including strings, and pointers), both declspec and decltor have corresponding fields carrying necessary information.  

A function has its specifiers in declspec, its name and parameters in decltor and its body in compStmt. A compound statement consists of a sequence of block structs, which can either point to ExcDecl or stmt. (note: compound statement is applicable to any curly-brace enclosed block). See [ganymede.h](./ganymede.h) for more details.

## Resources
- Introduction to Compilers and Language Design by Douglas Thain is a short book but has a complete description of data structures used in building an AST, which is nice to have when you are trying to come up with an AST design of your own.  


