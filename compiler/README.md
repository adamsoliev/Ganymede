# Baikal COMPILER

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
It is a hand-coded simple recursive descent parser

## Semantic Routines
Here we do things like type checking

## Intermediate Representation


## Optimizers
Each optimizer (pass) takes in IR and returns optimized IR

## Code Generator
My initial plan is to target RV32I. 
Noteworthy things at this stage are register allocation, instruction selection and
sequencing.

