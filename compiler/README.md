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
A simple recursive descent parser

## AST

## Optimizers

## Code Generator

