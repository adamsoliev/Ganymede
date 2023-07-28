
#include "ganymede.h"

static FILE* in;
static FILE* out;
static const int MAX_LINE_LENGTH = 256;
static const int MAX_WORD_LENGTH = 63;

static void prologue() {
    //
}

static void epilogue() {
    //
}

static void function(const char* definition) {
    char* start = strchr(definition, '@');  // Find the '@' character
    if (start != NULL) {
        char* end = strchr(start, '(');     // Find the '(' character
        if (end != NULL) {
            size_t length = end - (start + 1);
            char function_name[length + 1];
            strncpy(function_name, start + 1, length);
            function_name[length] = '\0';  // Null-terminate the function name
            // name
            fprintf(out, "  .globl  %s\n", function_name);
            fprintf(out, "  .type   %s,@function\n", function_name);
            fprintf(out, "%s:\n", function_name);
        }
    }

    // prologue();

    // body
    char line[MAX_LINE_LENGTH];
    char firstWord[MAX_WORD_LENGTH];
    while (fgets(line, sizeof(line), in) != NULL) {
        sscanf(line, "%s", firstWord);
        if (strcmp(firstWord, "ret") == 0) {
            char secondWord[MAX_WORD_LENGTH];
            char thirdWord[MAX_WORD_LENGTH];
            if (sscanf(line, "%*s %s %s", secondWord, thirdWord) == 2) {
                fprintf(out, "  li a0, %d\n", atoi(thirdWord));  // return 2
            }
            fprintf(out, "  ret\n");
        }
    }

    // epilogue();
}

void codegen() {
    in = fopen("./build/ir.ll", "r");
    if (in == NULL) {
        error(true, "cannot open file\n");
    }

    out = fopen("./build/tmp.s", "w+");
    if (out == NULL) {
        error(true, "cannot open file\n");
    }

    char line[MAX_LINE_LENGTH];
    char firstWord[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), in) != NULL) {
        sscanf(line, "%s", firstWord);
        if (strcmp(firstWord, "define") == 0) {
            function(line);
        }
    }

    fclose(in);
    fclose(out);
}