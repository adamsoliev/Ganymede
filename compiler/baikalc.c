#include "baikalc.h"

int main(int argc, char **argv) {
    int opt;
    char *optstring = "hvf:";

    while ((opt = getopt(argc, argv, optstring)) != -1) {
        switch (opt) {
            case 'h':
                printf("Usage: tc [options]\n");
                printf("Options:\n");
                printf("  -h, --help        Print this help message\n");
                printf("  -v, --version     Print the version number\n");
                printf("  -f, --file        Specify the input file\n");
                printf("  -s, --string      Specify the input string\n");
                break;
            case 'v': printf("tc version 1.0\n"); break;
            case 'f': printf("Input file: %s\n", optarg); break;
            case 's': printf("String: %s\n", optarg); break;
            default: printf("Unknown option: %c\n", optarg); break;
        }
    }
    struct Token *token = b_scan(argv[1]);
    // print(token);
    struct decl *program = parse(token);
    print_decl(program, 0);

    // semantic_analysis(program);
    // irgen(program);
    // codegen(program);

    return 0;
}