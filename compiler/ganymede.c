#include "ganymede.h"

FILE *outfile = NULL;

char *readFile(const char *filename) {
    // Open the file in read mode
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening the file.\n");
        return NULL;
    }

    // Determine the size of the file
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate memory for the char array
    char *content = (char *)malloc(file_size + 1);  // +1 for null-terminator
    if (content == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        fclose(file);
        return NULL;
    }

    // Read the contents of the file
    size_t bytes_read = fread(content, 1, file_size, file);
    if (bytes_read != file_size) {
        fprintf(stderr, "Error reading the file.\n");
        free(content);
        fclose(file);
        return NULL;
    }

    // Null-terminate the content
    content[bytes_read] = '\0';

    // Close the file and clean up resources
    fclose(file);

    return content;
}

int main(int argc, char **argv) {
    int opt;
    char *optstring = "hvf:o:s:";
    char *input;
    FILE *infile;

    while ((opt = getopt(argc, argv, optstring)) != -1) {
        switch (opt) {
            case 'h':
                printf("Usage: tc [options]\n");
                printf("Options:\n");
                printf("  -h, --help        Print this help message\n");
                printf("  -v, --version     Print the version number\n");
                printf("  -f, --file        Specify the input file\n");
                printf("  -o, --file        Specify the output file\n");
                printf("  -s, --string      Specify the input string\n");
                break;
            case 'v': printf("tc version 1.0\n"); break;
            case 'f': {
                input = readFile(optarg);
                if (input == NULL) {
                    exit(1);
                }
                break;
            }
            case 'o': outfile = fopen(optarg, "w+"); break;
            case 's': input = optarg; break;
            default: printf("Unknown option: %s\n", optarg); break;
        }
    }
    if (outfile == NULL) {
        outfile = stdout;
    }
    struct Token *token = b_scan(input);
    // print(token);
    struct decl *program = parse(token);
    print_decl(program, 0);

    // semantic_analysis(program);
    // irgen(program);
    // codegen(program);

    return 0;
}