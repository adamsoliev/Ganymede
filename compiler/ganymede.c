#include "ganymede.h"

FILE *outfile = NULL;
const char *outfile_name = NULL;
const char *test_suite = NULL;
bool run_tests = false;

char *readFile(const char *filename) {
    // Open the file in read mode
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "%s: Error opening file\n", filename);
        return NULL;
    }

    // Determine the size of the file
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate memory for the char array
    char *content = (char *)malloc(file_size + 1);  // +1 for null-terminator
    if (content == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        fclose(file);
        return NULL;
    }

    // Read the contents of the file
    size_t bytes_read = fread(content, 1, file_size, file);
    if (bytes_read != file_size) {
        fprintf(stderr, "%s: Error reading file\n", filename);
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

char *concat(const char *str1, const char *str2) {
    int len1 = strlen(str1);
    int len2 = strlen(str2);
    char *concatted = calloc(len1 + len2 + 1, sizeof(char));
    strcpy(concatted, str1);
    strcat(concatted, str2);
    return concatted;
}

int main(int argc, char **argv) {
    int opt;
    char *optstring = "hvf:o:s:t:";
    char *input = NULL;
    FILE *infile;

    while ((opt = getopt(argc, argv, optstring)) != -1) {
        switch (opt) {
            case 'h':
                printf("Usage: ganymede [OPTION]... [FILE]...\n");
                printf("Options:\n");
                printf("  -h                 Print this help message\n");
                printf("  -v                 Print the version number\n");
                printf("  -f                 Specify the input file\n");
                printf("  -o                 Specify the output file\n");
                printf("  -s                 Specify the input string\n");
                printf("  -t TESTSUITE       Run testsuite\n");
                printf(
                    "                     TESTSUITE is 'scan' and 'parse'\n");
                return 0;
            case 'v': {
                printf("ganymede version 0.0.2\n");
                return 0;
            }
            case 'f': {
                input = readFile(optarg);
                if (input == NULL) {
                    exit(1);
                }
                break;
            }
            case 'o': {
                outfile = fopen(optarg, "w+");
                outfile_name = optarg;
                break;
            }
            case 's': input = optarg; break;
            case 't': {
                // we can't handle this from the compiler itself yet and need test.sh
                run_tests = true;
                test_suite = optarg;
                break;
            }
            default: {
                exit(1);
            }
        }
    }

    if (input == NULL) {
        printf("No input file specified\n");
        exit(1);
    }

    if (outfile == NULL) {
        outfile = stdout;
    }

    // SCANNING
    // struct Token *token = b_scan(input);
    // if (run_tests && strcmp(test_suite, "scan") == 0) {
    //     char *scan_file_name = concat(outfile_name, ".scan");
    //     FILE *temp = outfile;
    //     outfile = fopen(scan_file_name, "w+");
    //     print(token);
    //     outfile = temp;
    //     free(scan_file_name);
    //     if (strcmp(test_suite, "scan") == 0) return 0;
    // }

    // PARSING
    // struct decl *program = parse(token);
    // if (run_tests && strcmp(test_suite, "parse") == 0) {
    //     char *scan_file_name = concat(outfile_name, ".parse");
    //     FILE *temp = outfile;
    //     outfile = fopen(scan_file_name, "w+");
    //     print_decl(program, 0);
    //     outfile = temp;
    //     free(scan_file_name);
    //     if (strcmp(test_suite, "parse") == 0) return 0;
    // }

    // semantic_analysis(program);
    // irgen(program);
    // codegen(program);

    return 0;
}