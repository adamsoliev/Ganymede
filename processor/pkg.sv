package pkg
/////////////
// Opcodes //
/////////////

typedef enum logic [6:0] {
    OPCODE_LOAD         = 7'b0000011,
    OPCODE_LOAD_FP      = 7'b0000111,
    OPCODE_MISC_MEM     = 7'b0001111,
    OPCODE_OP_IMM       = 7'b0010011,
    OPCODE_AUIPC        = 7'b0010111,
    OPCODE_OP_IMM_32    = 7'b0011011,
    OPCODE_STORE        = 7'b0100011,
    OPCODE_STORE_FP     = 7'b0100111,
    OPCODE_AMO          = 7'b0101111,
    OPCODE_OP           = 7'b0110011,
    OPCODE_LUI          = 7'b0110111,
    OPCODE_OP_32        = 7'b0111011,
    OPCODE_MADD         = 7'b1000011,
    OPCODE_MSUB         = 7'b1000111,
    OPCODE_NMSUB        = 7'b1001011,
    OPCODE_NMADD        = 7'b1001111,
    OPCODE_OP_FP        = 7'b1010011,
    OPCODE_BRANCH       = 7'b1100011,
    OPCODE_JALR         = 7'b1100111,
    OPCODE_JAL          = 7'b1101111,
    OPCODE_SYSTEM       = 7'b1110011
} opcode_e;

// Type of decoded op
typedef enum logic [3:0] {
    OP_ALU,
    OP_JUMP,
    OP_BRANCH,
    OP_MEM,
    OP_MUL,
    OP_DIV,
    OP_FP,
    OP_SYSTEM
} op_type_e;

// ALU operations
typedef enum logic [2:0] {
    // Arithmetics
    // For add, adder.use_pc and adder.use_imm should be set properly.
    ALU_ADD = 3'b000,
    ALU_SUB = 3'b001,

    // Shifts
    // Actual shift ops determined via shift_op_e.
    ALU_SHIFT = 3'b010,

    // Compare and set
    // Actual condition determined via condition_code_e
    ALU_SCC = 3'b011,

    // Logic operation
    ALU_XOR = 3'b100,
    ALU_OR  = 3'b110,
    ALU_AND = 3'b111
} alu_op_e;

// Branch/comparison condition codes
typedef enum logic [2:0] {
    CC_EQ    = 3'b000,
    CC_NE    = 3'b001,
    CC_LT    = 3'b100,
    CC_GE    = 3'b101,
    CC_LTU   = 3'b110,
    CC_GEU   = 3'b111
} condition_code_e;

// MEM operations
typedef enum logic [2:0] {
    MEM_LOAD  = 3'b001,
    MEM_STORE = 3'b010,
    MEM_LR    = 3'b101,
    MEM_SC    = 3'b110,
    MEM_AMO   = 3'b111
} mem_op_e;

// Size extension methods
typedef enum logic {
    SizeExtZero,
    SizeExtSigned
} size_ext_e;

// Opcode for shifter
// [0] determines direction (0 - left, 1 - right)
// [1] determines sign-ext (0 - logical, 1 - arithmetic)
typedef enum logic [1:0] {
  SHIFT_OP_SLL = 2'b00,
  SHIFT_OP_SRL = 2'b01,
  SHIFT_OP_SRA = 2'b11
} shift_op_e;
    
endpackage