module CPU import pkg::*; (
    input logic clk,       // system clock 
    input logic rst        // system reset
);

    logic [31:0] IMEM [0:4096];
    logic [63:0] DMEM [0:50];
    bit [63:0] REGISTERS [0:31];

    initial begin 
        $readmemh("./test/mem_instr", IMEM);
        $readmemh("./test/mem_data", DMEM);
    end

////////////////////////////////////////////
// IF
////////////////////////////////////////////

    logic [63:0] pc;
    logic [31:0] instr;
    wire [63:0] pc_next = pc + 4;
    always_ff @(posedge clk or negedge rst) begin
        if (!rst) begin
            pc <= 0;
            instr <= 0;
        end else begin
            instr <= IMEM[pc];
            pc <= pc_next;
        end
    end

////////////////////////////////////////////
// ID
////////////////////////////////////////////

    wire [6:0] id_funct7 = instr[31:25];
    wire [4:0] id_rs2    = instr[24:20];
    wire [4:0] id_rs1    = instr[19:15];
    wire [2:0] id_funct3 = instr[14:12];
    wire [4:0] id_rd     = instr[11:7];
    wire [6:0] id_opcode = instr[6:0];

    wire [31:0] id_iimm = { {20{instr[31]}}, instr[31:20] };
    wire [31:0] id_simm = { {20{instr[31]}}, instr[31:25], instr[11:7] };
    wire [31:0] id_bimm = { {20{instr[31]}}, instr[7], instr[30:25], instr[11:8], 1'b0 };
    wire [31:0] id_uimm = { instr[31:12], 12'b0 };
    wire [31:0] id_jimm = { {11{instr[31]}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0 };

    op_type_e           id_optype;
    logic [1:0]         id_size;
    size_ext_e          id_size_ext;
    alu_op_e            id_alu_op;
    shift_op_e          id_shift_op;
    condition_code_e    id_condition;
    mem_op_e            id_mem_op;
    logic               id_adder_use_pc;
    logic               id_adder_use_imm;
    logic               id_use_imm;
    logic [31:0]        id_immediate;

    always_comb begin
        id_optype = OP_ALU;
        id_size = 2'b11;
        id_size_ext = SizeExtSigned;
        id_alu_op = alu_op_e'('x);
        id_shift_op = shift_op_e'('x);
        id_condition = condition_code_e'('x);
        unique case (opcode)
            OPCODE_JAL: begin
                id_optype = OP_JUMP;
            end
            OPCODE_JALR: begin
                id_optype = OP_JUMP;
            end
            OPCODE_BRANCH: begin
                id_optype = OP_BRANCH;
                unique case (funct3)
                    3'b000,
                    3'b001,
                    3'b100,
                    3'b101,
                    3'b110,
                    3'b111: id_condition = condition_code_e'(funct3);
                    default: ;
                endcase
            end
            OPCODE_STORE: begin
                id_optype = OP_MEM;
                id_size = funct3[1:0];
                id_mem_op = MEM_STORE;
            end
            OPCODE_LOAD: begin
                id_optype = OP_MEM;
                id_size = funct3[1:0];
                id_size_ext = funct3[2] ? SizeExtZero : SizeExtSigned;
                id_mem_op = MEM_STORE;
            end
            OPCODE_LUI: begin
                id_optype = OP_ALU;
                id_alu_op = ALU_ADD;
            end
            OPCODE_AUIPC: begin
                id_optype = OP_ALU;
                id_alu_op = ALU_ADD;
            end
            OPCODE_OP_IMM: begin
                unique case (funct3)
                    3'b000: begin
                        id_alu_op = ALU_ADD;
                    end
                    3'b010: begin
                        id_alu_op = ALU_SCC;
                        id_condition = CC_LT;
                    end
                    3'b011: begin
                        id_alu_op = ALU_SCC;
                        id_condition = CC_LTU;
                    end
                    3'b100: id_alu_op = ALU_XOR;
                    3'b110: id_alu_op = ALU_OR;
                    3'b111: id_alu_op = ALU_AND;
                    3'b001: begin
                        id_alu_op = ALU_SHIFT;
                        id_shift_op = SHIFT_OP_SLL;
                    end
                    3'b101: begin
                        id_alu_op = ALU_SHIFT;
                        if (funct7[6:1] == 6'b0) id_shift_op = SHIFT_OP_SRL;
                        else if (funct7[6:1] == 6'b010000) id_shift_op = SHIFT_OP_SRA;
                        else ;
                    end
                    default:;
                endcase
            end
            OPCODE_OP_IMM_32: begin
                id_size = 2'b10;
                unique case (funct3)
                    3'b000: begin
                        id_alu_op = ALU_ADD;
                    end
                    3'b001: begin
                        id_alu_op = ALU_SHIFT;
                        id_shift_op = SHIFT_OP_SLL;
                    end
                    3'b101: begin
                        id_alu_op = ALU_SHIFT;
                        if (funct7 == 7'b0) id_shift_op = SHIFT_OP_SRL;
                        else if (funct7 == 7'b0100000) id_shift_op = SHIFT_OP_SRA;
                        else ; 
                    end
                    default: ; 
                endcase
            end
            OPCODE_OP: begin
                unique casez ({funct7, funct3})
                    {7'b0000000, 3'b000}: begin
                        id_alu_op = ALU_ADD;
                    end
                    {7'b0100000, 3'b000}: begin
                        id_alu_op = ALU_SUB;
                    end
                    {7'b0000000, 3'b010}: begin
                        id_alu_op = ALU_SCC;
                        id_condition = CC_LT;
                    end
                    {7'b0000000, 3'b011}: begin
                        id_alu_op = ALU_SCC;
                        id_condition = CC_LTU;
                    end
                    {7'b0000000, 3'b100}: id_alu_op = ALU_XOR;
                    {7'b0000000, 3'b110}: id_alu_op = ALU_OR;
                    {7'b0000000, 3'b111}: id_alu_op = ALU_AND;
                    {7'b0000000, 3'b001}: begin
                        id_alu_op = ALU_SHIFT;
                        id_shift_op = SHIFT_OP_SLL;
                    end
                    {7'b0000000, 3'b101}: begin
                        id_alu_op = ALU_SHIFT;
                        id_shift_op = SHIFT_OP_SRL;
                    end
                    {7'b0100000, 3'b101}: begin
                        id_alu_op = ALU_SHIFT;
                        id_shift_op = SHIFT_OP_SRA;
                    end
                    default: ;
                endcase
            end
            OPCODE_OP_32: begin
                id_size = 2'b10;
                unique casez ({funct7, funct3})
                    {7'b0000000, 3'b000}: begin
                        id_alu_op = ALU_ADD;
                    end
                    {7'b0100000, 3'b000}: begin
                        id_alu_op = ALU_SUB;
                    end
                    {7'b0000000, 3'b001}: begin
                        id_alu_op = ALU_SHIFT;
                        id_shift_op = SHIFT_OP_SLL;
                    end
                    {7'b0000000, 3'b101}: begin
                        id_alu_op = ALU_SHIFT;
                        id_shift_op = SHIFT_OP_SRL;
                    end
                    {7'b0100000, 3'b101}: begin
                        id_alu_op = ALU_SHIFT;
                        id_shift_op = SHIFT_OP_SRA;
                    end
                    default: ;
                endcase
            end
            OPCODE_SYSTEM: begin
                id_optype = OP_SYSTEM;
            end
        endcase

        /////////////////////////////////////
        // Adder and ALU operand select
        /////////////////////////////////////
        id_adder_use_pc = 1'bx;
        id_adder_use_imm = 1'bx;
        id_use_imm = 1'bx;
        unique case (opcode)
            OPCODE_LOAD, OPCODE_LOAD_FP, OPCODE_STORE, OPCODE_STORE_FP, OPCODE_AMO, OPCODE_LUI, OPCODE_JALR: begin
                id_adder_use_pc = 1'b0;
                id_adder_use_imm = 1'b1;
            end
            OPCODE_OP_IMM, OPCODE_OP_IMM_32: begin
                id_adder_use_pc = 1'b0;
                id_adder_use_imm = 1'b1;
                id_use_imm = 1'b1;
            end
            OPCODE_AUIPC, OPCODE_JAL: begin
                id_adder_use_pc = 1'b1;
                id_adder_use_imm = 1'b1;
            end
            OPCODE_BRANCH: begin
                id_adder_use_pc = 1'b1;
                id_adder_use_imm = 1'b1;
                id_use_imm = 1'b0;
            end
            OPCODE_OP, OPCODE_OP_32: begin
                id_adder_use_pc = 1'b0;
                id_adder_use_imm = 1'b0;
                id_use_imm = 1'b0;
            end
            default:;
        endcase

        /////////////////////////////////////
        // Immediate select
        /////////////////////////////////////
        id_immediate = 'x;
        unique case (opcode)
            // I-type
            OPCODE_LOAD, OPCODE_OP_IMM, OPCODE_OP_IMM_32, OPCODE_JALR: begin
                id_immediate = i_imm;
            end
            // U-Type
            OPCODE_AUIPC, OPCODE_LUI: begin
                id_immediate = u_imm;
            end
            // S-Type
            OPCODE_STORE: begin
                id_immediate = s_imm;
            end
            // B-Type
            OPCODE_BRANCH: begin
                id_immediate = b_imm;
            end
            // J-Type
            OPCODE_JAL: begin
                id_immediate = j_imm;
            end
            default:;
        endcase
    end

////////////////////////////////////////////
// EX
////////////////////////////////////////////
    logic [63:0]        ex_rs1v;
    logic [63:0]        ex_rs2v;
    op_type_e           ex_optype;
    logic [1:0]         ex_wordsized;

    assign ex_rs1v = REGISTERS[id_rs1];
    assign ex_rs2v = REGISTERS[id_rs2];

    logic [63:0] alu_result;
    ALU alu (
        .wsize_i(),
        .rs1_i(ex_rs1v),
        .rs2_i(ex_rs1v),
        .result_o(alu_result)
    );

    always_comb begin 
        unique case (opcode)
            // LOAD
            OPCODE_LOAD: begin
            end
            // STORE
            OPCODE_STORE: begin
            end
            // ALU
            default: ;
        endcase
    end

////////////////////////////////////////////
// MEM
////////////////////////////////////////////

////////////////////////////////////////////
// WB
////////////////////////////////////////////

endmodule


module ALU (
    input  logic           wsize_i, // For ALU op, it is word(10) or dword(11)
    input  [63:0]          rs1_i,
    input  [63:0]          rs2_i,
    output logic [63:0]    result_o
);
    assign sum_o = rs1_i + rs2_i;

    logic [63:0] difference;
    assign difference = rs1_i - rs2_i;

    /* Result Multiplexer */
    logic [63:0] alu_result;
    always_comb begin
        unique case (decoded_op_i.alu_op)
        ALU_ADD:   alu_result = sum_o;
        ALU_SUB:   alu_result = difference;
        //   ALU_AND:   alu_result = rs1_i & operand_b;
        //   ALU_OR:    alu_result = rs1_i | operand_b;
        //   ALU_XOR:   alu_result = rs1_i ^ operand_b;
        //   ALU_SHIFT: alu_result = shift_result;
        //   ALU_SCC:   alu_result = {63'b0, compare_result_o};
        default:   alu_result = 'x;
        endcase

        result_o = wsize ? {{32{alu_result[31]}}, alu_result[31:0]} : alu_result;
    end
endmodule


