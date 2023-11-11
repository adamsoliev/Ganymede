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

    logic [63:0] if_pc;
    logic [31:0] instr;
    wire [63:0] pc_next = pc + 4;
    always_ff @(posedge clk or negedge rst) begin
        if (!rst) begin
            if_pc <= 0;
            instr <= 0;
        end else begin
            instr <= IMEM[pc];
            if_pc <= pc_next;
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
    logic [63:0]        id_pc;

    always_comb begin
        id_optype = OP_ALU;
        id_size = 2'b11;
        id_size_ext = SizeExtSigned;
        id_alu_op = alu_op_e'('x);
        id_shift_op = shift_op_e'('x);
        id_condition = condition_code_e'('x);

        // Forward
        id_pc = if_pc;

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

    assign ex_rs1v = REGISTERS[id_rs1];
    assign ex_rs2v = REGISTERS[id_rs2];

    logic [63:0]    ex_sum;
    logic           ex_compare_result;
    logic [63:0]    ex_alu_result;
    ALU alu (
        .size_i(id_size),
        .immediate_i(id_immediate),
        .adder_use_pc_i(id_adder_use_pc),
        .adder_use_imm_i(id_adder_use_imm),
        .use_imm_i(id_use_imm),
        .pc_i(id_pc),
        .shift_op_i(id_shift_op),
        .alu_op_i(id_alu_op),
        .condition_i(id_condition),
        .rs1_i(ex_rs1v),
        .rs2_i(ex_rs2v),
        .sum_o(ex_sum),
        .compare_result_o(ex_compare_result),
        .result_o(ex_alu_result)
    );

    // logic [63:0]    ex_alu_data_q;
    // logic [63:0]    ex_expected_pc_q;

    // logic [63:0]    ex_alu_data_d;
    // logic [63:0]    ex_expected_pc_d;
    // always_comb begin 
    //     ex_alu_data_d = ex_alu_data_q;
    //     ex_expected_pc_d = ex_alu_data_q;
    //     unique case (id_optype)
    //         OP_ALU: begin
    //             ex_alu_data = ex_alu_result;
    //         end
    //         OP_JUMP: begin
    //             ex_alu_data = 64'(signed'(pc_next));
    //             ex_expected_pc_d = {sum[63:1], 1'b0};
    //         end
    //         OP_BRANCH: begin
    //             ex_alu_data = 64'(signed'(pc_next));
    //             ex_expected_pc_d = ex_compare_result ? {sum[63:1], 1'b0} : 64'(signed'(pc_next));
    //         end
    //         OP_MEM: begin
    //         end
    //         default: ;
    //     endcase
    // end

////////////////////////////////////////////
// MEM
////////////////////////////////////////////

////////////////////////////////////////////
// WB
////////////////////////////////////////////
    always_comb begin
        if ()
    end

endmodule


module comparator import pkg::*; (
    input  logic [63:0]     operand_a_i,
    input  logic [63:0]     operand_b_i,
    input  condition_code_e condition_i,
    input  logic [63:0]     difference_i,
    output logic            result_o
);

  logic eq_flag;
  logic lt_flag;
  logic ltu_flag;
  logic result_pre_neg;

  always_comb begin
    // We don't check for difference_i == 0 because it will make the critical path longer.
    eq_flag = operand_a_i == operand_b_i;

    // If MSBs are the same, look at the sign of the result is sufficient.
    // Otherwise the one with MSB 0 is larger.
    lt_flag = operand_a_i[63] == operand_b_i[63] ? difference_i[63] : operand_a_i[63];

    // If MSBs are the same, look at the sign of the result is sufficient.
    // Otherwise the one with MSB 1 is larger.
    ltu_flag = operand_a_i[63] == operand_b_i[63] ? difference_i[63] : operand_b_i[63];

    unique case ({condition_i[2:1], 1'b0})
      CC_EQ: result_pre_neg = eq_flag;
      CC_LT: result_pre_neg = lt_flag;
      CC_LTU: result_pre_neg = ltu_flag;
      default: result_pre_neg = 'x;
    endcase

    result_o = condition_i[0] ? !result_pre_neg : result_pre_neg;
  end
endmodule

module shifter import pkg::*; (
    input  logic [63:0] operand_a_i,
    input  logic [63:0] operand_b_i,
    input  shift_op_e   shift_op_i,
    // If set, this is a word op (32-bit)
    input  logic        word_i,
    output logic [63:0] result_o
);

  // Determine the operand to be fed into the right shifter.
  logic [63:0] shift_operand;
  logic shift_fill_bit;
  logic [5:0] shamt;
  logic [64:0] shift_operand_ext;
  logic [63:0] shift_result;

  always_comb begin
    shift_operand = 'x;
    unique casez ({word_i, shift_op_i[0]})
      2'b?0: begin
        // For left shift, we reverse the contents and perform a right shift
        for (int i = 0; i < 64; i++) shift_operand[i] = operand_a_i[63 - i];
      end
      2'b01: begin
        shift_operand = operand_a_i;
      end
      2'b11: begin
        // For 32-bit shift, pad 32-bit dummy bits on the right
        shift_operand = {operand_a_i[31:0], 32'dx};
      end
      default:;
    endcase

    shift_fill_bit = shift_op_i[1] && shift_operand[63];
    shamt = word_i ? {1'b0, operand_b_i[4:0]} : operand_b_i[5:0];

    shift_operand_ext = {shift_fill_bit, shift_operand};
    shift_result = signed'(shift_operand_ext) >>> shamt;

    result_o = 'x;
    unique casez ({word_i, shift_op_i[0]})
      2'b?0: begin
        // For left shift, reverse the shifted result back.
        for (int i = 0; i < 64; i++) result_o[i] = shift_result[63 - i];
      end
      2'b01: begin
        result_o = shift_result;
      end
      2'b11: begin
        // For 32-bit shift, remove the 32-bit padded dummy bits.
        // MSBs will be fixed by the ALU unit.
        result_o = {32'dx, shift_result[63:32]};
      end
      default:;
    endcase
  end
endmodule


module alu import pkg::*; (
    input logic [1:0]      size_i,              
    input logic [31:0]     immediate_i,             
    input logic            adder_use_pc_i,              
    input logic            adder_use_imm_i,             
    input logic            use_imm_i,               
    input logic [63:0]     pc_i,                
    input shift_op_e       shift_op_i,              
    input alu_op_e         alu_op_i,                
    input condition_code_e condition_i;             
    input [63:0]           rs1_i,               
    input [63:0]           rs2_i,               
    output logic [63:0]    sum_o,               
    output logic           compare_result_o,                
    output logic [63:0]    result_o             
);

    // Determine if op is word-sized.
    // For ALU op this can only be word(10) or dword (11), so just check LSB.
    wire word = size_i[0] == 1'b0;

    wire [63:0] imm_sext = { {32{immediate_i[31]}}, immediate_i };

    // Adder. Used for ADD, LOAD, STORE, AUIPC, JAL, JALR, BRANCH
    // This is the core component of the ALU.
    // Because the adder is also used for address (load/store and branch/jump) calculation, it uses
    //   adder_use_pc and adder_use_imm to mux inputs rather than use_imm.
    assign sum_o =
        (adder_use_pc_i ? pc_i : rs1_i) +
        (adder_use_imm_i ? imm_sext : rs2_i);

    wire [63:0] operand_b = use_imm_i ? imm_sext : rs2_i;

    // Subtractor. Used for SUB, BRANCH, SLT, SLTU
    logic [63:0] difference;
    assign difference = rs1_i - operand_b;

    // Comparator. Used for BRANCH, SLT, and SLTU
    comparator comparator (
        .operand_a_i  (rs1_i),
        .operand_b_i  (operand_b),
        .condition_i  (condition_i),
        .difference_i (difference),
        .result_o     (compare_result_o)
    );

    logic [63:0] shift_result;
    shifter shifter(
        .operand_a_i (rs1_i),
        .operand_b_i (operand_b),
        .shift_op_i  (shift_op_i),
        .word_i      (word),
        .result_o    (shift_result)
    );

    /* Result Multiplexer */
    logic [63:0] alu_result;
    always_comb begin
        unique case (alu_op_i)
            ALU_ADD:   alu_result = sum_o;
            ALU_SUB:   alu_result = difference;
            ALU_AND:   alu_result = rs1_i & operand_b;
            ALU_OR:    alu_result = rs1_i | operand_b;
            ALU_XOR:   alu_result = rs1_i ^ operand_b;
            ALU_SHIFT: alu_result = shift_result;
            ALU_SCC:   alu_result = {63'b0, compare_result_o};
            default:   alu_result = 'x;
        endcase
        result_o = word ? {{32{alu_result[31]}}, alu_result[31:0]} : alu_result;
    end
endmodule

