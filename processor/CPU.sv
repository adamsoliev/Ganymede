/* verilator lint_off DECLFILENAME */
/* verilator lint_off UNUSED */

module CPU(input    logic   clk,
           input    logic   rst);

    ////////////////////
    // IF
    ////////////////////
    logic [63:0] PC, PCNext, PCPlus4;
    logic [31:0] instr;
    always_ff @(posedge clk or negedge rst) begin
        if (!rst) begin
            PC <= 0;
        end
        else begin
            PC <= PCNext;
        end
    end

    assign PCPlus4 = PC + 1;
    assign PCNext = PCPlus4;

    icache ic(
        .address_i(PC[31:0]), .rd_o(instr)
    );

    ////////////////////
    // DE
    ////////////////////
    logic [6:0] id_opcode = instr[6:0];
    logic [4:0] id_rd     = instr[11:7];
    logic [2:0] id_funct3 = instr[14:12];
    logic [4:0] id_rs1    = instr[19:15];
    logic [4:0] id_rs2    = instr[24:20];
    logic [6:0] id_funct7 = instr[31:25];

    logic [31:0] id_iimm = { {20{instr[31]}}, instr[31:20] };
    logic [31:0] id_simm = { {20{instr[31]}}, instr[31:25], instr[11:7] };
    logic [31:0] id_bimm = { {20{instr[31]}}, instr[7], instr[30:25], instr[11:8], 1'b0 };
    logic [31:0] id_uimm = { instr[31:12], 12'b0 };
    logic [31:0] id_jimm = { {11{instr[31]}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0 };


    ////////////////////
    // EX
    ////////////////////
    logic [63:0] ex_SrcA, ex_SrcB;
    registerfile rf(
                .clk_i(clk),
                .a1_i(id_rs1), .a2_i(id_rs2), .a3_i(id_rd),
                .we3_i(RegWrite),
                .wd3_i(WriteBackData),
                .rd1_o(ex_SrcA),
                .rd2_o(ex_SrcB)
    );

    // immSrc mux
    logic [63:0] Imm;
    assign Imm = id_iimm;

    // aluSrc mux
    logic [63:0] SrcB;
    always_comb begin
        unique case (id_opcode)
            6'b0010011: begin // imm
                SrcB = Imm;
            end
            6'b0110011: begin // rs2
                SrcB = ex_SrcB;
            end
        endcase
    end

    logic [1:0] ex_alusrc;
    logic [63:0] ex_alu_result;
    alu alu(
        .SrcA_i(ex_SrcA),
        .SrcB_i(SrcB),
        .result_o(ex_alu_result)
    );

    ////////////////////
    // MEM
    ////////////////////

    ////////////////////
    // WB
    ////////////////////

    // RegWrite signal
    logic RegWrite;
    always_comb begin
        RegWrite = 1'b0;
        unique case (id_opcode)
            6'b0010011, 6'b0110011: begin 
                RegWrite = 1'b1;
            end
        endcase
    end

    // RegWriteBack Date
    logic [63:0] WriteBackData;
    assign WriteBackData = ex_alu_result;

    ////////////////////
    // DEBUG
    ////////////////////
    always_ff @(posedge clk or negedge rst) begin
        if (!rst) begin
            $display("rst   PC:%h", PC);
        end
        else begin
            $display("clk   PC:%h   instr:%h", PC, instr);
        end
    end

endmodule


module icache(input     logic [31:0] address_i, 
              output    logic [31:0] rd_o
);
    logic [31:0] ICACHE[100:0];
    initial begin 
        $readmemh("./test/mem_instr", ICACHE);
    end

    assign rd_o = ICACHE[address_i];

endmodule

module dcache(input     logic        clk,
              input     logic [63:0] address, 
              input     logic [63:0] wd,
              input     logic        we,
              output    logic [63:0] rd
);
    logic [63:0] DCACHE[100:0];
    initial begin 
        $readmemh("./test/mem_data", DCACHE);
    end

    assign rd = DCACHE[address];

    always_ff @(posedge clk) begin
        if (we) DCACHE[address] <= wd;
    end

endmodule

module registerfile(input   logic           clk_i,
                    input   logic [4:0]     a1_i, a2_i, a3_i,
                    input   logic           we3_i,
                    input   logic [63:0]    wd3_i,
                    output  logic [63:0]    rd1_o,
                    output  logic [63:0]    rd2_o
);
    logic [63:0] REGS[31:0];

    assign rd1_o = (a1_i != 0) ? REGS[a1_i] : 0;
    assign rd2_o = (a2_i != 0) ? REGS[a2_i] : 0;

    always_ff @(posedge clk_i) begin
        if (we3_i) REGS[a3_i] <= wd3_i;

        // DEBUG
        $display(" x0:     %h     x1(ra): %h        x2(sp): %h        x3(gp): %h", REGS[ 0], REGS[ 1], REGS[ 2], REGS[ 3]);
        $display(" x4(tp): %h     x5(t0): %h        x6(t1): %h        x7(t2): %h", REGS[ 4], REGS[ 5], REGS[ 6], REGS[ 7]);
        $display(" x8(fp): %h     x9(s1): %h       x10(a0): %h       x11(a1): %h", REGS[ 8], REGS[ 9], REGS[10], REGS[11]);
        $display("x12(a2): %h    x13(a3): %h       x14(a4): %h       x15(a5): %h", REGS[12], REGS[13], REGS[14], REGS[15]);
        $display("x16(a6): %h    x17(a7): %h       x18(s2): %h       x19(s3): %h", REGS[16], REGS[17], REGS[18], REGS[19]);
        $display("x20(s4): %h    x21(s5): %h       x22(s6): %h       x23(s7): %h", REGS[20], REGS[21], REGS[22], REGS[23]);
        $display("x24(s8): %h    x25(s9): %h      x26(s10): %h      x27(s11): %h", REGS[24], REGS[25], REGS[26], REGS[27]);
        $display("x28(t3): %h    x29(t4): %h       x30(t5): %h       x31(t6): %h", REGS[28], REGS[29], REGS[30], REGS[31]);
        $display("\n");
    end

endmodule

module alu(input    logic [63:0]   SrcA_i,
           input    logic [63:0]   SrcB_i,
           output   logic [63:0]   result_o
);
    assign result_o = SrcA_i + SrcB_i;

endmodule

/* verilator lint_on DECLFILENAME */
/* verilator lint_off UNUSED */