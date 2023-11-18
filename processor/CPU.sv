/* verilator lint_off DECLFILENAME */
/* verilator lint_off UNUSED */

module CPU(input    logic   clk,
           input    logic   rst);

    ////////////////////
    // IF
    ////////////////////
    logic [63:0] PC, PCNext, PCPlus4;
    logic [31:0] if_instr;
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
        .address_i(PC[31:0]), .rd_o(if_instr)
    );

    ////////////////////
    // DE
    ////////////////////
    logic [31:0] id_instr;
    always_ff @(posedge clk or negedge rst) begin
        if (!rst) begin
            id_instr <= 0;
        end
        else begin
            id_instr <= if_instr;
        end
    end

    logic [6:0] id_opcode = id_instr[6:0];
    logic [4:0] id_rd     = id_instr[11:7];
    logic [2:0] id_funct3 = id_instr[14:12];
    logic [4:0] id_rs1    = id_instr[19:15];
    logic [4:0] id_rs2    = id_instr[24:20];
    logic [6:0] id_funct7 = id_instr[31:25];

    logic [31:0] id_iimm = { {20{id_instr[31]}}, id_instr[31:20] };
    logic [31:0] id_simm = { {20{id_instr[31]}}, id_instr[31:25], id_instr[11:7] };
    logic [31:0] id_bimm = { {20{id_instr[31]}}, id_instr[7], id_instr[30:25], id_instr[11:8], 1'b0 };
    logic [31:0] id_uimm = { id_instr[31:12], 12'b0 };
    logic [31:0] id_jimm = { {11{id_instr[31]}}, id_instr[31], id_instr[19:12], id_instr[20], id_instr[30:21], 1'b0 };

    // TODO: MemWrite, ResultSrc
    // AluControl, RegWrite, AluSrcB, ImmSrc
    logic       RegWrite;
    logic [2:0] ImmSrc;
    logic       AluSrcB;
    logic [3:0] AluControl;
    // logic       MemWrite;
    // logic [1:0] ResultSrc;
    always_comb begin
        AluControl = 4'bxxxx;
        RegWrite = 1'bx;
        AluSrcB = 1'bx; 
        ImmSrc = 3'bxxx;
        unique case (id_opcode)
            7'b0000011, 7'b0010011: begin // I-type
                unique case (id_funct3)
                    3'b000: AluControl = 4'b0000; // addi
                    default: AluControl = 4'bxxxx; // error
                endcase 
                RegWrite = 1'b1;
                AluSrcB = 1'b1; 
                ImmSrc = 3'b000;
            end
            7'b0010111: begin // U-type
                AluControl = 4'b0000;
                RegWrite = 1'b1;
                AluSrcB = 1'b1; 
                ImmSrc = 3'b001;
            end
            7'b0110011: begin // R-type
                unique case ({id_funct3, id_funct7[5]})
                    4'b0000: AluControl = 4'b0000; // add
                    4'b0001: AluControl = 4'b0001; // sub
                    4'b0010: AluControl = 4'b0010; // sll
                    4'b0100: AluControl = 4'b0011; // slt
                    4'b0110: AluControl = 4'b0100; // sltu
                    4'b1000: AluControl = 4'b0101; // xor
                    4'b1010: AluControl = 4'b0110; // srl
                    4'b1011: AluControl = 4'b0111; // sra
                    4'b1100: AluControl = 4'b1000; // or
                    4'b1110: AluControl = 4'b1001; // and
                    default: AluControl = 4'bxxxx; // error
                endcase
                RegWrite = 1'b1;
                AluSrcB = 1'b0; 
                ImmSrc = 3'bxxx;
            end
            default: begin
                AluControl = 4'bxxxx; // error
                RegWrite = 1'bx;
                AluSrcB = 1'bx; 
                ImmSrc = 3'bxxx;
            end
        endcase
    end
    
    ////////////////////
    // EX
    ////////////////////
    logic [4:0]  ex_rd, ex_rs1, ex_rs2;
    logic [31:0] ex_imm;
    logic        ex_RegWrite;
    logic [2:0]  ex_ImmSrc;
    logic        ex_AluSrcB;
    logic [3:0]  ex_AluControl;
    always_ff @(posedge clk or negedge rst) begin
        if (!rst) begin
            ex_rd <= 0;
            ex_rs1 <= 0;
            ex_rs2 <= 0;
            ex_imm <= 0;
            ex_RegWrite <= 1'bx;
            ex_ImmSrc <= 3'bxxx;
            ex_AluSrcB <= 1'bx;
            ex_AluControl <= 4'bxxxx;
        end
        else begin
            ex_rd <= id_rd;
            ex_rs1 <= id_rs1;
            ex_rs2 <= id_rs2;
            ex_imm <= id_iimm; // TODO: ImmSrc mux
            ex_RegWrite <= RegWrite;
            ex_ImmSrc <= ImmSrc;
            ex_AluSrcB <= AluSrcB;
            ex_AluControl <= AluControl;
        end
    end

    logic [63:0] ex_rs1V, ex_rs2V;
    registerfile rf(
                .clk_i(clk),
                .a1_i(ex_rs1), .a2_i(ex_rs2), .a3_i(wb_rd),
                .we3_i(wb_RegWrite),
                .wd3_i(wb_alu_result), 
                .rd1_o(ex_rs1V),
                .rd2_o(ex_rs2V)
    );

    // AluSrc mux
    logic [63:0] ex_SrcA, ex_SrcB;
    assign ex_SrcA = ex_rs1V;
    assign ex_SrcB = !ex_AluSrcB ? ex_rs2V : {{32{ex_imm[31]}}, ex_imm};

    logic [63:0] ex_alu_result;
    alu alu(
        .SrcA_i(ex_SrcA),
        .SrcB_i(ex_SrcB),
        .AluControl_i(ex_AluControl),
        .result_o(ex_alu_result)
    );

    ////////////////////
    // MEM
    ////////////////////
    logic [4:0]  mem_rd;
    logic        mem_RegWrite;
    logic [63:0] mem_alu_result;
    always_ff @(posedge clk or negedge rst) begin
        if (!rst) begin
            mem_RegWrite <= 0;
            mem_alu_result <= 0;
            mem_rd <= 0;
        end
        else begin
            mem_RegWrite <= ex_RegWrite;
            mem_alu_result <= ex_alu_result;
            mem_rd <= ex_rd;
        end
    end

    ////////////////////
    // WB
    ////////////////////
    logic [4:0]  wb_rd;
    logic        wb_RegWrite;
    logic [63:0] wb_alu_result;
    always_ff @(posedge clk or negedge rst) begin
        if (!rst) begin
            wb_RegWrite <= 0;
            wb_alu_result <= 0;
            wb_rd <= 0;
        end
        else begin
            wb_RegWrite <= mem_RegWrite;
            wb_alu_result <= mem_alu_result;
            wb_rd <= mem_rd;
        end
    end

    ////////////////////
    // DEBUG
    ////////////////////
    logic [5:0] cycle;
    always_ff @(posedge clk or negedge rst) begin
        if (!rst) begin
            cycle <= 1;
            $display("rst   PC:%h", PC);
        end
        else begin
            cycle <= cycle + 1;
            $display("%d clk   PC:%h   instr:%h", cycle, PC, if_instr);
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
    end

    always_ff @(negedge clk_i) begin
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
           input    logic [3:0]    AluControl_i,
           output   logic [63:0]   result_o
);
    always_comb begin
        unique case (AluControl_i)
            4'b0000: result_o = SrcA_i + SrcB_i; // add
            4'b0001: result_o = SrcA_i - SrcB_i; // sub
            // 4'b0010; // sll
            // 4'b0011; // slt
            // 4'b0100; // sltu
            // 4'b0101; // xor
            4'b0101: result_o = SrcA_i ^ SrcB_i; // xor
            // 4'b0110; // srl
            // 4'b0111; // sra
            4'b1000: result_o = SrcA_i | SrcB_i; // or
            4'b1001: result_o = SrcA_i & SrcB_i; // and
            default: result_o = {64{1'bx}};
        endcase
    end

endmodule

/* verilator lint_on DECLFILENAME */
/* verilator lint_off UNUSED */