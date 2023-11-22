/* verilator lint_off DECLFILENAME */
/* verilator lint_off UNUSED */
/* verilator lint_off WIDTH */
module CPU(input    logic   clk_i,
           input    logic   rst_i);

    ////////////////////
    // IF
    ////////////////////

    // IF STATE 
    logic [63:0] if_pc, pcnext, if_pcplus4;
    logic [31:0] if_instr;
    always_ff @(posedge clk_i) begin
        if (rst_i) if_pc <= 0;
        else if_pc <= pcnext;
    end

    assign if_pcplus4 = if_pc + 1;
    assign pcnext = ex_pcsrc ? ex_pctarget : if_pcplus4;

    // INSTRUCTION CACHE LOGIC
    icache ic(
        .address_i(if_pc[31:0]), .rd_o(if_instr)
    );

    ////////////////////
    // DE
    ////////////////////

    // DE STATE 
    logic [31:0] id_instr;
    logic [63:0] id_pc, id_pcplus4;
    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            id_instr <= 0;
            id_pc <= 0;
            id_pcplus4 <= 0;
        end
        else begin
            id_instr <= if_instr;
            id_pc <= if_pc;
            id_pcplus4 <= if_pcplus4;
        end
    end

    // REGISTER FILE LOGIC
    logic [63:0] id_rs1v, id_rs2v;
    registerfile rf(
                .clk_i(clk_i),
                .a1_i(id_rs1), .a2_i(id_rs2), .a3_i(wb_rd),
                .we_i(wb_RegWrite),
                .wd_i(wb_data), 
                .rd1_o(id_rs1v),
                .rd2_o(id_rs2v)
    );

    // CONTROL SIGNAL GENERATION 
    logic [6:0] id_opcode, id_funct7;
    logic [4:0] id_rd, id_rs1, id_rs2;
    logic [2:0] id_funct3;
    assign id_opcode = id_instr[6:0];
    assign id_rd = id_instr[11:7];
    assign id_rs1 = id_instr[19:15];
    assign id_rs2 = id_instr[24:20];
    assign id_funct3 = id_instr[14:12];
    assign id_funct7 = id_instr[31:25];

    logic [31:0] id_iimm, id_simm, id_bimm, id_uimm, id_jimm;
    assign id_iimm = { {20{id_instr[31]}}, id_instr[31:20] };
    assign id_simm = { {20{id_instr[31]}}, id_instr[31:25], id_instr[11:7] };
    assign id_bimm = { {20{id_instr[31]}}, id_instr[7], id_instr[30:25], id_instr[11:8], 1'b0 };
    assign id_uimm = { id_instr[31:12], 12'b0 };
    assign id_jimm = { {11{id_instr[31]}}, id_instr[31], id_instr[19:12], id_instr[20], id_instr[30:21], 1'b0 };

    logic [3:0] AluControl;
    logic       RegWrite;
    logic       AluSrcB;
    logic [2:0] ImmSrc;
    logic       Branch;
    logic [1:0] AluResultSrc;
    logic       Jump;
    logic       WriteBackSrc; // others vs load
    logic       MemWrite;
    always_comb begin
        AluControl = 4'bxxxx;
        RegWrite = 1'bx;
        AluSrcB = 1'bx; 
        ImmSrc = 3'bxxx;
        Branch = 1'bx;
        AluResultSrc = 2'bxx;
        Jump = 1'bx;
        WriteBackSrc = 1'bx;
        MemWrite = 1'bx;
        unique case (id_opcode)
            7'b0010011: begin // I-type
                unique case (id_funct3)
                    3'b000: AluControl = 4'b0000; // addi
                    default: AluControl = 4'bxxxx; // error
                endcase 
                RegWrite = 1'b1;
                AluSrcB = 1'b1; 
                ImmSrc = 3'b000;
                Branch = 1'b0;
                AluResultSrc = 2'b00;
                Jump = 1'b0;
                WriteBackSrc = 1'b0;
                MemWrite = 1'b0;
            end
            7'b0010111, 7'b0110111: begin // auipc, lui U-type
                if (id_opcode == 7'b0010111) begin 
                    AluControl = 4'b0000; // auipc
                    AluResultSrc = 2'b01;
                end
                else begin 
                    AluControl = 4'b1010;                         // lui
                    AluResultSrc = 2'b00;
                end
                RegWrite = 1'b1;
                AluSrcB = 1'b1; 
                ImmSrc = 3'b001;
                Branch = 1'b0;
                Jump = 1'b0;
                WriteBackSrc = 1'b0;
                MemWrite = 1'b0;
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
                Branch = 1'b0;
                AluResultSrc = 2'b00;
                Jump = 1'b0;
                WriteBackSrc = 1'b0;
                MemWrite = 1'b0;
            end
            7'b1100011: begin // B-type
                AluControl = 4'b0001;
                RegWrite = 1'b0;
                AluSrcB = 1'b0; 
                ImmSrc = 3'b010;
                Branch = 1'b1;
                AluResultSrc = 2'b00;
                Jump = 1'b0;
                WriteBackSrc = 1'b0;
                MemWrite = 1'b0;
            end
            7'b1101111: begin // J-type (jal)
                AluControl = 4'bxxxx;
                RegWrite = 1'b1;
                AluSrcB = 1'b0; 
                ImmSrc = 3'b011;
                Branch = 1'b0;
                AluResultSrc = 2'b11;
                Jump = 1'b1;
                WriteBackSrc = 1'b0;
                MemWrite = 1'b0;
            end
            7'b0000011: begin // I-type (ld)
                AluControl = 4'b0000;
                RegWrite = 1'b1;
                AluSrcB = 1'b1; 
                ImmSrc = 3'b000;
                Branch = 1'b0;
                AluResultSrc = 2'b00;
                Jump = 1'b0;
                WriteBackSrc = 1'b1;
                MemWrite = 1'b0;
            end
            7'b0100011: begin // S-type (sd)
                AluControl = 4'b0000;
                RegWrite = 1'b0;
                AluSrcB = 1'b1; 
                ImmSrc = 3'b100;
                Branch = 1'b0;
                AluResultSrc = 2'b00;
                Jump = 1'b0;
                WriteBackSrc = 1'b0;
                MemWrite = 1'b1;
            end
            default: begin
                AluControl = 4'bxxxx; // error
                RegWrite = 1'bx;
                AluSrcB = 1'bx; 
                ImmSrc = 3'bxxx;
                Branch = 1'bx;
                AluResultSrc = 2'bxx;
                Jump = 1'bx;
                WriteBackSrc = 1'bx;
                MemWrite = 1'bx;
            end
        endcase
    end

    // ImmSrc mux
    logic [63:0] id_immext;
    always_comb begin
        id_immext = {64{1'b0}};
        unique case (ImmSrc)
            3'b000: id_immext = {{32{id_iimm[31]}}, id_iimm};
            3'b001: id_immext = {{32{id_uimm[31]}}, id_uimm};
            3'b010: id_immext = {{32{id_bimm[31]}}, id_bimm};
            3'b011: id_immext = {{32{id_jimm[31]}}, id_jimm};
            3'b100: id_immext = {{32{id_simm[31]}}, id_simm};
            3'b101: id_immext = {64{1'b0}};
            3'b110: id_immext = {64{1'b0}};
            3'b111: id_immext = {64{1'b0}};
            default: id_immext = {64{1'b0}};
        endcase
    end

    ////////////////////
    // EX
    ////////////////////

    // EX STATE 
    logic [63:0]  ex_pc, ex_pctarget, ex_pcplus4;
    logic [63:0]  ex_rs1v, ex_rs2v;
    logic [4:0]   ex_rd;
    logic [63:0]  ex_imm;
    logic [3:0]   ex_AluControl;
    logic         ex_RegWrite;
    logic         ex_AluSrcB;
    logic         ex_Branch;
    logic         ex_Jump;
    logic [1:0]   ex_AluResultSrc;
    logic         ex_WriteBackSrc;
    logic         ex_MemWrite;
    logic         ex_pcsrc;
    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            ex_pc <= 0;
            ex_pcplus4 <= 0;
            ex_rs1v <= 0;
            ex_rs2v <= 0;
            ex_rd <= 0;
            ex_imm <= 0;
            ex_AluControl <= 0;
            ex_RegWrite <= 0;
            ex_AluSrcB <= 0;
            ex_Branch <= 0;
            ex_Jump <= 0;
            ex_AluResultSrc <= 0;
            ex_WriteBackSrc <= 0;
            ex_MemWrite <= 0;
        end
        else begin
            ex_pc <= id_pc;
            ex_pcplus4 <= id_pcplus4;
            ex_rs1v <= id_rs1v;
            ex_rs2v <= id_rs2v;
            ex_rd <= id_rd;
            ex_imm <= id_immext;
            ex_AluControl <= AluControl;
            ex_RegWrite <= RegWrite;
            ex_AluSrcB <= AluSrcB;
            ex_Branch <= Branch;
            ex_Jump <= Jump;
            ex_AluResultSrc <= AluResultSrc;
            ex_WriteBackSrc <= WriteBackSrc;
            ex_MemWrite <= MemWrite;
        end
    end

    // PC
    assign ex_pcsrc = (ex_Branch & ex_alu_ne) || ex_Jump;

    // Branch address
    assign ex_pctarget = ex_pc + ex_imm;

    // AluSrc MUX
    logic [63:0] ex_SrcA, ex_SrcB;
    assign ex_SrcA = ex_rs1v;
    assign ex_SrcB = ex_AluSrcB ? ex_imm : ex_rs2v;

    // ALU
    logic [63:0] ex_alu_rs1_result;
    logic        ex_alu_ne;
    alu alu(
        .SrcA_i(ex_SrcA),
        .SrcB_i(ex_SrcB),
        .AluControl_i(ex_AluControl),
        .ne_o(ex_alu_ne),
        .result_o(ex_alu_rs1_result)
    );

    // ALU result mux
    logic [63:0] ex_result;
    always_comb begin
        unique case (ex_AluResultSrc)
            2'b00: ex_result = ex_alu_rs1_result;
            2'b01: ex_result = ex_pctarget; // auipc
            2'b11: ex_result = ex_pcplus4; // jal
            default: ex_result = 0;
        endcase
    end

    ////////////////////
    // MEM
    ////////////////////

    // MEM STATE 
    logic [4:0]  mem_rd;
    logic        mem_RegWrite, mem_WriteBackSrc, mem_MemWrite;
    logic [63:0] mem_result, mem_rs2v;
    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            mem_rd <= 0;
            mem_RegWrite <= 0;
            mem_result <= 0;
            mem_WriteBackSrc <= 0;
            mem_MemWrite <= 0;
            mem_rs2v <= 0;
        end
        else begin
            mem_rd <= ex_rd;
            mem_RegWrite <= ex_RegWrite;
            mem_result <= ex_result;
            mem_WriteBackSrc <= ex_WriteBackSrc;
            mem_MemWrite <= ex_MemWrite;
            mem_rs2v <= ex_rs2v;
        end
    end

    // DATA CACHE LOGIC
    logic [63:0] mem_load_result;
    dcache dc(
        .clk(clk_i), 
        .address(mem_result),
        .wd(mem_rs2v),
        .we(mem_MemWrite),
        .rd(mem_load_result)
    );

    ////////////////////
    // WB
    ////////////////////

    // WB STATE 
    logic [4:0]  wb_rd;
    logic        wb_RegWrite, wb_WriteBackSrc;
    logic [63:0] wb_result, wb_load_result;
    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            wb_rd <= 0;
            wb_RegWrite <= 0;
            wb_result <= 0;
            wb_load_result <= 0;
            wb_WriteBackSrc <= 0;
        end
        else begin
            wb_rd <= mem_rd;
            wb_RegWrite <= mem_RegWrite;
            wb_result <= mem_result;
            wb_load_result <= mem_load_result;
            wb_WriteBackSrc <= mem_WriteBackSrc;
        end
    end

    // Write Back data mux
    logic [63:0] wb_data;
    assign wb_data = wb_WriteBackSrc ? wb_load_result : wb_result;

    ////////////////////
    // TEST
    ////////////////////
    logic [63:0] if_pc_copy;
    assign if_pc_copy = if_pc;
    always_comb begin
        if (if_pc_copy > 1000) begin
            $finish();
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
                    input   logic           we_i,
                    input   logic [63:0]    wd_i,
                    output  logic [63:0]    rd1_o,
                    output  logic [63:0]    rd2_o
);
    logic [63:0] REGS[31:0];

    assign rd1_o = (a1_i != 0) ? REGS[a1_i] : 0;
    assign rd2_o = (a2_i != 0) ? REGS[a2_i] : 0;

    always_ff @(posedge clk_i) begin
        if (we_i) REGS[a3_i] <= wd_i;
    end
endmodule

module alu(input    logic [63:0]   SrcA_i,
           input    logic [63:0]   SrcB_i,
           input    logic [3:0]    AluControl_i,
           output   logic          ne_o,
           output   logic [63:0]   result_o
);
    always_comb begin
        unique case (AluControl_i)
            4'b0000: result_o = SrcA_i + SrcB_i; // add
            4'b0001: result_o = SrcA_i - SrcB_i; // sub
            4'b0010: result_o = SrcA_i << SrcB_i; // sll
            // 4'b0011; // slt
            // 4'b0100; // sltu
            4'b0101: result_o = SrcA_i ^ SrcB_i; // xor
            // 4'b0110; // srl
            // 4'b0111; // sra
            4'b1000: result_o = SrcA_i | SrcB_i; // or
            4'b1001: result_o = SrcA_i & SrcB_i; // and
            4'b1010: result_o = SrcB_i;          // lui
            default: result_o = {64{1'bx}};
        endcase
    end
    assign ne_o = SrcA_i != SrcB_i;
endmodule
/* verilator lint_on WIDTH */
/* verilator lint_on UNUSED */
/* verilator lint_on DECLFILENAME */
