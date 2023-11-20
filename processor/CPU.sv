/* verilator lint_off DECLFILENAME */
/* verilator lint_off UNUSED */
module CPU(input    logic   clk_i,
           input    logic   rst_i);

    ////////////////////
    // IF
    ////////////////////

    // IF STATE MANAGEMENT
    logic [63:0] if_pc;
    logic [31:0] if_instr;
    always_ff @(posedge clk_i) begin
        if (rst_i) if_pc <= 0;
        else if_pc <= if_pc + 1;
    end

    // INSTRUCTION CACHE LOGIC
    icache ic(
        .address_i(if_pc[31:0]), .rd_o(if_instr)
    );

    ////////////////////
    // DE
    ////////////////////

    // DE STATE MANAGEMENT
    logic [31:0] id_instr;
    logic [63:0] id_pc;
    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            id_instr <= 0;
            id_pc <= 0;
        end
        else begin
            id_instr <= if_instr;
            id_pc <= if_pc;
        end
    end

    // REGISTER FILE LOGIC
    logic [63:0] id_rs1v, id_rs2v;
    logic [4:0] temp_rd;
    logic       temp_regwrite;
    logic [63:0] temp_wdata;
    assign temp_rd = 2;
    assign temp_regwrite = 0;
    assign temp_wdata = 32;
    registerfile rf(
                .clk_i(clk_i),
                .a1_i(id_instr[19:15]), .a2_i(id_instr[24:20]), .a3_i(temp_rd),
                .we_i(temp_regwrite),
                .wd_i(temp_wdata), 
                .rd1_o(id_rs1v),
                .rd2_o(id_rs2v)
    );

    // CONTROL SIGNAL GENERATION 
    logic [6:0] id_opcode = id_instr[6:0];
    logic [4:0] id_rd = id_instr[11:7];
    logic [2:0] id_funct3 = id_instr[14:12];
    logic [6:0] id_funct7 = id_instr[31:25];

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

/* verilator lint_off DECLFILENAME */
/* verilator lint_off UNUSED */