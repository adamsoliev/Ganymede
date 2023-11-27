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
        else  begin
            if (!ex_stallIF) if_pc <= pcnext;
        end
    end

    assign if_pcplus4 = if_pc + 4;
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
        if (rst_i || ex_flushID) begin
            id_instr <= 0;
            id_pc <= 0;
            id_pcplus4 <= 0;
        end
        else begin
            if (!ex_stallID) begin
                id_instr <= if_instr;
                id_pc <= if_pc;
                id_pcplus4 <= if_pcplus4;
            end
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

    logic [2:0] LoadStoreControl; // carries whether load is signed/unsigned and load/store width 
    assign LoadStoreControl = id_funct3[1:0] == 2'b00 ? { id_funct3[2], 2'b11} :  // byte
                               id_funct3[1:0] == 2'b01 ? { id_funct3[2], 2'b10} :  // halfword
                               id_funct3[1:0] == 2'b10 ? { id_funct3[2], 2'b01} :  // word
                                                         { id_funct3[2], 2'b00};   // doubleword

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
    logic       Word;
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
        Word = 1'bx;
        unique case (id_opcode)
            7'b0010011, 7'b0011011: begin // I-type
                unique case (id_funct3)
                    3'b000: AluControl = 4'b0000; // addi
                    3'b001: AluControl = 4'b0010; // slli
                    3'b010: AluControl = 4'b0011; // slti
                    3'b011: AluControl = 4'b0100; // sltiu
                    3'b100: AluControl = 4'b0101; // xori
                    3'b101: begin
                        unique case (id_funct7[5])
                            1'b0: AluControl = 4'b0110;     // srli
                            1'b1: AluControl = 4'b0111;     // srai
                            default: AluControl = 4'bxxxx;  // error
                        endcase
                    end
                    3'b110: AluControl = 4'b1000; // ori
                    3'b111: AluControl = 4'b1001; // andi
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
                Word = id_opcode == 7'b0010011 ? 1'b0 : 1'b1;
            end
            7'b0010111, 7'b0110111: begin // auipc, lui U-type
                if (id_opcode == 7'b0010111) begin 
                    AluControl = 4'b0000; // auipc
                    AluResultSrc = 2'b01;
                end
                else begin 
                    AluControl = 4'b1010; // lui
                    AluResultSrc = 2'b00;
                end
                RegWrite = 1'b1;
                AluSrcB = 1'b1; 
                ImmSrc = 3'b001;
                Branch = 1'b0;
                Jump = 1'b0;
                WriteBackSrc = 1'b0;
                MemWrite = 1'b0;
                Word = 1'b0;
            end
            7'b0110011, 7'b0111011: begin // R-type
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
                Word = id_opcode == 7'b0110011 ? 1'b0 : 1'b1;
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
                Word = 1'b0;
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
                Word = 1'b0;
            end
            7'b0000011: begin // I-type (loads)
                AluControl = 4'b0000;
                RegWrite = 1'b1;
                AluSrcB = 1'b1; 
                ImmSrc = 3'b000;
                Branch = 1'b0;
                AluResultSrc = 2'b00;
                Jump = 1'b0;
                WriteBackSrc = 1'b1;
                MemWrite = 1'b0;
                Word = 1'b0;
            end
            7'b0100011: begin // S-type (stores)
                AluControl = 4'b0000;
                RegWrite = 1'b0;
                AluSrcB = 1'b1; 
                ImmSrc = 3'b100;
                Branch = 1'b0;
                AluResultSrc = 2'b00;
                Jump = 1'b0;
                WriteBackSrc = 1'b0;
                MemWrite = 1'b1;
                Word = 1'b0;
            end
            default: begin
                if (id_pc != 0 && id_rd == 0 && id_rs1 == 0 && id_rs2 == 0 && id_funct3 == 0 && id_funct7 == 0) begin
                    $display("RETURN VALUE: %d at PC:%h", ex_result, if_pc);
                    $finish;
                end
                AluControl = 4'b0000; // error
                RegWrite = 1'b0;
                AluSrcB = 1'b0; 
                ImmSrc = 3'b000;
                Branch = 1'b0;
                AluResultSrc = 2'b00;
                Jump = 1'b0;
                WriteBackSrc = 1'b0;
                MemWrite = 1'b0;
                Word = 1'b0;
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

    // HAZARD HANDLING
    logic [1:0] ex_forwardA, ex_forwardB;
    logic       ex_loadStall, ex_stallIF, ex_stallID, ex_flushEX, ex_flushID;
    always_comb begin
        /////////////
        // RAW HAZARD
        /////////////
        ex_forwardA = 2'bxx;
        if (((ex_rs1 == mem_rd) && mem_RegWrite) && (ex_rs1 != 0)) begin
            // forward from mem stage
            ex_forwardA = 2'b10;
        end
        else if (((ex_rs1 == wb_rd) && wb_RegWrite) && (ex_rs1 != 0)) begin
            // forward from wb stage
            ex_forwardA = 2'b11;
        end
        else begin
            // no forward
            ex_forwardA = 2'b00;
        end

        ex_forwardB = 2'bxx;
        if (((ex_rs2 == mem_rd) && mem_RegWrite) && (ex_rs2 != 0)) begin
            // forward from mem stage
            ex_forwardB = 2'b10;
        end
        else if (((ex_rs2 == wb_rd) && wb_RegWrite) && (ex_rs2 != 0)) begin
            // forward from wb stage
            ex_forwardB = 2'b11;
        end
        else begin
            // no forward
            ex_forwardB = 2'b00;
        end

        /////////////
        // LOAD HAZARD
        /////////////
        // if load is in EX stage and next instr uses to-be loaded value, stall
        ex_loadStall = ex_WriteBackSrc && ((id_rs1 == ex_rd) || (id_rs2 == ex_rd));
        ex_stallIF = ex_loadStall;
        ex_stallID = ex_loadStall;
        // ex_pcsrc is here to flush EX pipeline register when branch is taken, 
        // since we predict branch not taken
        ex_flushEX = ex_loadStall || ex_pcsrc;

        // flush ID pipeline register when branch is taken
        ex_flushID = ex_pcsrc;
    end

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
    logic         ex_Word;
    logic [4:0] ex_rs1, ex_rs2; // for forwarding
    logic [2:0] ex_LoadStoreControl;
    always_ff @(posedge clk_i) begin
        if (rst_i || ex_flushEX) begin
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
            ex_rs1 <= 0;
            ex_rs2 <= 0;
            ex_LoadStoreControl <= 0;
            ex_Word <= 0;
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
            ex_rs1 <= id_rs1;
            ex_rs2 <= id_rs2;
            ex_LoadStoreControl <= LoadStoreControl;
            ex_Word <= Word;
        end
    end

    // PC
    assign ex_pcsrc = (ex_Branch & ex_alu_ne) || ex_Jump;

    // Branch address
    assign ex_pctarget = ex_pc + ex_imm;

    // AluSrc MUX
    logic [63:0] ex_SrcA, ex_SrcB, ex_rs2vf;
    always_comb begin
        // ex_SrcA
        unique case (ex_forwardA)
            2'b00: ex_SrcA = ex_rs1v;       // no forward
            2'b10: ex_SrcA = mem_result;    // forward from mem stage
            2'b11: ex_SrcA = wb_data;       // forward from wb stage
            default: ex_SrcA = 0; 
        endcase

        unique case (ex_forwardB)
            2'b00: ex_rs2vf = ex_rs2v;       // no forward
            2'b10: ex_rs2vf = mem_result;    // forward from mem stage
            2'b11: ex_rs2vf = wb_data;       // forward from wb stage
            default: ex_rs2vf = 0; 
        endcase

        // ex_SrcB
        if (ex_AluSrcB) begin
            ex_SrcB = ex_imm;
        end
        else begin
            ex_SrcB = ex_rs2vf;
        end
    end

    // ALU
    logic [63:0] ex_alu_rs1_result;
    logic        ex_alu_ne;
    alu alu(
        .SrcA_i(ex_SrcA),
        .SrcB_i(ex_SrcB),
        .AluControl_i(ex_AluControl),
        .Word_i(ex_Word),
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
    logic [2:0]  mem_LoadStoreControl;
    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            mem_rd <= 0;
            mem_RegWrite <= 0;
            mem_result <= 0;
            mem_WriteBackSrc <= 0;
            mem_MemWrite <= 0;
            mem_rs2v <= 0;
            mem_LoadStoreControl <= 0;
        end
        else begin
            mem_rd <= ex_rd;
            mem_RegWrite <= ex_RegWrite;
            mem_result <= ex_result;
            mem_WriteBackSrc <= ex_WriteBackSrc;
            mem_MemWrite <= ex_MemWrite;
            mem_rs2v <= ex_rs2vf;
            mem_LoadStoreControl <= ex_LoadStoreControl;
        end
    end

    // DATA CACHE LOGIC
    logic [63:0] load_data;
    dcache dc(
        .clk(clk_i), 
        .address(mem_result),
        .wd(store_data),
        .wm(store_mask),
        .we(mem_MemWrite),
        .rd(load_data)
    );

    // STORES
    logic [63:0] store_data;
    logic [7:0]  store_mask;
    always_comb begin
        store_data = 0;
        store_mask = 8'b00000000;
        if (mem_LoadStoreControl[1:0] == 2'b11) begin // byte
            unique case (mem_result[2:0])
                3'b000: begin store_data = {{56{1'b0}}, mem_rs2v[7:0]};              store_mask = 8'b00000001; end
                3'b001: begin store_data = {{48{1'b0}}, mem_rs2v[7:0], {8{1'b0}}};   store_mask = 8'b00000010; end
                3'b010: begin store_data = {{40{1'b0}}, mem_rs2v[7:0], {16{1'b0}}};  store_mask = 8'b00000100; end
                3'b011: begin store_data = {{32{1'b0}}, mem_rs2v[7:0], {24{1'b0}}};  store_mask = 8'b00001000; end
                3'b100: begin store_data = {{24{1'b0}}, mem_rs2v[7:0], {32{1'b0}}};  store_mask = 8'b00010000; end
                3'b101: begin store_data = {{16{1'b0}}, mem_rs2v[7:0], {40{1'b0}}};  store_mask = 8'b00100000; end
                3'b110: begin store_data = {{8{1'b0}},  mem_rs2v[7:0], {48{1'b0}}};  store_mask = 8'b01000000; end
                3'b111: begin store_data = {mem_rs2v[7:0], {56{1'b0}}};              store_mask = 8'b10000000; end
                default:begin store_data = {64{1'b0}};                               store_mask = 8'b00000000; end
            endcase
        end
        else if (mem_LoadStoreControl[1:0] == 2'b10) begin // halfword
            unique case (mem_result[2:0])
                3'b000: begin store_data = {{48{1'b0}}, mem_rs2v[15:0]};              store_mask = 8'b00000011; end
                3'b010: begin store_data = {{32{1'b0}}, mem_rs2v[15:0], {16{1'b0}}};  store_mask = 8'b00001100; end
                3'b100: begin store_data = {{16{1'b0}}, mem_rs2v[15:0], {32{1'b0}}};  store_mask = 8'b00110000; end
                3'b110: begin store_data = {mem_rs2v[15:0], {48{1'b0}}};              store_mask = 8'b11000000; end
                default:begin store_data = {64{1'b0}};                                store_mask = 8'b00000000; end
            endcase
        end
        else if (mem_LoadStoreControl[1:0] == 2'b01) begin // word
            unique case (mem_result[2:0])
                3'b000: begin store_data = {{32{1'b0}}, mem_rs2v[31:0]};  store_mask = 8'b00001111; end
                3'b100: begin store_data = {mem_rs2v[31:0], {32{1'b0}}};  store_mask = 8'b11110000; end
                default:begin store_data = {64{1'b0}};                    store_mask = 8'b00000000; end
            endcase
        end
        else begin // doubleword
            store_data = mem_rs2v;
            store_mask = 8'b11111111;
        end
    end

    // LOADS
    logic [31:0] load_word;
    logic [15:0] load_halfword;
    logic [7:0]  load_byte;
    assign load_word        = mem_result[2] ? load_data[63:32]    : load_data[31:0];
    assign load_halfword    = mem_result[1] ? load_word[31:16]    : load_word[15:0];
    assign load_byte        = mem_result[0] ? load_halfword[15:8] : load_halfword[7:0];

    logic load_sign;
    assign load_sign = !mem_LoadStoreControl[2] & (mem_LoadStoreControl[1:0] == 2'b11 ? load_byte[7]      : 
                                                 mem_LoadStoreControl[1:0] == 2'b10 ? load_halfword[15] : 
                                                                                    load_word[31]);
    
    logic [63:0] mem_load_data;
    assign mem_load_data = mem_LoadStoreControl[1:0] == 2'b11 ? {{56{load_sign}}, load_byte}        :
                           mem_LoadStoreControl[1:0] == 2'b10 ? {{48{load_sign}}, load_halfword}    :
                           mem_LoadStoreControl[1:0] == 2'b01 ? {{32{load_sign}}, load_word}        :
                                                              load_data;

    ////////////////////
    // WB
    ////////////////////

    // WB STATE 
    logic [4:0]  wb_rd;
    logic        wb_RegWrite, wb_WriteBackSrc;
    logic [63:0] wb_result, wb_load_data;
    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            wb_rd <= 0;
            wb_RegWrite <= 0;
            wb_result <= 0;
            wb_load_data <= 0;
            wb_WriteBackSrc <= 0;
        end
        else begin
            wb_rd <= mem_rd;
            wb_RegWrite <= mem_RegWrite;
            wb_result <= mem_result;
            wb_load_data <= mem_load_data;
            wb_WriteBackSrc <= mem_WriteBackSrc;
        end
    end

    // Write Back data mux
    logic [63:0] wb_data;
    assign wb_data = wb_WriteBackSrc ? wb_load_data : wb_result;

endmodule

module icache(input     logic [31:0] address_i, 
              output    logic [31:0] rd_o
);
    logic [31:0] ICACHE[4096:0];
    initial begin 
        $readmemh("./test/mem_instr", ICACHE);
    end
    assign rd_o = ICACHE[address_i[31:2]];
endmodule

module dcache(input     logic        clk,
              input     logic [63:0] address, 
              input     logic [63:0] wd, // data
              input     logic [7:0]  wm, // mask
              input     logic        we, // enable
              output    logic [63:0] rd
);
    logic [63:0] DCACHE[100:0];
    initial begin 
        $readmemh("./test/mem_data", DCACHE);
    end

    // recalculate address for testing purposes
    logic [63:0] maddress;
    assign maddress = ({address[63:3], 3'b000} - {{48{1'b0}}, 16'h2000})/8;

    assign rd = DCACHE[maddress];
    always_ff @(posedge clk) begin
        if (we) begin
            if(wm[0]) DCACHE[maddress][ 7:0 ] <= wd[ 7:0 ];
            if(wm[1]) DCACHE[maddress][15:8 ] <= wd[15:8 ];
            if(wm[2]) DCACHE[maddress][23:16] <= wd[23:16];
            if(wm[3]) DCACHE[maddress][31:24] <= wd[31:24];
            if(wm[4]) DCACHE[maddress][39:32] <= wd[39:32];
            if(wm[5]) DCACHE[maddress][47:40] <= wd[47:40];
            if(wm[6]) DCACHE[maddress][55:48] <= wd[55:48];
            if(wm[7]) DCACHE[maddress][63:56] <= wd[63:56];
        end
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

    always_ff @(negedge clk_i) begin
        if (we_i) REGS[a3_i] <= wd_i;
    end
endmodule

module alu(input    logic [63:0]   SrcA_i,
           input    logic [63:0]   SrcB_i,
           input    logic [3:0]    AluControl_i,
           input    logic          Word_i,
           output   logic          ne_o,
           output   logic [63:0]   result_o
);
    logic [63:0] difference;
    assign difference = SrcA_i - SrcB_i;

    logic [63:0] shift_result;
    shifter sh(
        .a_i(SrcA_i),
        .b_i(SrcB_i),
        .aluControl_i(AluControl_i),
        .word_i(Word_i),
        .result_o(shift_result)
    );

    logic [63:0] alu_result;
    logic lt_flag, ltu_flag;
    always_comb begin
        lt_flag  = SrcA_i[63] == SrcB_i[63] ? difference[63] : SrcA_i[63];
        ltu_flag = SrcA_i[63] == SrcB_i[63] ? difference[63] : SrcB_i[63];
        unique case (AluControl_i)
            4'b0000: alu_result = SrcA_i + SrcB_i;      // add
            4'b0001: alu_result = difference;           // sub
            4'b0010: alu_result = shift_result;         // sll
            4'b0011: alu_result = {63'b0, lt_flag};     // slt
            4'b0100: alu_result = {63'b0, ltu_flag};    // sltu
            4'b0101: alu_result = SrcA_i ^ SrcB_i;      // xor
            4'b0110: alu_result = shift_result;         // srl
            4'b0111: alu_result = shift_result;         // sra
            4'b1000: alu_result = SrcA_i | SrcB_i;      // or
            4'b1001: alu_result = SrcA_i & SrcB_i;      // and
            4'b1010: alu_result = SrcB_i;               // lui
            default: alu_result = {64{1'bx}};
        endcase
        result_o = Word_i ? {{32{alu_result[31]}}, alu_result[31:0]} : alu_result;
    end

    assign ne_o = SrcA_i != SrcB_i;
endmodule

module shifter(input    logic [63:0] a_i,
               input    logic [63:0] b_i,
               input    logic [3:0]  aluControl_i,
               input    logic        word_i,
               output   logic [63:0] result_o
);
    logic [63:0] shift_operand;
    logic shift_fill_bit;
    logic [5:0] shamt;
    logic [64:0] shift_operand_ext;
    logic [63:0] shift_result;

    always_comb begin
        shift_operand = 'x;
        unique casez ({word_i, aluControl_i[2:1]})
            // left
            3'b?01: for (int i = 0; i < 64; i++) shift_operand[i] = a_i[63 - i]; 
            // right
            3'b011: shift_operand = a_i; 
            3'b111: shift_operand = {a_i[31:0], 32'dx};
            default: ;
        endcase

        shift_fill_bit = aluControl_i == 4'b0111 && shift_operand[63];
        shamt = word_i ? {1'b0, b_i[4:0]} : b_i[5:0];

        shift_operand_ext = {shift_fill_bit, shift_operand};
        shift_result = signed'(shift_operand_ext) >>> shamt;

        result_o = 'x;
        unique casez ({word_i, aluControl_i[2:1]})
            // left
            3'b?01: for (int i = 0; i < 64; i++) result_o[i] = shift_result[63 - i]; 
            // right
            3'b011: result_o = shift_result;
            3'b111: result_o = {32'dx, shift_result[63:32]};
            default: ;
        endcase
    end

endmodule

/* verilator lint_on WIDTH */
/* verilator lint_on UNUSED */
/* verilator lint_on DECLFILENAME */
