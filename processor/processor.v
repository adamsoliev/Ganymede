`default_nettype none

module processor(
    input wire clk
);
    reg [31:0] ROM [0:4096];
    initial $readmemh("/home/adam/dev/computer-stuff/cpu/test_a/rv64ui-p-add", ROM);


    //////////////////////////////
    // FETCH
    //////////////////////////////
    reg [31:0] PC = 0;
    always @(posedge clk) begin
        PC <= PC + 1;
    end

    //////////////////////////////
    // DECODE
    //////////////////////////////

    //////////////////////////////
    // EXECUTE
    //////////////////////////////

    //////////////////////////////
    // MEMORY ACCESS
    //////////////////////////////

    //////////////////////////////
    // WRITE BACK
    //////////////////////////////


    always @(posedge clk) begin
        $display("%h %d", ROM[PC], PC);
    end

endmodule
