`default_nettype none

module uart(i_clk, o_led);
    input wire  i_clk;
    output wire o_led;

    reg [16:0] counter;
    initial counter = 0;
    always @(posedge i_clk) 
        counter <= counter + 1'b1;

    assign o_led = counter[16];
endmodule

