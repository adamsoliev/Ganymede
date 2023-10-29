#include <poll.h>
#include <stdio.h>
#include <stdlib.h>

#include <bitset>
#include <iostream>

#include "Vuart.h"
#include "verilated.h"

#define TXIDLE 0
#define TXDATA 1
#define RXIDLE 0
#define RXDATA 1

void print() { printf("here\n"); }

void tick(Vuart *tb) {
        tb->eval();
        tb->clk = 1;
        tb->eval();
        tb->clk = 0;
        tb->eval();
}

class UARTSIM {
        // And the pieces of the setup register broken out.
        int m_baud_counts;

        // UART state
        int m_rx_baudcounter, m_rx_state, m_rx_bits, m_last_tx;
        int m_tx_baudcounter, m_tx_state, m_tx_busy;
        unsigned m_rx_data, m_tx_data;

       public:
        UARTSIM(void) {
                setup(25);  // Set us up for a baud rate of CLK/25
                m_rx_baudcounter = 0;
                m_tx_baudcounter = 0;
                m_rx_state = RXIDLE;
                m_tx_state = TXIDLE;
        }

        void setup(unsigned isetup) { m_baud_counts = (isetup & 0x0ffffff); }

        int operator()(const int i_tx) {
                int o_rx = 1, nr = 0;

                m_last_tx = i_tx;

                if (m_rx_state == RXIDLE) {
                        if (!i_tx) {
                                m_rx_state = RXDATA;
                                m_rx_baudcounter = m_baud_counts + m_baud_counts / 2 - 1;
                                m_rx_bits = 0;
                                m_rx_data = 0;
                        }
                } else if (m_rx_baudcounter <= 0) {
                        if (m_rx_bits >= 8) {
                                m_rx_state = RXIDLE;
                                putchar(m_rx_data);
                                fflush(stdout);
                        } else {
                                m_rx_bits++;
                                m_rx_data = ((i_tx & 1) ? 0x80 : 0) | (m_rx_data >> 1);
                        }
                        m_rx_baudcounter = m_baud_counts - 1;
                } else {
                        m_rx_baudcounter--;
                }
                if (m_tx_state == TXIDLE) {
                        struct pollfd pb;
                        pb.fd = STDIN_FILENO;
                        pb.events = POLLIN;
                        if (poll(&pb, 1, 0) < 0) perror("Polling error:");

                        if (pb.revents & POLLIN) {
                                char buf[1];

                                nr = read(STDIN_FILENO, buf, 1);
                                if (1 == nr) {
                                        // << nstart_bits
                                        m_tx_data = (-1 << 10) | ((buf[0] << 1) & 0x01fe);
                                        m_tx_busy = (1 << (10)) - 1;
                                        m_tx_state = TXDATA;
                                        o_rx = 0;
                                        m_tx_baudcounter = m_baud_counts - 1;
                                } else if (nr < 0) {
                                        fprintf(stderr, "UARTSIM()::read() ERR\n");
                                }
                        }
                } else if (m_tx_baudcounter <= 0) {
                        m_tx_data >>= 1;
                        m_tx_busy >>= 1;
                        if (!m_tx_busy)
                                m_tx_state = TXIDLE;
                        else
                                m_tx_baudcounter = m_baud_counts - 1;
                        o_rx = m_tx_data & 1;
                } else {
                        m_tx_baudcounter--;
                        o_rx = m_tx_data & 1;
                }

                return o_rx;
        }
};

int main(int argc, char **argv) {
        Verilated::commandArgs(argc, argv);
        Vuart *tb = new Vuart;
        UARTSIM *uart = new UARTSIM();

        int CLOCK_RATE_HZ = 100000000;  // 100 MHz clock
        int BAUD_RATE = 115200;         // 115.2 KBaud
        unsigned baudclocks = CLOCK_RATE_HZ / BAUD_RATE;

        uart->setup(baudclocks);

        std::string message = "Hello World!";
        int charCount = message.size();
        int index = 0;
        int output = 0;
        int hz_counter = 22;
        bool sendchar = false;
        for (int i = 0; i < 12 * baudclocks * charCount; i++) {
                tick(tb);

                if (hz_counter == 0) {
                        hz_counter = CLOCK_RATE_HZ - 1;
                } else {
                        hz_counter--;
                }

                if (hz_counter == 1) sendchar = true;

                if (sendchar) {
                        tb->wr = 1;
                        tb->data = message[index];
                }

                if (sendchar && (!tb->busy)) {
                        index++;
                }
                (*uart)(tb->tx);
        }
        printf("\n");
        return 0;
}
