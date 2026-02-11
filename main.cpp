#include <iostream>
#include <string>
#include "tests/perfts.h"
#include "src/uci/uci.h"

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--perft") {
        std::cout << "Starting Chess Engine Core...\n";
        return run_epd_test_suite("../test_data/standard.epd", 5);
    }

    return UCI::run();
}
