#include <iostream>
#include "tests/perfts.h"

int main() {
    std::cout << "Starting Chess Engine Core...\n";

    // Run the EPD suite up to Depth 5 max (to keep tests under ~30 seconds)
    // The engine will return 0 if all tests pass, and 1 if any fail.
    int exit_code = run_epd_test_suite("../test_data/standard.epd", 5);

    return exit_code;
}