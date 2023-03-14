#include <thread>
#include <vector>

#include "common.hpp"


int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Invalid CLI args. Read documentation for help.\n");
        return 1;
    }

    const int width = std::stoi(argv[1]);
    const int height = std::stoi(argv[2]);
    const int max_iters = std::stoi(argv[3]);

    Query q;
    q.width = width;
    q.height = height;
    q.max_iters = max_iters;

    while (true) {
        double x_start, x_end, y_start, y_end;
        std::cin >> x_start >> x_end >> y_start >> y_end;

        char* data;
        data = (char*)malloc(width * height);

        q.x_start = x_start;
        q.x_end = x_end;
        q.y_start = y_start;
        q.y_end = y_end;
        q.data = data;

        std::vector<std::thread> threads;
        for (int i = 0; i < CPU_THREADS; i++) {
            threads.push_back(std::thread(compute, q, i, CPU_THREADS));
        }
        for (int i = 0; i < CPU_THREADS; i++) {
            threads[i].join();
        }

        fwrite(data, 1, width*height, stdout);
        fflush(stdout);
    }
}
