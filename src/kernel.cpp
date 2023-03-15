#include <thread>
#include <vector>

#include "common.hpp"


int main(int argc, char** argv) {
    while (true) {
        int width, height, max_iters;
        double x_start, x_end, y_start, y_end;
        std::cin >> width >> height >> max_iters >> x_start >> x_end >> y_start >> y_end;

        unsigned char* data;
        data = (unsigned char*)malloc(width * height);

        Query q;
        q.width = width;
        q.height = height;
        q.max_iters = max_iters;
        q.x_start = x_start;
        q.x_end = x_end;
        q.y_start = y_start;
        q.y_end = y_end;

        std::vector<std::thread> threads;
        for (int i = 0; i < CPU_THREADS; i++) {
            threads.push_back(std::thread(compute, q, data, i, CPU_THREADS));
        }
        for (int i = 0; i < CPU_THREADS; i++) {
            threads[i].join();
        }

        fwrite(data, 1, width*height, stdout);
        fflush(stdout);
    }
}
