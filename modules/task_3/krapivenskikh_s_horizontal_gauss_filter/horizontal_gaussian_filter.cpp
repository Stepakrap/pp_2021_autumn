//  Copyright 2021 Krapivenskikh Stepan

#define _USE_MATH_DEFINES
#include <mpi.h>
#include <vector>
#include <ctime>
#include <iostream>
#include <cmath>
#include <random>
#include "../../../modules/task_3/orlov_m_horizontal_gaussian_filter/horizontal_gaussian_filter.h"

int rand(std::random_device* dev, std::mt19937* rng, int max) {
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, max - 1);
    return dist(*rng);
}

double gaussFunction(int k, int l, double sigma) {
    return 1 / (2 * M_PI * sigma * sigma) * exp(-(k * k + l * l) / (2 * sigma * sigma));
}

std::vector<std::vector<char>> gaussFilterSequential(std::vector<std::vector<char>> picture, double sigma) {
    int Y = static_cast<int>(picture.size());
    int X = static_cast<int>(picture[0].size());
    std::vector<std::vector<char>> res;
    for (int y = 1; y < Y - 1; y++) {
        std::vector<char> line;
        for (int x = 1; x < X - 1; x++) {
            double brightness = 0;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    brightness += gaussFunction(k, l, sigma) * static_cast<int>(picture[y + k][x + l]);
                }
            }
            line.push_back(static_cast<char>(brightness));
        }
        res.push_back(line);
    }
    return res;
}

std::vector<std::vector<char>> gaussFilter(std::vector<std::vector<char>> picture, double sigma) {
    int Y = static_cast<int>(picture.size()), lineBlocks = Y - 2, rem, lineBlocksPerProc, procs, rank;
    int X = static_cast<int>(picture[0].size());
    std::vector<std::vector<char>> res;
    int* displs;
    int* scounts;
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    rem = lineBlocks % procs;
    lineBlocksPerProc = lineBlocks / procs;
    displs = new int[procs];
    scounts = new int[procs];
    int offset = 1;
    for (int k = 0; k < procs; k++) {
        displs[k] = offset;
        if (k < rem) {
            offset += lineBlocksPerProc + 1;
            scounts[k] = lineBlocksPerProc + 1;
        } else {
            offset += lineBlocksPerProc;
            scounts[k] = lineBlocksPerProc;
        }
    }
    //  std::cout << rank << " 1 " << displs[rank] << " " << scounts[rank] << std::endl;
    char* pixels = new char[scounts[rank] * (X - 2)];
    for (int y = displs[rank]; y < displs[rank] + scounts[rank]; y++) {
        //  std::cout << "working on line " << y << std::endl;
        for (int x = 1; x < X - 1; x++) {
            double brightness = 0;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    brightness += gaussFunction(k, l, sigma) * static_cast<int>(picture[y + k][x + l]);
                }
            }
            pixels[(y - displs[rank]) * (X - 2) + (x - 1)] = static_cast<char>(brightness);
        }
    }
    //  std::cout << rank << " 2" << std::endl;
    if (rank != 0) {
        MPI_Send(pixels, scounts[rank] * (X - 2), MPI_CHAR, 0, rank, MPI_COMM_WORLD);
    }
    //  std::cout << rank << " 3" << std::endl;
    if (rank == 0) {
        for (int k = 0; k < procs; k++) {
            char* _p = nullptr;
            if (k != 0) {
                _p = new char[scounts[k] * (X - 2)];
                MPI_Recv(_p, scounts[k] * (X - 2), MPI_CHAR, k, k, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
            for (int y = displs[k]; y < displs[k] + scounts[k]; y++) {
                std::vector<char> line(X - 2);
                for (int x = 1; x < X - 1; x++) {
                    if (k != 0) {
                        line[x - 1] = _p[(y - displs[k]) * (X - 2) + (x - 1)];
                    } else {
                        line[x - 1] = pixels[(y - displs[k]) * (X - 2) + (x - 1)];
                    }
                }
                res.push_back(line);
            }
            if (k != 0) {
                delete[] _p;
            }
        }
    }
    //  std::cout << rank << " 4" << std::endl;
    delete[] pixels;
    delete[] displs;
    delete[] scounts;
    return res;
}
