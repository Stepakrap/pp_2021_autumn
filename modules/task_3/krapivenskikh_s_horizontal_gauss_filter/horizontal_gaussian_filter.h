//  Copyright 2021 Krapivenskikh Stepan

#pragma once
#include <mpi.h>
#include <vector>
#include <random>

int rand(std::random_device* dev, std::mt19937* rng, int max = RAND_MAX);
double gaussFunction(int k, int l, double sigma);
std::vector<std::vector<char>> gaussFilterSequential(std::vector<std::vector<char>> picture, double sigma);
std::vector<std::vector<char>> gaussFilter(std::vector<std::vector<char>> picture, double sigma);
