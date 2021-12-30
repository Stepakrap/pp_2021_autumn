//  Copyright 2021 Krapivenskikh Stepan


//  #include <chrono>
#include <gtest/gtest.h>
#include "./horizontal_gaussian_filter.h"
#include <gtest-mpi-listener.hpp>
#define MATRIX_SIZE 128


TEST(MPI, small_random_picture) {
    std::random_device dev;
    std::mt19937 rng(dev());
    rng.seed(0);
    std::vector<std::vector<char>> picture;
    for (int k = 0; k < 16; k++) {
        picture.push_back({});
        for (int l = 0; l < 16; l++) {
            if (k != l) {
                picture[k].push_back(static_cast<char>(rand(&dev, &rng) % 256));
            }
        }
    }
    //  std::cout << "1" << std::endl;
    std::vector<std::vector<char>> res1 = gaussFilter(picture, 0.5);
    //  std::cout << "2" << std::endl;
    std::vector<std::vector<char>> res2 = gaussFilterSequential(picture, 0.5);
    //  std::cout << "3" << std::endl;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        ASSERT_EQ(res1, res2);
    }
}

TEST(MPI, random_picture_1) {
    std::random_device dev;
    std::mt19937 rng(dev());
    rng.seed(1);
    std::vector<std::vector<char>> picture;
    for (int k = 0; k < MATRIX_SIZE; k++) {
        picture.push_back({});
        for (int l = 0; l < MATRIX_SIZE; l++) {
            if (k != l) {
                picture[k].push_back(static_cast<char>(rand(&dev, &rng) % 256));
            }
        }
    }
    std::vector<std::vector<char>> res1 = gaussFilter(picture, 0.5);
    std::vector<std::vector<char>> res2 = gaussFilterSequential(picture, 0.5);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        ASSERT_EQ(res1, res2);
    }
}

TEST(MPI, random_picture_2) {
    std::random_device dev;
    std::mt19937 rng(dev());
    rng.seed(2);
    std::vector<std::vector<char>> picture;
    for (int k = 0; k < MATRIX_SIZE; k++) {
        picture.push_back({});
        for (int l = 0; l < MATRIX_SIZE; l++) {
            if (k != l) {
                picture[k].push_back(static_cast<char>(rand(&dev, &rng) % 256));
            }
        }
    }
    std::vector<std::vector<char>> res1 = gaussFilter(picture, 0.5);
    std::vector<std::vector<char>> res2 = gaussFilterSequential(picture, 0.5);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        ASSERT_EQ(res1, res2);
    }
}

TEST(MPI, sigma_0) {
    std::random_device dev;
    std::mt19937 rng(dev());
    rng.seed(3);
    std::vector<std::vector<char>> picture;
    for (int k = 0; k < MATRIX_SIZE; k++) {
        picture.push_back({});
        for (int l = 0; l < MATRIX_SIZE; l++) {
            if (k != l) {
                picture[k].push_back(static_cast<char>(rand(&dev, &rng) % 256));
            }
        }
    }
    std::vector<std::vector<char>> res1 = gaussFilter(picture, 0);
    std::vector<std::vector<char>> res2 = gaussFilterSequential(picture, 0);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        ASSERT_EQ(res1, res2);
    }
}

TEST(MPI, sigma_large) {
    std::random_device dev;
    std::mt19937 rng(dev());
    rng.seed(4);
    std::vector<std::vector<char>> picture;
    for (int k = 0; k < MATRIX_SIZE; k++) {
        picture.push_back({});
        for (int l = 0; l < MATRIX_SIZE; l++) {
            if (k != l) {
                picture[k].push_back(static_cast<char>(rand(&dev, &rng) % 256));
            }
        }
    }
    std::vector<std::vector<char>> res1 = gaussFilter(picture, 10);
    std::vector<std::vector<char>> res2 = gaussFilterSequential(picture, 10);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        ASSERT_EQ(res1, res2);
    }
}

/*
TEST(MPI, random_matrix_3) {
    std::random_device dev;
    std::mt19937 rng(dev());
    rng.seed(3);
    std::vector<std::vector<double>> A;
    std::vector<double> b;
    for (int k = 0; k < MATRIX_SIZE; k++) {
        A.push_back({});
        double exceptDiagonal = 0;
        for (int l = 0; l < MATRIX_SIZE; l++) {
            if (k != l) {
                A[k].push_back(static_cast<double>(rand(&dev, &rng)) / RAND_MAX * 10);
                exceptDiagonal += A[k][l];
            }
            else {
                A[k].push_back(0);
            }
        }
        A[k][k] = static_cast<double>(rand(&dev, &rng)) / RAND_MAX * 100 + exceptDiagonal;
        b.push_back(static_cast<double>(rand(&dev, &rng)) / RAND_MAX * 10);
    }

    double t1 = MPI_Wtime();

    std::vector<double> res = gaussSeidel(A, b, 0.00001);
    double t2 = MPI_Wtime();
    double time_span1 = t2 - t1;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
        std::cout << "Parallel: " << time_span1 << " seconds\n";

    if (rank == 0) {
        for (int k = 0; k < MATRIX_SIZE; k++) {
            double sum = 0;
            for (int l = 0; l < MATRIX_SIZE; l++) {
                sum += A[k][l] * res[l];
            }
            ASSERT_NEAR(sum, b[k], 0.1);
        }
    }


    t1 = MPI_Wtime();
    if (rank == 0)
        res = sequentialGaussSeidel(A, b, 0.00001);
    t2 = MPI_Wtime();
    double time_span2 = t2 - t1;
    if (rank == 0) {
        std::cout << "Sequential: " << time_span2 << " seconds\n";
        std::cout << "Ratio: " << time_span2 / time_span1;
    }


}
*/

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
