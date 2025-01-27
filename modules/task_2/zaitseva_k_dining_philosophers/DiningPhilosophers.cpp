// Copyright 2021 Zaitseva Ksenia
#include "DiningPhilosophers.h"

namespace params {
MPI_Status status;
int state[MAX_COUNT];
int key_fork;
}  // namespace params

void think(int idx) {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(10, 100);
  int ms = distribution(generator);
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

void eat(int idx) { think(idx); }

void test(int i, int current_idx, int count) {
  if ((params::state[i] == HUNGRY) &&
      (params::state[(i - 1 + count) % count] != EAT) &&
      (params::state[(i + 1) % count] != EAT)) {
    params::state[i] = EAT;
    // printf("pholisopher %d eats\n", i);
    // fflush(stdout);
    if (current_idx != i)
      MPI_Send(params::state, count, MPI_INT, i, KEY, MPI_COMM_WORLD);
  }
}

void take_forks(int idx, int count) {
  MPI_Send(&params::key_fork, 1, MPI_INT, count, IN, MPI_COMM_WORLD);
  MPI_Recv(params::state, count, MPI_INT, count, IN, MPI_COMM_WORLD,
           &params::status);
  params::state[idx] = HUNGRY;
  // printf("pholisopher %d is hungry\n", idx);
  // fflush(stdout);
  test(idx, idx, count);
  MPI_Send(params::state, count, MPI_INT, count, OUT, MPI_COMM_WORLD);
  if (params::state[idx] != EAT)
    MPI_Recv(params::state, count, MPI_INT, MPI_ANY_SOURCE, KEY, MPI_COMM_WORLD,
             &params::status);
}

void put_forks(int idx, int count) {
  MPI_Send(&params::key_fork, 1, MPI_INT, count, IN, MPI_COMM_WORLD);
  MPI_Recv(params::state, count, MPI_INT, count, IN, MPI_COMM_WORLD,
           &params::status);
  params::state[idx] = THINK;
  // printf("pholisopher %d thinks\n", idx);
  // fflush(stdout);
  test((idx - 1 + count) % count, idx, count);
  test((idx + 1) % count, idx, count);
  MPI_Send(params::state, count, MPI_INT, count, OUT, MPI_COMM_WORLD);
}

void DiningPholisophers(int rank, int size, int run) {
  int i = 0;
  const int count = size - 1;
  int run_thread6 = run * count * 2;

  if (rank == size - 1) {
    while (i < run_thread6) {
      MPI_Status status1;
      MPI_Recv(&params::key_fork, 1, MPI_INT, MPI_ANY_SOURCE, IN,
               MPI_COMM_WORLD, &status1);
      if (status1.MPI_TAG == IN) {
        MPI_Send(params::state, count, MPI_INT, status1.MPI_SOURCE, IN,
                 MPI_COMM_WORLD);
        MPI_Recv(params::state, count, MPI_INT, status1.MPI_SOURCE, OUT,
                 MPI_COMM_WORLD, &status1);
      }
      i++;
    }
  } else {
    while (i < run) {
      think(rank);
      take_forks(rank, count);
      eat(rank);
      put_forks(rank, count);
      i++;
    }
  }
}
