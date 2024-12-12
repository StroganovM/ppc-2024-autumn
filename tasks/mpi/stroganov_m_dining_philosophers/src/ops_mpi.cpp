// Copyright 2024 Stroganov Mikhail
#include "mpi/stroganov_m_dining_philosophers/include/ops_mpi.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool stroganov_m_dining_philosophers::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (count_philosophers == 0) {
    return false;
  }

  l_philosopher = (world.rank() + world.size() - 1) % world.size();
  r_philosopher = (world.rank() + 1) % world.size();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, 2);
  status = distrib(gen);  // 0-размышляет, 1 - ест, 2 - голоден
  return true;
}

bool stroganov_m_dining_philosophers::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    count_philosophers = taskData->inputs_count[0];
  } else {
    count_philosophers = world.size();
  }

  return count_philosophers >= 2;
}

void stroganov_m_dining_philosophers::TestMPITaskParallel::think() {
  status = 0;
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void stroganov_m_dining_philosophers::TestMPITaskParallel::eat() {
  status = 1;
  std::this_thread::sleep_for(std::chrono::milliseconds(80));
}

void stroganov_m_dining_philosophers::TestMPITaskParallel::release_forks() {
  status = 0;
  world.send(l_philosopher, 0, status);
  world.send(r_philosopher, 0, status);
}

bool stroganov_m_dining_philosophers::TestMPITaskParallel::distribution_forks() {
  if (count_philosophers == 0) {
    return false;
  }

  status = 2;
  int l_status = -1;
  int r_status = -1;

  if (world.rank() % 2 == 0) {
    world.isend(l_philosopher, 0, status);
    world.irecv(l_philosopher, 0, l_status);
    if (l_status == 0) {
      world.isend(r_philosopher, 0, status);
      world.irecv(r_philosopher, 0, r_status);
      if (r_status == 0) {
        status = 1;
        world.isend(l_philosopher, 0, status);
        world.isend(r_philosopher, 0, status);
      }
    }
  } else {
    world.recv(r_philosopher, 0, r_status);
    if (r_status == 0) {
      world.isend(l_philosopher, 0, status);
      world.irecv(l_philosopher, 0, l_status);
      if (l_status == 0) {
        status = 1;
        world.isend(l_philosopher, 0, status);
        world.isend(r_philosopher, 0, status);
      }
    }
  }
  return true;
}

bool stroganov_m_dining_philosophers::TestMPITaskParallel::run() {
  internal_order_test();
  bool done = false;
  while (!done) {
    think();
    distribution_forks();
    eat();
    release_forks();
    if (check_deadlock()) return false;
    done = check_all_think();
  }
  return true;
}

bool stroganov_m_dining_philosophers::TestMPITaskParallel::check_all_think() {
  std::vector<int> all_states;
  boost::mpi::all_gather(world, status, all_states);
  world.barrier();
  return std::all_of(all_states.begin(), all_states.end(), [](int state) { return state == 0; });
}

bool stroganov_m_dining_philosophers::TestMPITaskParallel::check_deadlock() {
  std::vector<int> all_states(world.size(), 0);
  boost::mpi::all_gather(world, status, all_states);
  for (const int& state : all_states) {
    if (state != 2) {
      return false;
    }
  }
  return true;
}

bool stroganov_m_dining_philosophers::TestMPITaskParallel::post_processing() {
  internal_order_test();
  world.barrier();
  while (world.iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG)) {
    int leftover_message;
    world.recv(MPI_ANY_SOURCE, MPI_ANY_TAG, leftover_message);
  }
  return true;
}