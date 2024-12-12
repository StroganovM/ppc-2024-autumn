// Copyright 2024 Stroganov Mikhail
#include "mpi/stroganov_m_dining_philosophers/include/ops_mpi.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
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

  status = 0;  // 0-размышляет, 1 - ест, 2 - голоден
  return true;
}

bool stroganov_m_dining_philosophers::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (!taskData->inputs.empty() && !taskData->inputs_count.empty() &&
        taskData->inputs_count[0] >= static_cast<int>(sizeof(int))) {
      count_philosophers = *reinterpret_cast<int*>(taskData->inputs[0]);
    } else {
      count_philosophers = world.size();
    }
  }

  return count_philosophers > 1;
}

void stroganov_m_dining_philosophers::TestMPITaskParallel::think() {
  status = 0;
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void stroganov_m_dining_philosophers::TestMPITaskParallel::eat() {
  status = 1;
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
/*
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
*/
/*
bool stroganov_m_dining_philosophers::TestMPITaskParallel::distribution_forks() {
  if (count_philosophers == 0) {
    return false;
  }

  status = 2;
  int l_status = -1;
  int r_status = -1;

  // Сначала запрашиваем статус у левого философа
  world.isend(l_philosopher, 0, status);
  world.irecv(l_philosopher, 0, l_status);

  if (l_status == 0) {
    // Если левая вилка свободна, запрашиваем статус у правого философа
    world.isend(r_philosopher, 0, status);
    world.irecv(r_philosopher, 0, r_status);

    if (r_status == 0) {
      // Если обе вилки свободны, обновляем статус
      status = 1;
      world.isend(l_philosopher, 0, status);
      world.isend(r_philosopher, 0, status);
    }
  }

  return true;
}
*/

bool stroganov_m_dining_philosophers::TestMPITaskParallel::distribution_forks() {
  status = 2;
  world.isend(l_philosopher, 0, 2);
  world.isend(r_philosopher, 0, 2);
  int l_response = 0;
  int r_response = 2;
  world.irecv(l_philosopher, 0, l_response);
  world.irecv(r_philosopher, 0, r_response);

  return true;
}

void stroganov_m_dining_philosophers::TestMPITaskParallel::release_forks() {
  status = 0;
  world.isend(l_philosopher, 0, 1);
  world.isend(r_philosopher, 0, 1);
  while (world.iprobe(l_philosopher, 0)) {
    int ack;
    world.irecv(l_philosopher, 0, ack);
  }
  while (world.iprobe(r_philosopher, 0)) {
    int ack;
    world.irecv(r_philosopher, 0, ack);
  }
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
/*
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
*/

bool stroganov_m_dining_philosophers::TestMPITaskParallel::check_deadlock() {
  int local_state = (status == 2) ? 1 : 0;
  std::vector<int> all_states(world.size(), 0);
  boost::mpi::gather(world, local_state, all_states, 0);
  bool deadlock = true;
  if (world.rank() == 0) {
    for (std::size_t i = 0; i < all_states.size(); ++i) {
      if (all_states[i] == 0) {
        deadlock = false;
        break;
      }
    }
  }
  boost::mpi::broadcast(world, deadlock, 0);
  return deadlock;
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