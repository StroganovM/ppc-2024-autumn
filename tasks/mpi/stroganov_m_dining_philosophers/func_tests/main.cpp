// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>
#include <mpi.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <thread>
#include <vector>

#include "mpi/stroganov_m_dining_philosophers/include/ops_mpi.hpp"
/*
TEST(stroganov_m_dining_philosophers, Valid_Number_Of_Philosophers) {
  boost::mpi::communicator world;
  int count_philosophers = 5;
  auto taskDataMpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_philosophers));
    taskDataMpi->inputs_count.emplace_back(sizeof(count_philosophers));
  }

  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  if (world.size() < 2) {
    GTEST_SKIP();
  }
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
}

TEST(stroganov_m_dining_philosophers, Deadlock_Free_Execution) {
  boost::mpi::communicator world;
  auto taskDataMpi = std::make_shared<ppc::core::TaskData>();
  int count_philosophers = world.size();

  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_philosophers));
    taskDataMpi->inputs_count.emplace_back(sizeof(count_philosophers));
  }

  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  if (world.size() < 2) {
    GTEST_SKIP();
  }
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
}

TEST(stroganov_m_dining_philosophers, Custom_Logic_Execution) {
  boost::mpi::communicator world;
  int count_philosophers = 4;
  auto taskDataMpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_philosophers));
    taskDataMpi->inputs_count.emplace_back(sizeof(count_philosophers));
  }

  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);
  if (world.size() < 2) {
    GTEST_SKIP();
  }

  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
}

TEST(stroganov_m_dining_philosophers, Test_default_num_philosophers) {
  boost::mpi::communicator world;

  int count_philosophers = world.size();

  // Создание данных задачи
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(count_philosophers);

  // Создание тестового объекта с передачей taskData
  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskData);

  // Проверка предварительной валидации
  if (testMpiTaskParallel.validation()) {
    ASSERT_TRUE(testMpiTaskParallel.pre_processing());
    ASSERT_TRUE(testMpiTaskParallel.run());
    ASSERT_TRUE(testMpiTaskParallel.post_processing());

    bool deadlock_detected = testMpiTaskParallel.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP() << "Skipping test due to failed validation";
  }
}

TEST(stroganov_m_dining_philosophers, Test_with_5_philosophers) {
  boost::mpi::communicator world;

  int count_philosophers = 5;

  // Создание данных задачи
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(count_philosophers);

  // Создание тестового объекта с передачей taskData
  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskData);

  // Проверка предварительной валидации
  if (testMpiTaskParallel.validation()) {
    ASSERT_TRUE(testMpiTaskParallel.pre_processing());
    ASSERT_TRUE(testMpiTaskParallel.run());
    ASSERT_TRUE(testMpiTaskParallel.post_processing());

    bool deadlock_detected = testMpiTaskParallel.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP() << "Skipping test due to failed validation";
  }
}

TEST(stroganov_m_dining_philosophers, Test_with_15_philosophers) {
  boost::mpi::communicator world;

  int count_philosophers = 15;

  // Создание данных задачи
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(count_philosophers);

  // Создание тестового объекта с передачей taskData
  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskData);

  // Проверка предварительной валидации
  if (testMpiTaskParallel.validation()) {
    ASSERT_TRUE(testMpiTaskParallel.pre_processing());
    ASSERT_TRUE(testMpiTaskParallel.run());
    ASSERT_TRUE(testMpiTaskParallel.post_processing());

    bool deadlock_detected = testMpiTaskParallel.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP() << "Skipping test due to failed validation";
  }
}

TEST(stroganov_m_dining_philosophers, Test_with_25_philosophers) {
  boost::mpi::communicator world;

  int count_philosophers = 25;

  // Создание данных задачи
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(count_philosophers);

  // Создание тестового объекта с передачей taskData
  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskData);

  // Проверка предварительной валидации
  if (testMpiTaskParallel.validation()) {
    ASSERT_TRUE(testMpiTaskParallel.pre_processing());
    ASSERT_TRUE(testMpiTaskParallel.run());
    ASSERT_TRUE(testMpiTaskParallel.post_processing());

    bool deadlock_detected = testMpiTaskParallel.check_deadlock();
    if (world.rank() == 0) {
      ASSERT_FALSE(deadlock_detected);
    }
  } else {
    GTEST_SKIP() << "Skipping test due to failed validation";
  }
}

TEST(stroganov_m_dining_philosophers, Deadlock_Handling) {
  boost::mpi::communicator world;
  int count_philosophers = world.size();
  auto taskDataMpi = std::make_shared<ppc::core::TaskData>();

  // Подготовка данных
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_philosophers));
    taskDataMpi->inputs_count.emplace_back(sizeof(count_philosophers));
  }

  stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);

  // Пропуск теста, если недостаточно процессов
  if (world.size() < 2) {
    GTEST_SKIP();
  }

  // Общие проверки
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());

  // Запуск алгоритма
  ASSERT_TRUE(testMpiTaskParallel.run());

  // Проверка дедлока на всех процессах
  bool local_deadlock = testMpiTaskParallel.check_deadlock();
  bool global_deadlock = boost::mpi::all_reduce(world, local_deadlock, std::logical_or<bool>());
  ASSERT_FALSE(global_deadlock);  // Дедлок не должен произойти

  // Проверка завершения работы всех философов
  bool local_all_think = testMpiTaskParallel.check_all_think();
  bool global_all_think = boost::mpi::all_reduce(world, local_all_think, std::logical_and<bool>());
  ASSERT_TRUE(global_all_think);  // Все философы должны завершить работу

  // Постобработка
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
}

TEST(stroganov_m_dining_philosophers, Single_Philosopher) {
  boost::mpi::communicator world;

  // Проверка работы с одним философом (неправильный случай, должен вернуть false)
  if (world.rank() == 0) {
    int count_philosophers = 1;
    auto taskDataMpi = std::make_shared<ppc::core::TaskData>();
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_philosophers));
    taskDataMpi->inputs_count.emplace_back(sizeof(count_philosophers));

    stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);

    ASSERT_FALSE(testMpiTaskParallel.validation());  // Проверка, что валидация провалится
  }
}

TEST(stroganov_m_dining_philosophers, Invalid_Philosopher_Count) {
  boost::mpi::communicator world;

  // Проверка на случай, когда количество философов не соответствует ожиданиям
  if (world.rank() == 0) {
    int count_philosophers = -5;  // Некорректное количество философов
    auto taskDataMpi = std::make_shared<ppc::core::TaskData>();
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_philosophers));
    taskDataMpi->inputs_count.emplace_back(sizeof(count_philosophers));

    stroganov_m_dining_philosophers::TestMPITaskParallel testMpiTaskParallel(taskDataMpi);

    ASSERT_FALSE(testMpiTaskParallel.validation());  // Проверка, что валидация провалится
  }
}*/