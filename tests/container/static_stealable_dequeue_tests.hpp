//==--- tests/container/static_stealable_dequeue_tests.hpp - -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  static_stealable_dequeue_tests.hpp
/// \brief This file contains tests for a static stealable dequeue.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_TESTS_CONTAINER_STATIC_STEALABLE_DEQUEUE_TESTS_HPP
#define RIPPLE_TESTS_CONTAINER_STATIC_STEALABLE_DEQUEUE_TESTS_HPP

#include <ripple/core/arch/cpu_utils.hpp>
#include <ripple/core/container/static_stealable_dequeue.hpp>
#include <gtest/gtest.h>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

// This barrier is used to start threads at a similar time.
std::atomic<int> barrier = 0;
std::mutex       exclusive;

/// Fixture class for work stealing queue tests.
class StaticStealableDequeueTest : public ::testing::Test {
 protected:
  /// Defines the number of elements in the dequeue.
  static constexpr std::size_t queue_size = 1 << 13;
  /// Defines the number of element to push onto the queue.
  static constexpr std::size_t test_elements = queue_size << 5;

  // clang-format off
  /// Alias for the type of data in the queue.
  using data_t   = int;
  /// Alias for the type of the queue.
  using queue_t  = ripple::StaticStealableDequeue<data_t, queue_size>;
  /// Alias for the type of the restults container.
  using result_t = std::vector<data_t>;
  // clang-format on

  /// This constantly tries to steal elements from the queue, and stores any
  /// stolen results in the thread's results vector.
  /// \param thread_idx    The index of the thread.
  /// \param total_threads The total number of threads used for the test.
  void steal(int thread_idx, int total_threads) {
    ripple::set_affinity(thread_idx);
    thread_local auto thread_results = result_t{};
    barrier.fetch_add(1);

    // Spin while we wait for other theads:
    while (barrier.load() < total_threads) {}

    // Run until the main thread says stop:
    while (barrier.load() != 0) {
      // This loop is to avoid contention on the atomic:
      for (std::size_t i = 0; i < test_elements; ++i) {
        if (auto result = queue.steal()) {
          thread_results.push_back(*result);
        }
      }
    }

    std::lock_guard<std::mutex> guard(result_mutex);
    results[thread_idx] = thread_results;
  }

  /// This constantly pushes elements onto the queue, occasionally popping some.
  /// \param thread_idx      The index of the thread.
  /// \param total_threads   The total number of threads running.
  void push_and_pop(int thread_idx, int total_threads) {
    ripple::set_affinity(thread_idx);
    thread_local auto thread_results = result_t{};
    barrier.fetch_add(1);

    // Wait until all threads are ready:
    while (barrier.load() < total_threads) {}

    // Push and occasionally pop. Since threads are stealing, there should be a
    // lot of contention on the first element in the queue.
    for (size_t i = 0; i < test_elements; ++i) {
      while (!queue.try_push(i)) {
        /* Spin, queue full */
      }
      if (i & 1ull) {
        if (auto result = queue.pop()) {
          thread_results.push_back(*result);
        }
      }
    }

    // We don't need the lock here because the strealing threads can only
    // execute their writes to the global results container when this thread
    // tells them to stop stealing.
    results[thread_idx] = thread_results;

    // Tell the other threads to stop:
    barrier.store(0);
  }

  /// Generates \p elements elements for the queue.
  /// \param[in] elements   The number of elements to generate.
  void generate(std::size_t elements) {
    while (elements-- > 0) {
      queue.push(elements);
    }
  }

  /// Starts the threads by setting the first thread to push and pop, and
  /// setting the other threads to constantly steal from the queue.
  void run() {
    const std::size_t cores = ripple::topology().num_cores();
    threads.emplace_back(
      &StaticStealableDequeueTest::push_and_pop, this, 0, cores);
    run_stealers();
  }

  /// Runs only stealing threads.
  void run_stealers() {
    const std::size_t cores = ripple::topology().num_cores();
    for (std::size_t i = 1; i < cores; ++i) {
      threads.emplace_back(&StaticStealableDequeueTest::steal, this, i, cores);
    }
  }

  /// Joins the threads.
  void join() {
    barrier.store(0);
    for (auto& thread : threads) {
      thread.join();
    }
  }

  /// Set's up the results containers.
  void set_up() {
    for (std::size_t i = 0; i < ripple::topology().num_cores(); ++i)
      results.push_back(result_t());
  }

  queue_t                  queue;   //!< The queue to test.
  std::vector<result_t>    results; //!< Vectors of results for each thread.
  std::vector<std::thread> threads; //!< Thread to test the queue with.
  std::mutex               result_mutex; //!< Mutex for pushing results.
};

TEST_F(StaticStealableDequeueTest, correctly_determines_size) {
  generate(queue_size >> 1);
  EXPECT_EQ(queue.size(), (queue_size >> 1));
}

TEST_F(StaticStealableDequeueTest, can_pop_single_threaded) {
  generate(queue_size);
  for (std::size_t i = queue_size; i > 0; --i) {
    EXPECT_EQ(queue.size(), i);
    EXPECT_TRUE(queue.pop());
  }
  EXPECT_EQ(queue.size(), 0);

  for (std::size_t i = 0; i < (queue_size >> 4); ++i) {
    EXPECT_FALSE(queue.pop());
    EXPECT_EQ(queue.size(), 0);
  }
}

// This test is designed to test if the StaticStealableDequeue breaks. There are
// two cases which "break" the deque:
//
// 1. Pushing onto the queue when it is full: This is defined in the queue's
//    API, as it would require an extra check on each push to the dequeue.
//    The API requires that the user simply specify a large enough dequeue,
//    which should neven be a problem on current hardware.
//
// 2. Concurrent popping and stealing: If there is a single element in the queue
//    then the thread owning the deque (the one which may push), will be in a
//    race with the other threads (which are trying to steal) for the last
//    element. An incorrect implementation would allow pops and steals to
//    access the same element.
//
// To test case 2, this test has the first thread pusing to the queue, and
// occasionally popping, while other threads steal. As there will be more
// threads stealing, as well as the one thread having to push and pop, the
// queue will not grow and there will be constant contention to get the first
// element.
//
// Each thread stores the results of popped or stolen items, and if no item
// appears in multiple thread's results, then there has not been any error.
//
// While this test only runs a relatively small number of elements, the
// imlementation has been tested on large queue sizes (1 << 29), run multiple
// times (approximately 1 day of continuous contention on the last element) and
// ther were no data races.
TEST_F(StaticStealableDequeueTest, pop_and_steal_dont_race) {
  set_up();
  run();
  join();

  // Check that each item was only taken off the queue once:
  std::vector<size_t>                          counters(results.size(), 0);
  std::unordered_map<std::size_t, std::size_t> result_map;
  for (std::size_t i = 0; i < results.size(); ++i) {
    for (const auto& result : results[i]) {
      auto search = result_map.find(result);
      if (search != result_map.end()) {
        printf(
          "Duplicate: %12i, threads: %2lu,%2lu\n", result, i, search->second);
        EXPECT_TRUE(false);
      }
      result_map.insert({result, i});
    }
  }

  for (std::size_t element = 0; element < test_elements; ++element) {
    auto found = result_map.find(element);
    if (found == result_map.end()) {
      printf("Element %12lu was not found\n", element);
      EXPECT_TRUE(false);
    }
  }
}

// This test uses try_push, and should result in each element being found in a
// result vector. If push is used here instead of try_push, it's possible that
// the queue would have overflowed before one of the workers was able to steal
// the first elements pushed, and therefore some of the early elements wont
// be found.
TEST_F(StaticStealableDequeueTest, try_push_is_safe) {
  ripple::set_affinity(0);
  barrier.store(0);

  set_up();
  run_stealers();

  barrier.fetch_add(1);
  for (std::size_t i = 0; i < queue_size * 2; ++i) {
    while (!queue.try_push(static_cast<data_t>(i))) {}
  }

  while (queue.size()) {}
  join();

  // Check that every element is in a result vector.
  std::vector<size_t> counters(results.size(), 0);
  for (size_t element = 0; element < queue_size * 2; ++element) {
    bool found = false;
    for (std::size_t j = 0; j < results.size(); ++j) {
      if (results[j].empty()) {
        continue;
      }

      auto& index     = counters[j];
      auto& container = results[j];
      if (static_cast<size_t>(container[index]) == element) {
        found = true;
        index++;
      }
    }
    if (!found) {
      printf("Element %12lu was not found\n", element);
      ASSERT_TRUE(false);
    }
    EXPECT_TRUE(found);
  }
}

#endif // RIPPLE_TESTS_CONTAINER_STATIC_STEALABLE_DEQUEUE_TESTS_HPP
