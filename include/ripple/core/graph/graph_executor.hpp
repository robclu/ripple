//==--- ripple/core/graph/graph_executor.hpp --------------- -*- C++ -*- ---==//
//
//                                 Ripple
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  graph_executor.hpp
/// \brief This file implements an executor for graphs.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_GRAPH_GRAPH_EXECUTOR_HPP
#define RIPPLE_GRAPH_GRAPH_EXECUTOR_HPP

#include "graph.hpp"
#include "stealer.hpp"
#include <ripple/core/arch/topology.hpp>
#include <ripple/core/container/static_stealable_dequeue.hpp>
#include <chrono>
#include <thread>
#include <variant>

namespace ripple {

/// Returns a reference to the graph executor.
static auto graph_executor() -> GraphExecutor&;

/// Index of the thread.
static thread_local uint32_t thread_id = 0;

/// The GraphExecutor class maintains a pool of threads, which can execute a
/// Graph by pushing nodes of the Graph onto the threads and running the node's
/// work.
class GraphExecutor {
 public:
  /// Friend function to access the global graph executor.
  friend auto graph_executor() -> GraphExecutor&;

  /// Sets the stealing policy for the executor.
  auto set_steal_policy(StealPolicy policy) noexcept -> void {
    // clang-format off
    switch (policy) {
      case StealPolicy::random     : _stealer = random_stealer_t()   ; return;
      case StealPolicy::neighbour  : _stealer = neighbour_stealer_t(); return;
      case StealPolicy::topological: _stealer = topo_stealer_t()     ; return;
      default: return;
    }
    // clang-format on
  }

  /// Executes a graph.
  auto execute(const Graph& graph) -> void {
    // Check if the threads need to be woken up:
    if (_exec_state != ExecutionState::run) {
      _exec_state.store(ExecutionState::run, std::memory_order_relaxed);
      for (auto& thread_state : _thread_states) {
        thread_state.run_state.store(ThreadState::RunState::running);
      }
    }

    // Set all other threads to be

    auto& queue = _queues[thread_id];
    for (auto& node : graph._nodes) {
      while (!queue.try_push(&(*node))) {
        // Spin .. this should never happen ...
      }
    }
  }

  /// Waits for all unfinished operations are finished.
  auto wait_until_finished() noexcept -> void {
    while (!all_queues_empty()) {
      execute_node_work(_thread_states[thread_id]);
    }
  }

  /// Returns the number of non-empty work queues at the present time
  auto all_queues_empty() const noexcept -> bool {
    // Try from our queue first, so other queues aren't touched if there is
    // work hereL
    for (size_t i = thread_id; i < _queues.size(); ++i) {
      if (_queues[i].size() != 0) {
        return false;
      }
    }

    for (size_t i = 0; i < thread_id; ++i) {
      if (_queues[i].size() != 0) {
        return false;
      }
    }
    return true;
  }

 private:
  /// Defines the possible states of execution for the graph executor.
  enum class ExecutionState : uint8_t {
    paused    = 0, //!< Graph is paused.
    run       = 1, //!< Graph can run.
    terminate = 2  //!< Graph must terminate
  };

  /// The max number of nodes per queue. Nodes are always being pushed and
  /// popped onto the queue, so this doesn't need to be too large.
  static constexpr size_t max_nodes_per_queue = 2048;

  /// Wrapper struct for a thread with some state.
  struct alignas(avoid_false_sharing_size) ThreadState {
    /// Defines the priority of the thread.
    enum class Priority : uint8_t {
      cpu = 0, //!< CPU thread priority.
      gpu = 1  //!< GPU thread priority
    };

    /// Defines the running state of the thread.
    enum class RunState : uint8_t {
      shutdown = 0, //!< Thread is shutdown.
      paused   = 1, //!< Thread is paused.
      running  = 2  //!< Thread is running
    };

    /// Defines the type of the variable for the thread's state/
    using state_t = std::atomic<RunState>;

    /// Constructor to create the state.
    /// \param index   The index of the thread.
    /// \param threads The total number of threads.
    /// \param state   The initial run state for the thread.
    /// \param p       The thread's priority.
    ThreadState(uint32_t index, uint32_t threads, RunState state, Priority p)
    : id{index}, max_threads{threads}, run_state{state}, priority{p} {}

    /// Move constructor to move the \p other state to this one.
    /// \param other The other state to move.
    ThreadState(ThreadState&& other) noexcept
    : id{other.id},
      max_threads{other.max_threads},
      run_state{other.run_state.load()},
      priority{other.priority} {
      other.run_state.store(RunState::shutdown, std::memory_order_relaxed);
    }

    /// Copy constructor -- deleted.
    ThreadState(const ThreadState&) = delete;

    //==--- [members] ------------------------------------------------------==//

    uint32_t id          = 0;                //!< Thread's id.
    uint32_t max_threads = 0;                //!< Max threads.
    state_t  run_state   = RunState::paused; //!< Thread's run state.
    Priority priority    = Priority::cpu;    //!< The type of the thread.

    //==--- [methods] ------------------------------------------------------==//

    /// Returns true if the thread must shutdown.
    auto must_shutdown() const noexcept -> bool {
      return run_state.load(std::memory_order_relaxed) == RunState::shutdown;
    }

    /// Returns true if the thread is paused.
    auto paused() const noexcept -> bool {
      return run_state.load(std::memory_order_relaxed) == RunState::paused;
    }
  };

  // clang-format off
  /// Defines the container type for the threads.
  using threads_t       = std::vector<std::thread>;
  /// Defines the container type for thread states.
  using thread_states_t = std::vector<ThreadState>;
  /// Defines the type of the stealer for work stealing.
  using stealer_t       = std::variant<
    random_stealer_t, neighbour_stealer_t, topo_stealer_t>;

  /// Defines the type of the work stealing queues. Here we don't need to align
  /// the node pointers to the cacheline size because the node pointers are
  /// never written, and are just used to access the cachline aligned nodes.
  /// This allows the task queues to store a lot more elements. It also means
  /// that that for threads which consume their own work, they have the next
  /// node pointers in the cache line already.
  using queue_t  = StaticStealableDequeue<Graph::node_t*, max_nodes_per_queue>;
  /// Defines the type of the container for the queue.
  using queues_t = std::vector<queue_t>;
  /// Defines the type used to notify threads to start, pause, or stop.
  using exec_t   = std::atomic<ExecutionState>;
  // clang-format on

  //==--- [construction] ---------------------------------------------------==//

  /// Constructor to create the executor with \p cpu_threads whose priority is
  /// CPU execution, and \p gpu_threads whose priority is GPU execution.
  /// \param cpu_threads Number of threads with cpu priority.
  /// \param gpu_threads Number of threads with gpu priority.
  GraphExecutor(
    size_t cpu_threads = topology().num_cores() - topology().num_gpus(),
    size_t gpu_threads = topology().num_gpus())
  : _stealer{neighbour_stealer_t()} {
    _exec_state = ExecutionState::paused;
    create_threads(cpu_threads, gpu_threads);
  }

  /// Destructor to shutdown the threads. This will wait for all work to finish.
  ~GraphExecutor() {
    wait_until_finished();

    for (auto& thread_state : _thread_states) {
      thread_state.run_state.store(
        ThreadState::RunState::shutdown, std::memory_order_relaxed);
    }

    for (auto& thread : _threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  //==--- [members] --------------------------------------------------------==//

  thread_states_t _thread_states; //!< Info for each thread.
  threads_t       _threads;       //!< Thread for the executor.
  queues_t        _queues;        //!< Queues of work.
  stealer_t       _stealer;       //!< The stealer for work stealing.
  exec_t          _exec_state;    //!< Execution state of executor.

  /// Creates \p cpu_threads whose priority is CPU execution, and \p gpu_threads
  /// whose priority is GPU execution.
  /// \param cpu_threads Number of threads with cpu priority.
  /// \param gpu_threads Number of threads with gpu priority.
  auto create_threads(uint32_t cpu_threads, uint32_t gpu_threads) -> void {
    const uint32_t total_threads = cpu_threads + gpu_threads;

    // Greating thread is always the main thread.
    thread_id = 0;
    set_affinity(thread_id);
    _queues.emplace_back();
    _thread_states.emplace_back(
      uint32_t{0},
      total_threads,
      ThreadState::RunState::running,
      ThreadState::Priority::cpu);

    for (uint32_t i = 1; i < cpu_threads + gpu_threads; ++i) {
      _queues.emplace_back();
      _thread_states.emplace_back(
        i,
        total_threads,
        ThreadState::RunState::paused,
        i < cpu_threads ? ThreadState::Priority::cpu
                        : ThreadState::Priority::gpu);
      _threads.emplace_back(
        [&](ThreadState& state) {
          using namespace std::chrono_literals;
          thread_id = state.id;
          set_affinity(thread_id);

          while (!state.must_shutdown()) {
            if (state.paused()) {
              std::this_thread::sleep_for(50us);
              continue;
            }

            execute_node_work(state);
          }
        },
        std::ref(_thread_states[i]));
    }
  }

  /// Executes works for the nodes in the queue.
  /// \param thread_state The state for the thread.
  auto execute_node_work(ThreadState& thread_state) -> void {
    // First try and get some work from our own queue:
    if (auto node = _queues[thread_state.id].pop()) {
      // If we couldn't run the node, it's dependencies haven't been met, try
      // and steal from another thread
      if (!node.value()->try_run()) {
        steal(thread_state);

        // Try run the node again, if it fails put it back on the queue.
        if (!node.value()->try_run()) {
          _queues[thread_state.id].push(*node);
        }
      }
      return;
    }

    steal(thread_state);
  }

  /// Steals from one of the other threads based on the stealer.
  /// \param thread_state The state for the thread.
  auto steal(ThreadState& thread_state) noexcept -> void {
    // Keep trying to steal until we succeed or reach the max number of
    // attempts.
    uint32_t steal_id = thread_state.id;
    for (uint32_t i = 0; i < thread_state.max_threads - 2; ++i) {
      steal_id = std::visit(
        [&thread_state, steal_id](auto&& stealer) {
          return stealer(steal_id, thread_state.max_threads);
        },
        _stealer);

      if (auto node = _queues[steal_id].steal()) {
        // Couldnt execute here, so we push this into our own queue, and then
        // return.
        // return, but we may want to add some pause type utility.
        if (node.value()->try_run()) {
          return;
        }

        // Node couldn't  run, so push it onto our queue and try and steal
        // again.
        _queues[thread_state.id].push(*node);
      }
    }
  }
};

static auto graph_executor() -> GraphExecutor& {
  static GraphExecutor executor;
  return executor;
}

} // namespace ripple

#endif // RIPPLE_GRAPH_GRAPH_EXECUTOR_HPP