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

  /// Executes a \p graph.
  /// \param graph The graph to execute.
  auto execute(Graph& graph) -> void {
    // Check if the threads need to be woken up:
    if (_exec_state != ExecutionState::run) {
      _exec_state.store(ExecutionState::run, std::memory_order_relaxed);
      for (auto& thread_state : _thread_states) {
        thread_state.run_state.store(
          ThreadState::RunState::running, std::memory_order_relaxed);
      }
    }

    auto& queue = _queues[thread_id];
    for (auto& node : graph._nodes) {
      while (!queue.try_push(&(*node))) {
        // Spin .. this should never happen ...
      }
      _total_nodes.fetch_add(1, std::memory_order_relaxed);
    }
    graph._exec_count++;
  }

  /// Executes the \p graph \p n times.
  /// \param graph The graph to execute.
  /// \param n     The number of times to execute the graph.
  auto execute_n(Graph& graph, size_t n) -> void {
    const auto last = graph.num_executions() + n;
    graph.sync([this, &graph, end = graph.num_executions() + n] {
      if (graph.num_executions() < end) {
        execute(graph);
      }
    });
    execute(graph);
  }

  /// Executes the \p graph until the \p predicate returns true.
  /// \param  graph The graph to execute.
  /// \param  pred  The predicate which returns if the execution must end.
  /// \param  args  The arguments for the predicate.
  /// \tparam Pred  The type of the predicate.
  /// \tparam Args  The type of the predicate arguments.
  template <typename Pred, typename... Args>
  auto execute_until(Graph& graph, Pred&& pred, Args&&... args) -> void {
    graph.sync(
      [this, &graph](auto&& predicate, auto&&... as) {
        if (!predicate(std::forward<Args>(as)...)) {
          execute(graph);
        }
      },
      std::forward<Pred>(pred),
      std::forward<Args>(args)...);

    if (!pred(std::forward<Args>(args)...)) {
      execute(graph);
    }
  }

  /// Waits for all unfinished operations are finished.
  auto wait_until_finished() noexcept -> void {
    while (is_unfinished_work()) {
      execute_node_work(_thread_states[thread_id]);
    }
    for (auto i = 0; i < topology().num_gpus(); ++i) {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }
  }

  /// Returns true if there is unfinished work -- that is, if there are threads
  /// executing, or if there are non-empty queues.
  auto is_unfinished_work() noexcept -> bool {
    uint32_t sum = 0;
    for (size_t i = 0; i < _thread_states.size(); ++i) {
      sum += _thread_states[i].processed_nodes();
    }
    return sum < _total_nodes.load(std::memory_order_relaxed);
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

  /// Defines the type of the counter for processed nodes.
  using node_counter_t = std::atomic<uint32_t>;

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

    uint32_t       id          = 0;                //!< Thread's id.
    uint32_t       max_threads = 0;                //!< Max threads.
    node_counter_t processed   = 0;                //!< Nodes process.
    state_t        run_state   = RunState::paused; //!< Thread's run state.
    Priority       priority    = Priority::cpu;    //!< The type of the thread.

    //==--- [methods] ------------------------------------------------------==//

    /// Returns true if the thread is executing, otherwise returns false.
    auto processed_nodes() const noexcept -> uint32_t {
      return processed.load(std::memory_order_relaxed);
    }

    /// Sets that the thread is executing.
    auto inc_processed_nodes() noexcept -> void {
      processed.fetch_add(1, std::memory_order_relaxed);
    }

    /// Returns true if the thread must shutdown.
    auto shutdown() noexcept -> void {
      run_state.store(RunState::shutdown, std::memory_order_relaxed);
    }

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

  thread_states_t _thread_states;   //!< Info for each thread.
  threads_t       _threads;         //!< Thread for the executor.
  queues_t        _queues;          //!< Queues of work.
  stealer_t       _stealer;         //!< The stealer for work stealing.
  exec_t          _exec_state;      //!< Execution state of executor.
  node_counter_t  _total_nodes = 0; //!< Total number of nodes to execute.

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
      ThreadState::RunState::paused,
      ThreadState::Priority::cpu);

    for (uint32_t i = 1; i < cpu_threads + gpu_threads; ++i) {
      _queues.emplace_back();
      _thread_states.emplace_back(
        i,
        total_threads,
        ThreadState::RunState::running,
        i < cpu_threads ? ThreadState::Priority::cpu
                        : ThreadState::Priority::gpu);
      _threads.emplace_back(
        [&](ThreadState& state) {
          using namespace std::chrono_literals;
          thread_id = state.id;
          set_affinity(state.id);

          while (!state.must_shutdown()) {
            // if (state.paused()) {
            //  std::this_thread::sleep_for(500ns);
            //  continue;
            //}

            // Try and do some work. If we did some work, then there is likely
            // some more work to do, otherwise we couldn't find any work to
            // execute, so we aren't executing.
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
          return;
        }
      }
      thread_state.inc_processed_nodes();
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
        // If we stole a node, try and run it:
        if (node.value()->try_run()) {
          thread_state.inc_processed_nodes();
          return;
        }

        // Node couldn't  run, so push it onto our queue.
        _queues[thread_state.id].push(*node);
      }
    }
  }
};

/// Returns a reference to the graph executor.
static auto graph_executor() -> GraphExecutor& {
  static GraphExecutor executor;
  return executor;
}

} // namespace ripple

#endif // RIPPLE_GRAPH_GRAPH_EXECUTOR_HPP