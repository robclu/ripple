//==--- ripple/core/graph/executor.hpp --------------------- -*- C++ -*- ---==//
//
//                                 Ripple
//
//                      Copyright (c) 2019, 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//

/**=--- ripple/execution/executor.hpp ---------------------- -*- C++ -*- ---==**
 *
 *                                  Ripple
 *
 *                      Copyright (c) 2019 - 2021 Rob Clucas.
 *
 *  This file is distributed under the MIT License. See LICENSE for details.
 *
 *==-------------------------------------------------------------------------==*
 *
 * \file  executor.hpp
 * \brief This file implements an executor class.
 *
 *==------------------------------------------------------------------------==*/

#ifndef RIPPLE_EXECUTION_EXECUTOR_HPP
#define RIPPLE_EXECUTION_EXECUTOR_HPP

#include <ripple/graph/graph.hpp>
#include <ripple/graph/stealer.hpp>
#include <ripple/arch/topology.hpp>
#include <ripple/container/static_stealable_dequeue.hpp>
#include <chrono>
#include <thread>
#include <variant>

namespace ripple {

/*==-- [forward declarations] ----------------------------------------------==*/

/**
 * Gets a reference to the executor.
 * \return A reference to the executor.
 */
static auto executor() -> Executor&;

/**
 * Executes a graph.
 * \param graph The graph to execute.
 */
inline auto execute(Graph& graph) noexcept -> void;

/**
 * Executes a graph until the preduces returns false.
 *
 * \param  graph The graph to execute.
 * \param  pred  The predicate which returns true if the graph must execute.
 * \param  args  Arguments for the predicate.
 * \tparam Pred  The type of the predicate.
 * \tparam Args  The type of the arguments.
 */
template <typename Pred, typename... Args>
inline auto
execute_until(Graph& graph, Pred&& pred, Args&&... args) noexcept -> void;

/**
 * Creates a fence, synchronizing all execution on all accelerators at the
 * fence.
 */
inline auto fence() noexcept -> void;

/**
 * Creates a barrier, waiting for all pending operaitons on all host cores
 * and accelerators.
 */
inline auto barrier() noexcept -> void;

/** Index of a thread for the executor. */
static thread_local uint32_t thread_id = 0;

/**
 * The Executor class maintains a pool of threads, which maintain a queue of
 * nodes which can be executed by the workers. Each node defines work to be
 * executed and connecetivity information which defines its dependencies between
 * other nodes.
 */
class Executor {
 public:
  /** Friend function to access the global graph executor. */
  friend auto executor() -> Executor&;

  /**
   * Sets the stealing policy for the executor.
   *
   * \sa StealPolicy
   *
   * \param policy The policy to use for stealing nodes.
   */
  auto set_steal_policy(StealPolicy policy) noexcept -> void {
    /*
       // clang-format off
       switch (policy) {
         case StealPolicy::random     : stealer_ = RandomStealer()   ; return;
         case StealPolicy::neighbour  : stealer_ = NeighbourStealer(); return;
         case StealPolicy::topological: stealer_ = TopoStealer()     ; return;
         default: return;
       }
       // clang-format on
   */
  }

  /**
   * Executes a graph. This emplaced all nodes using this thread while other
   * threads start stealing.
   * \param graph The graph to execute.
   */
  auto execute(Graph& graph) noexcept -> void {
    // Check if the threads need to be woken up:
    if (exec_state_ != ExecutionState::run) {
      exec_state_.store(ExecutionState::run, std::memory_order_relaxed);
      for (auto& state : thread_states_) {
        state.run_state.store(
          ThreadState::RunState::running, std::memory_order_relaxed);
      }
    }

    auto& cpu_queue = cpu_queues_[thread_id];
    auto& gpu_queue = gpu_queues_[thread_id];
    for (auto& node : graph.nodes_) {
      auto& queue = node->execution_kind() == ExecutionKind::gpu ? gpu_queue
                                                                 : cpu_queue;
      while (!queue.try_push(&(*node))) {
        // Spin .. this should never happen ...
        printf("Error\n");
      }
      total_nodes_.fetch_add(1, std::memory_order_relaxed);
    }
    graph.exec_count_++;
  }

  /**
   * Executes the graph n times.
   * \param graph The graph to execute.
   * \param n     The number of times to execute the graph.
   */
  auto execute_n(Graph& graph, size_t n) noexcept -> void {
    const auto last = graph.num_executions() + n;
    graph.sync([this, &graph, end = graph.num_executions() + n] {
      if (graph.num_executions() < end) {
        execute(graph);
      }
    });
    execute(graph);
  }

  /**
   * Executes the  graph until the predicate returns false.
   * \param  graph The graph to execute.
   * \param  pred  The predicate which returns if the execution must end.
   * \param  args  The arguments for the predicate.
   * \tparam Pred  The type of the predicate.
   * \tparam Args  The type of the predicate arguments.
   */
  template <typename Pred, typename... Args>
  auto
  execute_until(Graph& graph, Pred&& pred, Args&&... args) noexcept -> void {
    graph.sync(
      [this, &graph](auto&& predicate, auto&&... as) {
        if (predicate(ripple_forward(as)...)) {
          execute(graph);
        }
      },
      ripple_forward(pred),
      ripple_forward(args)...);
    if (pred(ripple_forward(args)...)) {
      execute(graph);
    }
  }

  /**
   * Creates a fence, waiting for all threads to finish execution, synchronizing
   * the gpu work.
   */
  auto barrier() noexcept -> void {
    size_t gpu_id = 0;
    for (const auto& thread_state : thread_states_) {
      if (gpu_id >= topology().num_gpus()) {
        break;
      }
      if (thread_state.has_gpu_priority()) {
        topology().gpus[gpu_id++].execute_barrier();
      }
    }
    auto& state = thread_states_[thread_id];
    while (has_unfinished_work()) {
      execute_node_work(state, state.first_priority());
      execute_node_work(state, state.second_priority());
    }
  }

  /**
   * Creates a fence, waiting for all threads to finish execution, synchronizing
   * the gpu work.
   */
  auto fence() noexcept -> void {
    size_t gpu_id = 0;
    for (const auto& thread_state : thread_states_) {
      if (gpu_id >= topology().num_gpus()) {
        break;
      }
      if (thread_state.has_gpu_priority()) {
        topology().gpus[gpu_id++].synchronize_streams();
      }
    }
    auto& state = thread_states_[thread_id];
    while (has_unfinished_work()) {
      execute_node_work(state, state.first_priority());
      execute_node_work(state, state.second_priority());
    }
  }

  /**
   * Gets the number of nodes which have executed.
   */
  auto num_executed_nodes() const noexcept -> uint32_t {
    uint32_t sum = 0;
    for (size_t i = 0; i < thread_states_.size(); ++i) {
      sum += thread_states_[i].processed_nodes();
    }
    return sum;
  }

  /**
   * Determines if there is unfinished work to be executed.
   * \return true if there is unfinished work.
   */
  auto has_unfinished_work() noexcept -> bool {
    return num_executed_nodes() < total_nodes_.load(std::memory_order_relaxed);
  }

  /**
   * Sets the number of active gpu threads.
   * \param cpu_threads The number of threads with cpu priority.
   * \param gpu_threads The number of threads with gpu priority.
   */
  auto set_active_threads(int cpu_threads, int gpu_threads) noexcept -> void {
    int cpu_count = 0, gpu_count = 0;
    for (auto& state : thread_states_) {
      gpu_count += state.has_gpu_priority() && state.is_running() ? 1 : 0;
      cpu_count += state.has_cpu_priority() && state.is_running() ? 1 : 0;
    }

    int gpu_change = gpu_threads - gpu_count;
    int cpu_change = cpu_threads - cpu_count;
    gpu_count = 0, cpu_count = 0;

    auto modify = [](auto& state, auto& count, const auto& change) {
      if (state.is_running() && change < 0) {
        state.pause();
        count++;
      } else if (state.is_paused() && change > 0) {
        state.run();
        count++;
      }
    };

    for (auto& state : thread_states_) {
      state.max_gpu_threads = gpu_threads;
      state.max_cpu_threads = cpu_threads;
      if (state.has_gpu_priority() && gpu_count < std::abs(gpu_change)) {
        modify(state, gpu_count, gpu_change);
        state.priority = ExecutionKind::gpu;
        continue;
      }
      if (state.has_cpu_priority() && cpu_count < std::abs(cpu_change)) {
        modify(state, cpu_count, cpu_change);
        state.priority = ExecutionKind::cpu;
      }
    }
  }

 private:
  /** Defines the possible states of execution for the executor. */
  enum class ExecutionState : uint8_t {
    paused    = 0, //!< Graph is paused.
    run       = 1, //!< Graph can run.
    terminate = 2  //!< Graph must terminate
  };

  /**
   * The max number of nodes per queue. Nodes are always being pushed and
   * popped onto the queue, so this doesn't need to be too large.
   */
  static constexpr size_t max_nodes_per_queue = 4096;

  /** Defines the type of the counter for processed nodes. */
  using NodeCounter = std::atomic<uint32_t>;

  /**
   * Defines the state for threads in the executor.
   */
  struct alignas(avoid_false_sharing_size) ThreadState {
    /** Defines the running state of the thread. */
    enum class RunState : uint8_t {
      shutdown = 0, //!< Thread is shutdown.
      paused   = 1, //!< Thread is paused.
      running  = 2  //!< Thread is running
    };

    /**
     * Defines the type of the thread safe variable for the thread run state.
     */
    using SafeRunState = std::atomic<RunState>;

    uint32_t      id              = 0;                  //!< Thread's id.
    uint32_t      max_cpu_threads = 0;                  //!< Max cpu threads.
    uint32_t      max_gpu_threads = 0;                  //!< Max gpu threads.
    NodeCounter   processed       = 0;                  //!< Nodes process.
    SafeRunState  run_state       = RunState::paused;   //!< Thread's run state.
    ExecutionKind priority        = ExecutionKind::gpu; //!< Priority.

    /*==--- [construction] -------------------------------------------------==*/

    /**
     * Constructor to create the state.
     * \param index       The index of the thread.
     * \param cpu_threads The total number of cpu threads.
     * \param gpu_thread  The totla number of gpu threads.
     * \param state       The initial run state for the thread.
     * \param p           The threads priority.
     */
    ThreadState(
      uint32_t      index,
      uint32_t      cpu_threads,
      uint32_t      gpu_threads,
      RunState      state,
      ExecutionKind p) noexcept
    : id{index},
      max_cpu_threads{cpu_threads},
      max_gpu_threads{gpu_threads},
      run_state{state},
      priority{p} {}

    /**
     * Move constructor to move the other state to this one.
     * \param other The other state to move.
     */
    ThreadState(ThreadState&& other) noexcept
    : id{other.id},
      max_cpu_threads{other.max_cpu_threads},
      max_gpu_threads{other.max_gpu_threads},
      run_state{other.run_state.load()},
      priority{other.priority} {
      other.run_state.store(RunState::paused, std::memory_order_relaxed);
    }

    /** Copy constructor -- deleted. */
    ThreadState(const ThreadState&) = delete;

    /*==--- [methods] ------------------------------------------------------==*/

    /**
     * Gets the number of nodes processed by the threads.
     * \return The number of threads processed by the thread.
     */
    auto processed_nodes() const noexcept -> uint32_t {
      return processed.load(std::memory_order_relaxed);
    }

    /**
     * Increments the number of processed nodes by one.
     */
    auto inc_processed_nodes() noexcept -> void {
      processed.fetch_add(1, std::memory_order_relaxed);
    }

    /**
     * Gets the max number of threads of the given execution kind.
     * \param kind The kind of the execution.
     * \return The max number of threads for the given execution kind.
     */
    auto max_threads(ExecutionKind kind) noexcept -> uint32_t {
      return kind == ExecutionKind::gpu ? max_gpu_threads : max_cpu_threads;
    }

    /**
     * Determines if the thread has cpu priority.
     * \return true if the thread has cpu priority.
     */
    auto has_cpu_priority() const noexcept -> bool {
      return priority == ExecutionKind::cpu;
    }

    /**
     * Determines if the thread has gpu priority.
     * \return true if the thread has gpu priority.
     */
    auto has_gpu_priority() const noexcept -> bool {
      return priority == ExecutionKind::gpu;
    }

    /**
     * Gets the first priority type for the thread.
     * \return The first priority execution kind for the thread.
     */
    auto first_priority() const noexcept -> ExecutionKind {
      return priority;
    }

    /**
     * Gets the second priority type for the thread.
     * \return The second priortity execution kind for the thread.
     */
    auto second_priority() const noexcept -> ExecutionKind {
      return ExecutionKind::cpu;
    }

    /**
     * Determines if the thread is shutdown.
     * \return true if the thread is shutdown.
     */
    auto is_shutdown() const noexcept -> bool {
      return run_state.load(std::memory_order_relaxed) == RunState::shutdown;
    }

    /**
     * Determines if the thread is paused.
     * \return true if the thread is paused.
     */
    auto is_paused() const noexcept -> bool {
      return run_state.load(std::memory_order_relaxed) == RunState::paused;
    }

    /**
     * Determines if the thread is running.
     * \return true if the thread is running,
     */
    auto is_running() const noexcept -> bool {
      return run_state.load(std::memory_order_relaxed) == RunState::running;
    }

    /** Shuts down the thread. */
    auto shutdown() noexcept -> void {
      run_state.store(RunState::shutdown, std::memory_order_relaxed);
    }

    /** Sets the thread state to paused. */
    auto pause() noexcept -> void {
      return run_state.store(RunState::paused, std::memory_order_relaxed);
    }

    /** Sets the thread state to run. */
    auto run() noexcept -> void {
      return run_state.store(RunState::running, std::memory_order_relaxed);
    }
  };

  // clang-format off
  /** Defines the container type for the threads. */
  using Threads      = std::vector<std::thread>;
  /** Defines the container type for thread states. */
  using ThreadStates = std::vector<ThreadState>;
  /** Defines the type of the stealer for work stealing. */
  using Stealer      = NeighbourStealer;
//  std::variant<
//    RandomStealer, NeighbourStealer, TopoStealer>;

  /**
   * Defines the type of the work stealing queues. Here we don't need to align
   * the node pointers to the cacheline size because the node pointers are
   * never written, and are just used to access the cachline aligned nodes.
   * This allows the task queues to store a lot more elements. It also means
   * that for threads which consume their own work, they have the successor node
   * pointers in the cache line already.
   */
  using Queue     = 
    StaticStealableDequeue<Graph::NodeType*, max_nodes_per_queue>;
  /** Defines the type of the container for the queues. */
  using Queues    = std::vector<Queue>;
  /** Defines the type used to notify threads to start, pause, or stop. */
  using ExecState = std::atomic<ExecutionState>;
  // clang-format on

  /*==--- [construction] ---------------------------------------------------==*/

  /**
   * Constructor to create the executor with a number of threads whose priority
   * is  CPU execution, and a number of threads whose priority is GPU execution.
   * \param cpu_threads Number of threads with cpu priority.
   * \param gpu_threads Number of threads with gpu priority.
   */
  Executor(size_t cpu_threads, size_t gpu_threads)
  : stealer_{NeighbourStealer{}}, exec_state_{ExecutionState::run} {
    create_threads(cpu_threads, gpu_threads);
  }

  /**
   * Destructor which shuts down the threads. This will wait for all work to
   * finish before synchronizing the threads.
   */
  ~Executor() {
    fence();

    for (auto& thread_state : thread_states_) {
      thread_state.shutdown();
    }

    for (auto& thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  //==--- [members] --------------------------------------------------------==//

  ThreadStates thread_states_;                     //!< Thread states.
  Threads      threads_;                           //!< Worker threads.
  Queues       cpu_queues_;                        //!< CPU queues.
  Queues       gpu_queues_;                        //!< GPU queues.
  Stealer      stealer_;                           //!< Work stealer.
  ExecState    exec_state_  = ExecutionState::run; //!< Execution state.
  NodeCounter  total_nodes_ = 0;                   //!< Node count.

  /**
   * Creates the CPU threads whose priority is CPU execution, and GPU threads
   * whose priority is GPU execution.
   * \param cpu_threads Number of threads with cpu priority.
   * \param gpu_threads Number of threads with gpu priority.
   */
  auto create_threads(uint32_t cpu_threads, uint32_t gpu_threads) -> void {
    const uint32_t total_threads = cpu_threads + gpu_threads;

    // Creating thread is always the main thread and has GPU priority so that
    // it can push work to both queue types.
    thread_id = 0;
    set_affinity(thread_id);
    gpu_queues_.emplace_back();
    cpu_queues_.emplace_back();
    thread_states_.emplace_back(
      uint32_t{0},
      total_threads,
      gpu_threads + 1,
      ThreadState::RunState::paused,
      ExecutionKind::gpu);

    for (uint32_t i = 1; i < total_threads; ++i) {
      if (i <= gpu_threads) {
        gpu_queues_.emplace_back();
      }
      cpu_queues_.emplace_back();
      thread_states_.emplace_back(
        i,
        total_threads,
        gpu_threads + 1,
        ThreadState::RunState::running,
        i <= gpu_threads ? ExecutionKind::gpu : ExecutionKind::cpu);
    }
    for (uint32_t i = 1; i < total_threads; ++i) {
      threads_.emplace_back(
        [&](ThreadState& state) {
          using namespace std::chrono_literals;
          thread_id = state.id;
          set_affinity(state.id);
          constexpr uint32_t sleep_fail_count = 100;

          uint32_t fails = 0;
          while (!state.is_shutdown()) {
            if (state.is_paused()) {
              std::this_thread::sleep_for(100us);
              continue;
            }

            // Try and do some work. If we did some work, then there is likely
            // some more work to do, otherwise we couldn't find any work to
            // execute, so we aren't executing.
            if (!execute_node_work(state, state.first_priority())) {
              if (!execute_node_work(state, state.second_priority())) {
                if (++fails != sleep_fail_count) {
                  continue;
                }
                // state.pause();
                // std::this_thread::sleep_for(5us);
              }
            }
            fails = 0;
          }
        },
        std::ref(thread_states_[i]));
    }
  }

  /**
   * Executes works for the nodes in the queue.
   * \param thread_state The state for the thread.
   */
  auto execute_node_work(ThreadState& thread_state, ExecutionKind kind) noexcept
    -> bool {
    auto& queue = get_queue(thread_state.id, kind);
    if (auto node = queue.pop()) {
      if (node.value()->try_run()) {
        thread_state.inc_processed_nodes();
        return true;
      }
      queue.push(*node);
    }

    // Steal if we are empty:
    return steal(thread_state, kind);
  }

  /// Steals from one of the other threads based on the stealer.
  /// \param thread_state The state for the thread.
  auto steal(ThreadState& thread_state, ExecutionKind kind) noexcept -> bool {
    // Keep trying to steal until we succeed or reach the max number of
    // attempts.
    uint32_t       steal_id     = thread_state.id;
    const uint32_t max_threads  = thread_state.max_threads(kind);
    const uint32_t max_id       = std::max(max_threads, uint32_t{1});
    const uint32_t max_attempts = std::max(max_id, uint32_t{5});
    for (uint32_t i = 0; i < max_attempts; ++i) {
      steal_id = stealer_(steal_id, max_id);
      //      std::visit(
      //        [&thread_state, steal_id, max_id](auto&& stealer) {
      //          return stealer(steal_id, max_id);
      //        },
      //        stealer_);

      if (steal_id == thread_state.id) {
        continue;
      }

      if (auto node = get_queue(steal_id, kind).steal()) {
        // If we stole a node, try and run it:
        if (node.value()->try_run()) {
          thread_state.inc_processed_nodes();
          return true;
        }
        // Node couldn't  run, so push it onto our queue.
        get_queue(thread_state.id, kind).push(*node);
      }
    }
    return false;
  }

  /**
   * Gets the queue with a given index and the given type.
   * \param thread_id The index of the thread to get the queue for.
   * \param kind      The kind of the queue.
   * \return A reference to the queue.
   */
  auto get_queue(size_t id, ExecutionKind kind) noexcept -> Queue& {
    if (kind == ExecutionKind::cpu) {
      return cpu_queues_[id];
    }
    return gpu_queues_[id];
  }
};

/**
 * Gets a reference to the graph executor.
 * \return A reference to the global executor.
 */
inline auto executor() -> Executor& {
  static Executor exec(13, 3);
  // topology().num_cores() - topology().num_gpus(), topology().num_gpus());

  return exec;
}

/**
 * Executes the given graph.
 * \param graph The graph to execute.
 */
inline auto execute(Graph& graph) noexcept -> void {
  executor().execute(graph);
}

/**
 * Executes the graph until the predicate returns false.
 * \param  graph The graph to execute.
 * \param  pred  The predicate which returns true if the graph must execute.
 * \param  args  Arguments for the predicate.
 * \tparam Pred  The type of the predicate.
 * \tparam Args  The type of the arguments.
 */
template <typename Pred, typename... Args>
inline auto
execute_until(Graph& graph, Pred&& pred, Args&&... args) noexcept -> void {
  executor().execute_until(
    graph, ripple_forward(pred), ripple_forward(args)...);
}

/**
 * Creates a fence, flushing all pending operations on all accelerators.
 */
inline auto fence() noexcept -> void {
  executor().fence();
}

/**
 * Creates a barrier, waiting for all operations on all devices.
 */
inline auto barrier() noexcept -> void {
  executor().barrier();
}

} // namespace ripple

#endif // RIPPLE_EXECUTION_EXECUTOR_HPP