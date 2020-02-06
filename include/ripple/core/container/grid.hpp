//==--- ripple/core/container/grid.hpp -------------------------- -*- C++ -*- ---==//
//            
//                                  Ripple
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  grid.hpp
/// \brief This file defines a class which represents a grid.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_CONTAINER_GRID_HPP
#define RIPPLE_CONTAINER_GRID_HPP

#include "block_traits.hpp"
#include "grid_traits.hpp"
#include "device_block.hpp"
#include "host_block.hpp"
#include <ripple/core/arch/topology.hpp>
#include <ripple/core/functional/pipeline.hpp>
#include <ripple/core/multidim/dynamic_multidim_space.hpp>
#include <ripple/core/utility/dim.hpp>
#include <thread>

namespace ripple {

/// Implementation of the grid class.
/// \tparam T          The type of the data for the grid.
/// \tparam Dimensions The number of dimensions for the grid.
template <typename T, size_t Dimensions>
class Grid {
  /// Defines the type of the host block for the grid.
  using host_block_t   = HostBlock<T, Dimensions>;
  /// Defines the type of the device block for the grid.
  using device_block_t = DeviceBlock<T, Dimensions>;
  /// Defines the type of the space used by the grid.
  using space_t        = DynamicMultidimSpace<Dimensions>;
  /// Defines the type of the iterator over host data.
  using host_iter_t    = typename BlockTraits<host_block_t>::iter_t;
  /// Defines the type of the block for the grid.
  using block_t        = Block<T, Dimensions>;
  /// Defines the type of the state for the block.
  using block_state_t  = typename block_t::State;
  /// Defines the type of the container used for the blocks.
  using blocks_t       = HostBlock<block_t, Dimensions>;

  /// Defines a dimension overload for the grid.
  static constexpr auto dims_v          = Dimension<Dimensions - 1>();
  /// Defines the default elements in the x dimension.
  static constexpr auto block_size_x    = size_t{1024};
  /// Defines the number of streams per device.
  static constexpr auto streams_per_gpu = size_t{8};

 public:
  //==--- [construction] ---------------------------------------------------==//

  /// Initializes the size of each of the dimensions of the grid. This
  /// constructor will create blocks with the largest size such that the minimum
  /// number of blocks are used which fit into the device memory.
  ///
  /// \param  topo  A reference to the system topology.
  /// \param  sizes The sizes of the dimensions for the grid.
  /// \tparam Sizes The types of the dimension sizes.
  template <
    typename... Sizes,      
    all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0
  >
  Grid(Topology& topo, Sizes&&... sizes) 
  : _space{std::forward<Sizes>(sizes)...}, _topo{topo} {
    resize_block_size(dims_v);
    default_allocation();
  }

  /// Initializes the size of each of the dimensions of the grid, allocating the
  /// memory for the grid, with padding data for the boundary of the grid. This
  /// constructor is only enabled when the number of arguments matches the
  /// dimensionality of the grid and the \p sizes are numeric types.
  ///
  /// \param  topo    A reference to the system topology.
  /// \param  padding The amount of padding for the grid.
  /// \param  sizes   The sizes of the dimensions for the grid.
  /// \tparam Sizes   The types of other dimension sizes.
  template <
    typename... Sizes,      
    all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0
  >
  Grid(Topology& topo, size_t padding, Sizes&&... sizes) 
  : _space{padding, std::forward<Sizes>(sizes)...}, _topo{topo} {
    resize_block_size(dims_v); 
    default_allocation();
  }

  /// Destructor -- resets the amount of device memory allocated for the blocks
  /// on each of the gpus.
  ~Grid() {
    invoke(_blocks, [&] (auto b) {
      if (b->gpu_id >= 0) {
        _topo.gpus[b->gpu_id].mem_alloc -= b->device_data.mem_requirement();
      }
    });
  }

  //==--- [access] ---------------------------------------------------------==//
  
  /// Overload of operator() to get an iterator to the element at the location
  /// specified by the \p is indices. This iterator is only valid within the
  /// block to which the \p is indices belong.
  ///
  /// If this is called on the host, then this will copy the data for the block
  /// to the host first, and then return an iterator over the block data on the
  /// host. If a copy is performed, this will block until the copy from the
  /// device is finished.
  ///
  /// This access is provided for convenience, and should only be used in
  /// non-critical code paths, because there is a non-significant amount of
  /// overhead required to get the element at the exact index from the grid
  /// structure.
  ///
  /// Operations should be performed on the entire grid, or on blocks within the
  /// grid.
  ///
  /// \param  is      The indices to get the element at.
  /// \tparam Indices The types of the indices.
  template <typename... Indices>
  ripple_host_only auto operator()(Indices&&... is) -> host_iter_t {
    auto block = _blocks.begin();
    auto indices = std::array<size_t, Dimensions>{static_cast<size_t>(is)...};
    auto offsets = std::array<size_t, Dimensions>{0};
    // Offset to the correct block:
    unrolled_for<Dimensions>([&] (auto d) {
      constexpr auto dim = static_cast<size_t>(d);
      // Automatic round down to get the block index:
      offsets[dim] = indices[dim] / _block_size.size(dim);
      block.shift(dim, offsets[dim]);

      // Compute the offset in the block:
      offsets[dim] = indices[dim] - offsets[dim] * _block_size.size(dim);
    });

    // Copy data back from the device, if necessary.
    block->ensure_host_data_available();
    
    // Create and return the iterator ...
    auto iter = block->host_data.begin();
    unrolled_for<Dimensions>([&] (auto d) {
      constexpr auto dim = static_cast<size_t>(d);
      iter.shift(dim, offsets[dim]);
    });
    return iter;
  }

  //==--- [interface] ------------------------------------------------------==//

  /// Gets a const iterator to the first block in the grid.
  auto begin() const {
    return _blocks.begin();
  }

  /// Gets an const iterator to the first block in the grid.
  auto begin() {
    return _blocks.begin();
  }
  
  /// Returns the total number of elements in the grid.
  auto size() const -> std::size_t {
    return _space.internal_size();
  } 

  /// Returns the total number of elements in dimension \p dim.
  /// \param dim The dimension to get the size of.
  /// \param Dim The type of the dimension specifier.
  template <typename Dim>
  auto size(Dim&& dim) const -> std::size_t {
    return _space.internal_size(std::forward<Dim>(dim));
  }

  //==--- [pipeline invoke] ------------------------------------------------==//
 
  template <typename... Ops>
  auto invoke(Pipeline<Ops...>& pipeline) -> void {
    // First version:
    //    This is single threaded and simply invokes each block on its assigned
    //    GPU.
    invoke(_blocks, [&] (auto block_it) {
      block_it->ensure_device_data_available();

      invoke(block_it->device_data, pipeline);
    });
  }

/* 
  template <typename... Ops>
  auto invoke(Pipeline<Ops...>& pipeline) {
    std::vector<std::thread> runners();
    for (size_t i : range(_topo.num_gpus())) {
      runners.push_back([] (size_t thread_id) {
        const auto start = thread_id * blocks_per_gpu_x;
        const auto end   = start + blocks_per_gpu_x;
        invoke(_blocks, sub_range_t<dimx_t>{start, end}, [] (auto block_it) {
          auto pipeline = make_pipeline(
            // Start by moving the data and the padding to the device:
            [&] (auto it) {
              it->ensure_device_data_available();
              it->ensure_device_padding_available();
            },
            // Next, invoke the pipeline on the device block:
            [&] (auto it) {
              it->data_state = Block::State::not_updated_device;
              it->padding_state = Block::State::not_updated_device;
              invoke(it->device_data, pipeline);
            },
            // Set the state of the data for the block:
            [&] (auto it) {
              it->data_state = Block::State::updated_device;
              it->padding_state = Block::State::updated_device;
            }
          );
        });
      }, i);
    }
  };
 */
 private:
  space_t    _space;        //!< The space for the grid.
  space_t    _block_size;   //!< The size of the blocks in the grid.
  blocks_t   _blocks;       //!< Blocks for the grid.
  Topology&  _topo;         //!< Topology for the system.

  //==--- [resize] ---------------------------------------------------------==//

  /// Defines the size of the default blocks for 1D. The default is to split
  /// the x dimension across GPUs.
  auto resize_block_size(dimx_t) -> void {
    _block_size = DynamicMultidimSpace<1>{
      std::min(block_size_x, _space.internal_size(dim_x) / _topo.num_gpus())
    };
  }

  /// Defines the size of the default blocks for 2D. The default is to split
  /// the x dimension across GPUs. 
  auto resize_block_size(dimy_t) -> void {
    _block_size = DynamicMultidimSpace<2>{
      std::min(block_size_x, _space.internal_size(dim_x) / _topo.num_gpus()),
      _space.internal_size(dim_y)
    };
  }

  /// Defines the size of the default blocks for 2D. The default is to split
  /// the x dimension across GPUs. 
  auto resize_block_size(dimz_t) -> void {
    _block_size = DynamicMultidimSpace<3>{
      std::min(block_size_x, _space.internal_size(dim_x) / _topo.num_gpus()),
      _space.internal_size(dim_y),
      _space.internal_size(dim_z)
    };
  }

  //==--- [allocation] -----------------------------------------------------==//

  /// Allocates the blocks based on the defined block size.
  auto allocate_blocks() -> void {
    unrolled_for<Dimensions>([&] (auto dim) {
      const auto blocks_for_dim = 
        static_cast<size_t>(std::ceil(
            static_cast<float>(_space.internal_size(dim)) / 
            _block_size.internal_size(dim)
        ));
      _blocks.resize_dim(dim, blocks_for_dim);
    });
    _blocks.reallocate();
  }

  /// Allocates the actual data for the blocks. This requires that the blocks
  /// themselves be allocated so that there are blocks to allocate the data for.
  /// The current strategy for block allocation is the following:
  ///
  ///   - Compute the GPU index in the ideal case (that the data is split along
  ///   the x dimension for all the GPUs).
  ///   - Check if the GPU has enough memory remaining to allocate the block,
  ///   and if so, then allocate memory for the block.
  ///   - If the gpu doesn't have enough memory, then check if the next on does,
  ///   and then allocate the block there.
  ///   - If the next GPU doesn't have enough memory, keep looking through the
  ///   rest of the GPUs.
  ///   - If no GPU has any memory remaining, then use an already allocated
  ///   block's memory to share.
  auto allocate_data_for_blocks() -> void {
    std::vector<size_t> stream_ids(_topo.num_gpus());
    for (auto& stream_id : stream_ids) { stream_id = 0; }

    // Currently this assumes data is split along x dimension.
    const auto blocks_per_gpu_x = _blocks.size(dim_x) / _topo.num_gpus();
    invoke(_blocks, [&] (auto b) {

      // Set the host component of the block to enable asynchronous operations
      // so that we can overlap compute and transfer:
      b->host_data.set_op_kind(BlockOpKind::asynchronous);

      // Now set the padding for the block:
      b->host_data.set_padding(_space.padding());
      b->device_data.set_padding(_space.padding());

      // Resize each of the dimensions of the block, and set the block index.
      unrolled_for<Dimensions>([&] (auto dim) {
        b->indices[dim] = global_idx(dim);
        b->host_data.resize_dim(dim, _block_size.internal_size(dim));
        b->device_data.resize_dim(dim, _block_size.internal_size(dim));
      });

      // Allocate the host memory:
      b->host_data.reallocate();

      // Check if there is enough memory on the device to allocate:
      auto       gpu_id  = global_idx(dim_x) / blocks_per_gpu_x;
      size_t     tries   = 0;
      const auto mem_req = b->device_data.mem_requirement();
      while (mem_req >= _topo.gpus[gpu_id].mem_remaining() &&
             tries < _topo.num_gpus()) {
        gpu_id = (gpu_id + 1) % _topo.num_gpus();
        tries++;
      }

      // Here a block needs to be found to share the data with, but for now,
      // this functionality is not implemented.
      if (tries == _topo.num_gpus()) {
        assert(false);
      }
      b->gpu_id = gpu_id;

      // We can allocate on the device ...
      cudaSetDevice(gpu_id);
      auto& gpu       = _topo.gpus[gpu_id];
      auto& stream_id = stream_ids[gpu_id];
      b->device_data.set_stream(&gpu.streams[stream_id]);
      b->device_data.reallocate();
      b->data_data   = state_t::updated_device;
      gpu.mem_alloc += mem_req; 
      stream_id      = (stream_id + 1) % gpu.streams.size();
    });
  }

  /// Default allocation of the block data for the grid to create the blocks for
  /// the given block size, and allocate memory for the blocks on the
  /// appropriate devices.
  auto default_allocation() -> void {
    allocate_blocks();
    allocate_data_for_blocks();
  }
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_GRID_HPP

