//==--- ripple/container/grid.hpp -------------------------- -*- C++ -*- ---==//
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

#include "grid_traits.hpp"
#include "device_block.hpp"
#include "host_block.hpp"
#include <ripple/arch/topology.hpp>
#include <ripple/multidim/dynamic_multidim_space.hpp>
#include <ripple/utility/dim.hpp>
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

  /// Defines a type which wraps a host and device block into a single type, and
  /// which has a state for the different data components in the block.
  struct Block {
    /// Defines the type of the type for the block index.
    using index_t = size_t[Dimensions];

    /// Defines the state of the block data.
    enum class State : uint8_t {
      updated_host       = 0,   //!< Data is on the host and is updated.
      not_updated_host   = 1,   //!< Data is on the host but is not updated.
      updated_device     = 2,   //!< Data is on the device and is updated.
      not_updated_device = 3    //!< Data is on the device but is not updated.
    };

    host_block_t   host_data;   //!< Host block data.
    device_block_t device_data; //!< Device block data.
    index_t        indices;     //!< Indices of the block in the grid.
    Block*         sibling       = nullptr;                 //!< Data sharer.
    int            gpu_id        = -1;                      //!< Device index.
    State          data_state    = State::not_updated_host; //!< Data state.
    State          padding_state = State::not_updated_host; //!< Padding state.
  };

  /// Defines the type of the container used for the blocks.
  using blocks_t = HostBlock<Block, Dimensions>;

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

  //==--- [interface] ------------------------------------------------------==//
  
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

