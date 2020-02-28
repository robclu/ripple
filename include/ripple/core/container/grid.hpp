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
#include "block.hpp"
#include <ripple/core/arch/topology.hpp>
#include <ripple/core/functional/invoke.hpp>
#include <ripple/core/multidim/dynamic_multidim_space.hpp>
#include <ripple/core/utility/dim.hpp>
#include <bitset>
#include <cassert>
#include <cmath>
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
  /// Defines the type of the block for the grid.
  using block_t        = Block<T, Dimensions>;
  /// Defines the type of the state for the block.
  using block_state_t  = typename block_t::State;
  /// Defines the type of the container used for the blocks.
  using blocks_t       = std::vector<block_t>;

  /// Defines a dimension overload for the grid.
  static constexpr auto dims_v           = Dimension<Dimensions - 1>();
  /// Defines the default elements in a block when a dimension is split.
  static constexpr auto block_size_split = size_t{1024};
  /// Defines the number of streams per device.
  static constexpr auto streams_per_gpu  = size_t{8};
  /// Default mask value for a single gpu.
  static constexpr auto single_gpu_mask  = size_t{1};

 public:
  //==--- [aliases] --------------------------------------------------------==//
  
  /// Defines the type of the gpu mask for the grid.
  using gpu_mask_t    = std::bitset<16>;
  /// Defines the type of the iterator over host data.
  using host_iter_t   = typename BlockTraits<host_block_t>::iter_t;
  /// Defines the type of the iterator over device data.
  using device_iter_t = typename BlockTraits<device_block_t>::iter_t;

  //==--- [construction] ---------------------------------------------------==//

  /// Initializes the size of each of the dimensions of the grid. This
  /// constructor will create blocks with the largest size such that the minimum
  /// number of blocks are used which fit into the device memory.
  ///
  /// By default this creates the grid to run on a single gpu.
  ///
  /// \param  topo  A reference to the system topology.
  /// \param  sizes The sizes of the dimensions for the grid.
  /// \tparam Sizes The types of the dimension sizes.
  template <
    typename... Sizes,      
    all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0
  >
  Grid(Topology& topo, Sizes&&... sizes) 
  : _space{sizes...}, _topo{topo}, _gpu_mask{single_gpu_mask} {
    reallocate();
  }

  /// Initializes the size of each of the dimensions of the grid, allocating the
  /// memory for the grid, with padding data for the boundary of the grid. This
  /// constructor is only enabled when the number of arguments matches the
  /// dimensionality of the grid and the \p sizes are numeric types.
  ///
  /// By default this creates the grid to run on a single gpu.
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
  : _space{padding, sizes...}, _topo{topo}, _gpu_mask{single_gpu_mask} {
    reallocate();
  }

  /// Initializes the size of each of the dimensions of the grid, allocating the
  /// memory for the grid, with padding data for the boundary of the grid. This
  /// constructor is only enabled when the number of arguments matches the
  /// dimensionality of the grid and the \p sizes are numeric types.
  ///
  /// This also allows a mask for which gpus to use to be set.
  ///
  /// \param  topo    A reference to the system topology.
  /// \param  padding The amount of padding for the grid.
  /// \param  mask    The mask for which gpus can be used.
  /// \param  sizes   The sizes of the dimensions for the grid.
  /// \tparam Sizes   The types of other dimension sizes.
  template <
    typename... Sizes,      
    all_arithmetic_size_enable_t<Dimensions, Sizes...> = 0
  >
  Grid(Topology& topo, size_t padding, gpu_mask_t mask, Sizes&&... sizes) 
  : _space{padding, sizes...}, _topo{topo}, _gpu_mask{mask} {
    reallocate();
  }

  /// Destructor -- resets the amount of device memory allocated for the blocks
  /// on each of the gpus.
  ~Grid() {
    for (auto& block : _blocks) {
      if (block.gpu_id >= 0) {
        _topo.gpus[block.gpu_id].mem_alloc -=
          block.device_data.mem_requirement();
      }
    }
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
    if (single_gpu()) {
      return access_single_gpu(std::forward<Indices>(is)...);
    }
    assert(false);
    return access_multi_gpu(std::forward<Indices>(is)...);
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
  auto size(Dim&& dim) const -> size_t {
    return _space.internal_size(std::forward<Dim>(dim));
  }

  /// Returns the amount of padding around the grid, and around each of the
  /// blocks with make up the grid.
  auto padding() const -> size_t {
    return _space.padding();
  }

  /// Returns a const reference to the gpu mask for the grid.
  auto gpu_mask() const -> const gpu_mask_t& {
    return _gpu_mask;
  }

  /// Returns the max number of gpus which this grid may use.
  auto num_gpus_used() const -> size_t {
    return _gpu_mask.count();
  }

  /// Returns true if the grid is only running in single gpu mode.
  auto single_gpu() const -> bool {
    return num_gpus_used() == size_t{1};
  }

  //==--- [resizing] -------------------------------------------------------==//
  
  /// Resizes the grid in the \p dim dimension to a size of \p elements. Note 
  /// that this leaves the grid in a state where behaviour is undefined until a
  /// call to ``reallocate()`` is made.
  ///
  /// \param  dim      The dimension to resize.
  /// \param  elements The number of elements to resize to.
  /// \tparam Dim      The type of the dimension specifier.
  template <typename Dim>
  auto resize_dim(Dim&& dim, size_t elements) -> void {
    _space.resize_dim(dim, elements);
  }

  /// Cleans up and then realocates all data for the grid, on both the host and
  /// the device. This is expensive, and this should only be done at application
  /// startup. After a call to ``reallocate()`` all previous data is invalid.
  auto reallocate() -> void {
    resize_block_size(dims_v);
    default_allocation();
  }

  //==--- [pipeline invoke] ------------------------------------------------==//

  /// Applies the \p pipeline to the grid, performing the operations in the
  /// pipeline.
  /// \param  pipeline The pipeline to apply to the grid.
  /// \tparam Ops      The operations in the pipeline.
  template <typename... Ops>
  auto apply_pipeline(const Pipeline<Ops...>& pipeline) -> void {
    if (single_gpu()) {
      auto block = _blocks.begin();

      block->ensure_device_data_available();

      invoke(block->device_data, pipeline);
      block->data_state = block_state_t::updated_device;
    }
  }

  /// Applies the \p pipeline to the grid, performing the operations in the
  /// pipeline, also passing the \p other grid to the pipeline as the second
  /// argument to the pipeline.
  /// \param  other    The other grid to pass to the pipeline.
  /// \param  pipeline The pipeline to apply to the grid.
  /// \tparam U        The type of the data for the other grid.
  /// \tparam Ops      The operations in the pipeline.
  template <typename U, typename... Ops>
  auto apply_pipeline(
    const Pipeline<Ops...>& pipeline, Grid<U, Dimensions>& other
  ) -> void {
    if (single_gpu()) {
      auto block_other = other.begin();
      auto block       = _blocks.begin();

      block->ensure_device_data_available();
      block_other->ensure_device_data_available();

      invoke(block->device_data, block_other->device_data, pipeline);
      block->data_state = block_state_t::updated_device;
    }
  }

 private:
  space_t    _space;        //!< The space for the grid.
  space_t    _block_size;   //!< The size of the blocks in the grid.
  blocks_t   _blocks;       //!< Blocks for the grid.
  Topology&  _topo;         //!< Topology for the system.
  gpu_mask_t _gpu_mask;     //!< Max number of gpus for the grid.

  //==--- [resize] ---------------------------------------------------------==//

  /// Defines the size of the default blocks for 1D. The default is to split
  /// the x dimension across the allowed number of gpus for the grid. Since this
  /// is the only way to split the data, it's an obvious choice.
  auto resize_block_size(dimx_t) -> void {
    _block_size = DynamicMultidimSpace<1>{
      single_gpu() 
        ? _space.internal_size(dim_x)
        : _space.internal_size(dim_x) / num_gpus_used()
    };
  }

  /// Defines the size of the default blocks for 2D. The default is to split
  /// the y dimension across the GPUs, since this requires only copying data in
  /// the x plane, which is contiguously allocated and so reasults in coalesced
  /// access when copying.
  auto resize_block_size(dimy_t) -> void {
    _block_size = DynamicMultidimSpace<2>{
      _space.internal_size(dim_x),
      single_gpu()
        ? _space.internal_size(dim_y)
        : _space.internal_size(dim_y) / num_gpus_used()
    };
  }

  /// Defines the size of the default blocks for 2D. The default is to split
  /// the z dimension, since this means that when copying between blocks, the
  /// data to copy is 2D in the xy-plane, which is contiguously allocated and
  /// therefore results in coalesced access.
  auto resize_block_size(dimz_t) -> void {
    _block_size = DynamicMultidimSpace<3>{
      _space.internal_size(dim_x),
      _space.internal_size(dim_y),
      single_gpu()
        ? _space.internal_size(dim_z)
        : _space.internal_size(dim_z) / num_gpus_used()
    };
  }

  //==--- [allocation] -----------------------------------------------------==//

  /// Allocates the blocks based on the defined block size.
  auto allocate_blocks() -> void {
    const size_t blocks = single_gpu() ? 1 : std::ceil(
      static_cast<float>(_space.internal_size(dims_v)) / 
      static_cast<float>(_block_size.internal_size(dims_v))
    );
    for (auto i : range(blocks)) {
      _blocks.emplace_back();
    }
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
    std::vector<size_t> stream_ids(num_gpus_used());
    for (auto& stream_id : stream_ids) { 
      stream_id = 0; 
    }

    // Currently this assumes data is split along x dimension.
    const size_t blocks_per_gpu = _blocks.size() / num_gpus_used();
    size_t       iter           = 0;
    for (auto& block : _blocks) {
      // Set the host component of the block to enable asynchronous operations
      // so that we can overlap compute and transfer:
      block.host_data.set_op_kind(BlockOpKind::asynchronous);

      // Now set the padding for the block:
      block.host_data.set_padding(_space.padding());
      block.device_data.set_padding(_space.padding());

      // Resize each of the dimensions of the block, and set the block index.
      unrolled_for<Dimensions>([&] (auto dim) {
        //b->indices[dim] = global_idx(dim);
        block.host_data.resize_dim(dim, _block_size.internal_size(dim));
        block.device_data.resize_dim(dim, _block_size.internal_size(dim));
      });

      // Check if there is enough memory on the device to allocate:
      const size_t gpu_id = iter++ / blocks_per_gpu;
      auto& gpu           = _topo.gpus[gpu_id];
      auto& stream_id     = stream_ids[gpu_id];
      block.gpu_id        = gpu.index;
      cudaSetDevice(gpu.index); 

      // Allocate the host memory:
      block.host_data.reallocate();

      // Now alloate device data:
      block.device_data.set_stream(gpu.streams[stream_id]);
      block.device_data.reallocate();

      // Update global gpu memory data:
      gpu.mem_alloc += block.device_data.mem_requirement();
      stream_id      = (stream_id + 1) % gpu.streams.size();

      block.data_state = block_state_t::updated_device;
    }
  }

  /// Default allocation of the block data for the grid to create the blocks for
  /// the given block size, and allocate memory for the blocks on the
  /// appropriate devices.
  auto default_allocation() -> void {
    allocate_blocks();
    allocate_data_for_blocks();
  }

  //==--- [access] ---------------------------------------------------------==//
  
  /// Gets an iterator to the element at the location specified by the \p is 
  /// indices, for the case that only a single gpu is being used by the grid.
  ///
  /// \param  is      The indices to get the element at.
  /// \tparam Indices The types of the indices.
  template <typename... Indices>
  ripple_host_only auto access_single_gpu(Indices&&... is) -> host_iter_t {
    auto block = _blocks.begin();

    // Copy data back from the device, if necessary.
    block->ensure_host_data_available();
    auto it = block->host_data.operator()(std::forward<Indices>(is)...);
    block->data_state = block_state_t::updated_host;
    return it;
  }

  /// Gets an iterator to the element at the location specified by the \p is 
  /// indices, for the case that multiple gpus are being used by the grid.
  ///
  /// \param  is      The indices to get the element at.
  /// \tparam Indices The types of the indices.
  template <typename... Indices>
  ripple_host_only auto access_multi_gpu(Indices&&... is) -> host_iter_t {
    auto block = _blocks.begin();

    // Copy data back from the device, if necessary.
    block->ensure_host_data_available();
    auto it = block->host_data.operator()(std::forward<Indices>(is)...);
    block->data_state = block_state_t::updated_host;
    return it;
  }
};

} // namespace ripple

#endif // RIPPLE_CONTAINER_GRID_HPP

