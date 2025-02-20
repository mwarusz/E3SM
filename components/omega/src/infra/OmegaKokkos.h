#ifndef OMEGA_KOKKOS_H
#define OMEGA_KOKKOS_H
//===-- base/OmegaKokkos.h - Omega extension of Kokkos ------*- C++ -*-===//
//
/// \file
/// \brief Extends Kokkos for Omega
///
/// This header extends Kokkos for Omega.
//
//===----------------------------------------------------------------------===//

#include "DataTypes.h"
#include <type_traits>
#include <utility>

namespace OMEGA {

#define OMEGA_SCOPE(a, b) auto &a = b

/// An enum is used to provide a shorthand for determining the type of
/// field. These correspond to the supported Omega data types (Real will be
/// identical to R4 or R8 depending on settings)
enum class ArrayDataType { Unknown, I4, I8, R4, R8 };

/// An enum is used to identify the location of the data - currently
/// either the device (the default) or explicitly on the host. Both refers
/// to the CPU-only case where the host and device are identical.
enum class ArrayMemLoc { Unknown, Device, Host, Both };

// determine ArrayDataType from Kokkos array type
template <class T> constexpr ArrayDataType checkArrayType() {
   if (std::is_same_v<typename T::non_const_value_type, I4>) {
      return ArrayDataType::I4;
   }

   if (std::is_same_v<typename T::non_const_value_type, I8>) {
      return ArrayDataType::I8;
   }

   if (std::is_same_v<typename T::non_const_value_type, R4>) {
      return ArrayDataType::R4;
   }

   if (std::is_same_v<typename T::non_const_value_type, R8>) {
      return ArrayDataType::R8;
   }

   return ArrayDataType::Unknown;
}

// determine ArrayMemLoc from Kokkos array type
template <class T> constexpr ArrayMemLoc findArrayMemLoc() {
   if (std::is_same_v<MemSpace, HostMemSpace>) {
      return ArrayMemLoc::Both;
   } else if (T::is_hostspace) {
      return ArrayMemLoc::Host;
   } else {
      return ArrayMemLoc::Device;
   }
}

/// Struct template to specify the rank of a supported Array
template <class T> struct ArrayRank {
   static constexpr bool Is1D = T::rank == 1;
   static constexpr bool Is2D = T::rank == 2;
   static constexpr bool Is3D = T::rank == 3;
   static constexpr bool Is4D = T::rank == 4;
   static constexpr bool Is5D = T::rank == 5;
};

using ExecSpace     = MemSpace::execution_space;
using HostExecSpace = HostMemSpace::execution_space;

template <typename V>
auto createHostMirrorCopy(const V &View)
    -> Kokkos::View<typename V::data_type, HostMemLayout, HostMemSpace> {
   return Kokkos::create_mirror_view_and_copy(HostExecSpace(), View);
}

template <typename V>
auto createDeviceMirrorCopy(const V &View)
    -> Kokkos::View<typename V::data_type, MemLayout, MemSpace> {
   return Kokkos::create_mirror_view_and_copy(ExecSpace(), View);
}

// function alias to follow Camel Naming Convention
template <typename D, typename S> void deepCopy(D &&Dst, S &&Src) {
   Kokkos::deep_copy(std::forward<D>(Dst), std::forward<S>(Src));
}

template <typename E, typename D, typename S>
void deepCopy(E &Space, D &Dst, const S &Src) {
   Kokkos::deep_copy(Space, Dst, Src);
}

#if OMEGA_LAYOUT_RIGHT

template <int N, class... Args>
using Bounds = Kokkos::MDRangePolicy<
    ExecSpace, Kokkos::Rank<N, Kokkos::Iterate::Right, Kokkos::Iterate::Right>,
    Args...>;

#elif OMEGA_LAYOUT_LEFT

template <int N, class... Args>
using Bounds = Kokkos::MDRangePolicy<
    ExecSpace, Kokkos::Rank<N, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
    Args...>;

#else

#error "OMEGA Memory Layout is not defined."

#endif

// parallelFor: with label
template <int N, class F, class... Args>
inline void parallelFor(const std::string &Label, const int (&UpperBounds)[N],
                        const F &Functor,
                        const int (&Tile)[N] = DefaultTile<N>::value) {
   if constexpr (N == 1) {
      const auto Policy = Kokkos::RangePolicy<Args...>(0, UpperBounds[0]);
      Kokkos::parallel_for(Label, Policy, Functor);

   } else {
      const int LowerBounds[N] = {0};
      const auto Policy = Bounds<N, Args...>(LowerBounds, UpperBounds, Tile);
      Kokkos::parallel_for(Label, Policy, Functor);
   }
}

// parallelFor: without label
template <int N, class F>
inline void parallelFor(const int (&UpperBounds)[N], const F &Functor,
                        const int (&Tile)[N] = DefaultTile<N>::value) {
   parallelFor("", UpperBounds, Functor, Tile);
}

// parallelReduce: with label
template <int N, class F, class R, class... Args>
inline void parallelReduce(const std::string &Label,
                           const int (&UpperBounds)[N], const F &Functor,
                           R &&Reducer,
                           const int (&Tile)[N] = DefaultTile<N>::value) {
   if constexpr (N == 1) {
      const auto Policy = Kokkos::RangePolicy<Args...>(0, UpperBounds[0]);
      Kokkos::parallel_reduce(Label, Policy, Functor, std::forward<R>(Reducer));

   } else {
      const int LowerBounds[N] = {0};
      const auto Policy = Bounds<N, Args...>(LowerBounds, UpperBounds, Tile);
      Kokkos::parallel_reduce(Label, Policy, Functor, std::forward<R>(Reducer));
   }
}

// parallelReduce: without label
template <int N, class F, class R, class... Args>
inline void parallelReduce(const int (&UpperBounds)[N], const F &Functor,
                           R &&Reducer,
                           const int (&Tile)[N] = DefaultTile<N>::value) {
   parallelReduce("", UpperBounds, Functor, std::forward<R>(Reducer), Tile);
}

} // end namespace OMEGA

//===----------------------------------------------------------------------===//
#endif
