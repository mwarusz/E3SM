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
#include <functional>
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

using ExecSpace       = MemSpace::execution_space;
using HostExecSpace   = HostMemSpace::execution_space;
using TeamPolicy      = Kokkos::TeamPolicy<ExecSpace>;
using TeamMember      = TeamPolicy::member_type;
using ScratchMemSpace = ExecSpace::scratch_memory_space;
using Kokkos::MemoryUnmanaged;
using Kokkos::subview;
using Kokkos::TeamThreadRange;

/// team_size for hierarchical parallelism
#ifdef OMEGA_TARGET_DEVICE
constexpr int OMEGA_TEAMSIZE = 64;
#else
constexpr int OMEGA_TEAMSIZE = 1;
#endif

// Takes a functor that uses multidimensional indexing
// and converts it into one that also accepts linear index
template <class F, int Rank> struct LinearIdxWrapper : F {
   using F::operator();

   LinearIdxWrapper(F &&Functor, const int (&Bounds)[Rank])
       : F(std::move(Functor)) {

      if constexpr (Rank > 1) {
         Strides[Rank - 2] = Bounds[Rank - 1];
         for (int I = Rank - 3; I >= 0; --I) {
            Strides[I] = Bounds[I + 1] * Strides[I + 1];
         }
      }
   }

   template <int N = Rank, class... Args>
   KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<N == 2>
   operator()(int Idx, Args &&...OtherArgs) const {
      const int I1 = Idx / Strides[0];
      const int I2 = Idx - I1 * Strides[0];

      (*this)(I1, I2, std::forward<Args>(OtherArgs)...);
   }

   template <int N = Rank, class... Args>
   KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<N == 3>
   operator()(int Idx, Args &&...OtherArgs) const {
      const int I1 = Idx / Strides[0];
      Idx -= I1 * Strides[0];
      const int I2 = Idx / Strides[1];
      const int I3 = Idx - I2 * Strides[1];

      (*this)(I1, I2, I3, std::forward<Args>(OtherArgs)...);
   }

   template <int N = Rank, class... Args>
   KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<N == 4>
   operator()(int Idx, Args &&...OtherArgs) const {
      const int I1 = Idx / Strides[0];
      Idx -= I1 * Strides[0];
      const int I2 = Idx / Strides[1];
      Idx -= I2 * Strides[1];
      const int I3 = Idx / Strides[2];
      const int I4 = Idx - I3 * Strides[2];

      (*this)(I1, I2, I3, I4, std::forward<Args>(OtherArgs)...);
   }

   template <int N = Rank, class... Args>
   KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<N == 5>
   operator()(int Idx, Args &&...OtherArgs) const {
      const int I1 = Idx / Strides[0];
      Idx -= I1 * Strides[0];
      const int I2 = Idx / Strides[1];
      Idx -= I2 * Strides[1];
      const int I3 = Idx / Strides[2];
      Idx -= I3 * Strides[2];
      const int I4 = Idx / Strides[3];
      const int I5 = Idx - I4 * Strides[3];

      (*this)(I1, I2, I3, I4, I5, std::forward<Args>(OtherArgs)...);
   }

   int Strides[Rank - 1];
};

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

using Bounds1D = Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<int>>;

#if OMEGA_LAYOUT_RIGHT

template <int N>
using Bounds = Kokkos::MDRangePolicy<
    ExecSpace, Kokkos::Rank<N, Kokkos::Iterate::Right, Kokkos::Iterate::Right>,
    Kokkos::IndexType<int>>;

#elif OMEGA_LAYOUT_LEFT

template <int N>
using Bounds = Kokkos::MDRangePolicy<
    ExecSpace, Kokkos::Rank<N, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
    Kokkos::IndexType<int>>;

#else

#error "OMEGA Memory Layout is not defined."

#endif

// parallelFor: with label
template <int N, class F>
inline void parallelFor(const std::string &Label, const int (&UpperBounds)[N],
                        const F &Functor) {
   if constexpr (N == 1) {
      const auto Policy = Bounds1D(0, UpperBounds[0]);
      Kokkos::parallel_for(Label, Policy, Functor);

   } else {
#ifdef OMEGA_TARGET_DEVICE
      // On device convert the functor to use one dimensional indexing and use
      // 1D RangePolicy
      const auto LinFunctor = LinearIdxWrapper{std::move(Functor), UpperBounds};
      int LinBound          = 1;
      for (int Rank = 0; Rank < N; ++Rank) {
         LinBound *= UpperBounds[Rank];
      }
      const auto Policy = Bounds1D(0, LinBound);
      Kokkos::parallel_for(Label, Policy, LinFunctor);
#else
      // On host use MDRangePolicy
      const int LowerBounds[N] = {0};
      const auto Policy        = Bounds<N>(LowerBounds, UpperBounds);
      Kokkos::parallel_for(Label, Policy, Functor);
#endif
   }
}

// parallelFor: without label
template <int N, class F>
inline void parallelFor(const int (&UpperBounds)[N], const F &Functor) {
   parallelFor("", UpperBounds, Functor);
}

// parallelReduce: with label
template <int N, class F, class... R>
inline void parallelReduce(const std::string &Label,
                           const int (&UpperBounds)[N], const F &Functor,
                           R &&...Reducers) {
   if constexpr (N == 1) {
      const auto Policy = Bounds1D(0, UpperBounds[0]);
      Kokkos::parallel_reduce(Label, Policy, Functor,
                              std::forward<R>(Reducers)...);

   } else {

#ifdef OMEGA_TARGET_DEVICE
      // On device convert the functor to use one dimensional indexing and use
      // 1D RangePolicy
      const auto LinFunctor = LinearIdxWrapper{std::move(Functor), UpperBounds};
      int LinBound          = 1;
      for (int Rank = 0; Rank < N; ++Rank) {
         LinBound *= UpperBounds[Rank];
      }
      const auto Policy = Bounds1D(0, LinBound);
      Kokkos::parallel_reduce(Label, Policy, LinFunctor,
                              std::forward<R>(Reducers)...);
#else
      // On host use MDRangePolicy
      const int LowerBounds[N] = {0};
      const auto Policy        = Bounds<N>(LowerBounds, UpperBounds);
      Kokkos::parallel_reduce(Label, Policy, Functor,
                              std::forward<R>(Reducers)...);
#endif
   }
}

// parallelReduce: without label
template <int N, class F, class... R>
inline void parallelReduce(const int (&UpperBounds)[N], const F &Functor,
                           R &&...Reducers) {
   parallelReduce("", UpperBounds, Functor, std::forward<R>(Reducers)...);
}

#ifdef OMEGA_TARGET_DEVICE
constexpr int NElemPerTeam = 8;

#ifdef KOKKOS_ENABLE_HIP
constexpr int VectorSize = 32;
#else
constexpr int VectorSize = 16;
#endif

#else
constexpr int NElemPerTeam = 1;
constexpr int VectorSize   = 1;
#endif

// parallelForOuter: with label
template <int N, class F>
inline void parallelForOuter(const std::string &Label,
                             const int (&UpperBounds)[N], const F &Functor) {

   const auto LinFunctor = LinearIdxWrapper{std::move(Functor), UpperBounds};
   int LinBound          = 1;
   for (int Rank = 0; Rank < N; ++Rank) {
      LinBound *= UpperBounds[Rank];
   }

   if constexpr (NElemPerTeam == 1) {

      auto Policy = TeamPolicy(LinBound, OMEGA_TEAMSIZE);
      Kokkos::parallel_for(
          Label, Policy, KOKKOS_LAMBDA(const TeamMember &Team) {
             const int TeamId = Team.league_rank();
             LinFunctor(TeamId, Team);
          });

   } else {

      auto Policy = TeamPolicy((LinBound + NElemPerTeam - 1) / NElemPerTeam,
                               NElemPerTeam, VectorSize);
      Kokkos::parallel_for(
          Label, Policy, KOKKOS_LAMBDA(const TeamMember &Team) {
             const int TeamId   = Team.league_rank();
             const int ThreadId = Team.team_rank();
             const int Idx      = TeamId * NElemPerTeam + ThreadId;
             if (Idx < LinBound) {
                LinFunctor(Idx, Team);
             }
          });
   }
}

// parallelForOuter: without label
template <int N, class F>
inline void parallelForOuter(const int (&UpperBounds)[N], const F &Functor) {
   parallelForOuter("", UpperBounds, Functor);
}

// template<class R>
// struct ReducerValueType {
//   using ValueType =
//   std::conditional_t<std::is_arithmetic_v<std::remove_reference_t<R>>, R,
//   typename R::value_type>;
// };

// parallelReduceOuter: with label
template <int N, class F, class... R>
inline void parallelReduceOuter(const std::string &Label,
                                const int (&UpperBounds)[N], const F &Functor,
                                R &&...Reducers) {

   const auto LinFunctor = LinearIdxWrapper{std::move(Functor), UpperBounds};
   int LinBound          = 1;
   for (int Rank = 0; Rank < N; ++Rank) {
      LinBound *= UpperBounds[Rank];
   }

   auto Policy = TeamPolicy(LinBound, OMEGA_TEAMSIZE);
   Kokkos::parallel_reduce(
       // Label, Policy, KOKKOS_LAMBDA<class... A>(const TeamMember &Team,
       // A&&... Accums) {
       Label, Policy,
       KOKKOS_LAMBDA(const TeamMember &Team, auto &...Accums) {
          // Label, Policy, KOKKOS_LAMBDA (const TeamMember &Team, typename
          // ReducerValueType<R>::value_type&... Accums) {
          const int TeamId = Team.league_rank();
          // LinFunctor(TeamId, Team, std::forward<A>(Accums)...);
          LinFunctor(TeamId, Team, Accums...);
       },
       std::forward<R>(Reducers)...);
}

// parallelForOuter: without label
template <int N, class F, class... R>
inline void parallelReduceOuter(const int (&UpperBounds)[N], const F &Functor,
                                R &&...Reducers) {
   parallelReduceOuter("", UpperBounds, Functor, std::forward<R>(Reducers)...);
}

// parallelForInner
template <class F>
KOKKOS_FUNCTION void parallelForInner(const TeamMember &Team, int UpperBound,
                                      const F &Functor) {
   if constexpr (NElemPerTeam == 1) {
      const auto Policy = TeamThreadRange(Team, UpperBound);
      Kokkos::parallel_for(Policy, Functor);
   } else {
      const auto Policy = ThreadVectorRange(Team, UpperBound);
      Kokkos::parallel_for(Policy, Functor);
   }
}

template <int ChunkSize, class F>
KOKKOS_FUNCTION void parallelForInnerChunked(const TeamMember &Team,
                                             int LowerBound, int UpperBound,
                                             const F &Functor) {
   const int NItems  = UpperBound - LowerBound + 1;
   const int NChunks = NItems / ChunkSize;
   const int NRem    = NItems - NChunks * ChunkSize;

   const auto Policy1 = TeamThreadRange(Team, NChunks);
   Kokkos::parallel_for(Policy1, [=](int IChunk) {
      const int I = LowerBound + ChunkSize * IChunk;
      Functor(I, std::true_type{});
   });

   if (NRem > 0) {
      const auto Policy2 = TeamThreadRange(Team, 1);
      Kokkos::parallel_for(Policy2, [=](int IChunk) {
         const int I = LowerBound + ChunkSize * NChunks;
         Functor(I, std::false_type{});
      });
   }
}

// parallelReduceInner
template <class F, class... R>
KOKKOS_FUNCTION void parallelReduceInner(const TeamMember &Team, int UpperBound,
                                         const F &Functor, R &&...Reducers) {

   const auto Policy = TeamThreadRange(Team, UpperBound);
   Kokkos::parallel_reduce(Policy, Functor, std::forward<R>(Reducers)...);
}

} // end namespace OMEGA

//===----------------------------------------------------------------------===//
#endif
