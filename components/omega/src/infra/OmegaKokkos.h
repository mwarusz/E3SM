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
#include "Error.h"
#include <functional>
#include <type_traits>
#include <utility>
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda/std/tuple>
#else
#include <tuple>
#endif

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
using TeamPolicy      = Kokkos::TeamPolicy<ExecSpace, Kokkos::IndexType<int>>;
using TeamMember      = TeamPolicy::member_type;
using ScratchMemSpace = ExecSpace::scratch_memory_space;
using Kokkos::MemoryUnmanaged;
using Kokkos::PerTeam;
using Kokkos::TeamThreadRange;
using Kokkos::ThreadVectorRange;

/// Default team and vector sizes for hierarchical parallelism
#ifdef OMEGA_TARGET_DEVICE

#ifdef KOKKOS_ENABLE_HIP
constexpr int OMEGA_TEAMSIZE   = 8;
constexpr int OMEGA_VECTORSIZE = 32;
#endif

#ifdef KOKKOS_ENABLE_SYCL
constexpr int OMEGA_TEAMSIZE   = 8;
constexpr int OMEGA_VECTORSIZE = 16;
#endif

#ifdef KOKKOS_ENABLE_CUDA
constexpr int OMEGA_TEAMSIZE   = 8;
constexpr int OMEGA_VECTORSIZE = 16;
#endif

#else

constexpr int OMEGA_TEAMSIZE   = 1;
constexpr int OMEGA_VECTORSIZE = 1;
#endif

struct TeamConfig {
   int TeamSize   = OMEGA_TEAMSIZE;
   int VectorSize = OMEGA_VECTORSIZE;

   TeamConfig() = default;
   explicit TeamConfig(int TeamSize, int VectorSize)
       : TeamSize(TeamSize), VectorSize(VectorSize) {}
};

inline TeamConfig &defaultTeamConfig() {
   static TeamConfig DefaultTeamCfg{};
   return DefaultTeamCfg;
}
void readKokkosConfig();

template <class... T> struct ThreadScratch {
   size_t BytesPerThread = 0;

   ThreadScratch() = default;

   template <int N> ThreadScratch(const int (&NVals)[N]) {
      static_assert(N == sizeof...(T));
      int I = 0;
      ((BytesPerThread += sizeof(T) * NVals[I++]), ...);
   }

   ThreadScratch(int NVals) : ThreadScratch({{NVals}}) {}
};

template <int N> struct LaunchConfig {
   std::array<int, N> UpperBounds;
   TeamConfig TeamCfg;
   size_t ScratchBytesPerThread;

   template <class... T>
   LaunchConfig(const int (&UpperBoundsIn)[N], const TeamConfig &TeamCfg,
                const ThreadScratch<T...> &Scratch)
       : TeamCfg(TeamCfg), ScratchBytesPerThread(Scratch.BytesPerThread) {
      std::copy(std::begin(UpperBoundsIn), std::end(UpperBoundsIn),
                std::begin(UpperBounds));
   }

   template <class... T>
   LaunchConfig(const int (&UpperBounds)[N], const ThreadScratch<T...> &Scratch)
       : LaunchConfig(UpperBounds, defaultTeamConfig(), Scratch) {}

   LaunchConfig(const int (&UpperBounds)[N], const TeamConfig &TeamCfg)
       : LaunchConfig(UpperBounds, TeamCfg, ThreadScratch<>{}) {}

   LaunchConfig(const int (&UpperBounds)[N])
       : LaunchConfig(UpperBounds, defaultTeamConfig(), ThreadScratch<>{}) {}
};

// Takes a functor that uses multidimensional indexing
// and converts it into one that also accepts linear index
template <class F, int Rank> struct LinearIdxWrapper : F {
   static_assert(Rank >= 1 && Rank <= 5, "LinearIdxWrapper supports ranks 1-5");
   using F::operator();

   template <class Array>
   LinearIdxWrapper(F &&Functor, Array &&Bounds) : F(std::move(Functor)) {
      computeStrides(std::forward<Array>(Bounds));
   }

   template <class Array>
   LinearIdxWrapper(const F &Functor, Array &&Bounds) : F(Functor) {
      computeStrides(std::forward<Array>(Bounds));
   }

   template <class Array> void computeStrides(Array &&Bounds) {
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

// SYCL doesn't allow 0-length arrays so add one extra element even though
// it is not needed
#ifdef KOKKOS_ENABLE_SYCL
   int Strides[Rank];
#else
   int Strides[Rank - 1];
#endif
};

template <class F, int Rank>
LinearIdxWrapper(F, const int (&)[Rank]) -> LinearIdxWrapper<F, Rank>;
template <class F, size_t Rank>
LinearIdxWrapper(F, std::array<int, Rank>) -> LinearIdxWrapper<F, Rank>;

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

// Check if two arrays are identical
template <class ArrayTypeA, class ArrayTypeB>
bool arraysEqual(const ArrayTypeA &A, const ArrayTypeB &B) {
   OMEGA_REQUIRE(A.span_is_contiguous() && B.span_is_contiguous(),
                 "arraysEqual works only for contiguous arrays");
   OMEGA_REQUIRE(A.size() == B.size(),
                 "arrayEqual can only compare arrays of equal size");

   // This is a debug utility and not performance critical
   // so just copy to the host and compare there
   const auto AH = createHostMirrorCopy(A);
   const auto BH = createHostMirrorCopy(B);

   bool Equal = true;
   for (size_t I = 0; I < AH.size(); I++) {
      if (AH.data()[I] != BH.data()[I]) {
         Equal = false;
         break;
      }
   }
   return Equal;
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
                        F &&Functor) {
   if constexpr (N == 1) {
      const auto Policy = Bounds1D(0, UpperBounds[0]);
      Kokkos::parallel_for(Label, Policy, std::forward<F>(Functor));

   } else {
#ifdef OMEGA_TARGET_DEVICE
      // On device convert the functor to use one dimensional indexing and use
      // 1D RangePolicy
      auto LinFunctor = LinearIdxWrapper{std::forward<F>(Functor), UpperBounds};
      int LinBound    = 1;
      for (int Rank = 0; Rank < N; ++Rank) {
         LinBound *= UpperBounds[Rank];
      }
      const auto Policy = Bounds1D(0, LinBound);
      Kokkos::parallel_for(Label, Policy, std::move(LinFunctor));
#else
      // On host use MDRangePolicy
      const int LowerBounds[N] = {0};
      const auto Policy        = Bounds<N>(LowerBounds, UpperBounds);
      Kokkos::parallel_for(Label, Policy, std::forward<F>(Functor));
#endif
   }
}

// parallelFor: without label
template <int N, class F>
inline void parallelFor(const int (&UpperBounds)[N], F &&Functor) {
   parallelFor("", UpperBounds, std::forward<F>(Functor));
}

// parallelReduce: with label
template <int N, class F, class... R>
inline void parallelReduce(const std::string &Label,
                           const int (&UpperBounds)[N], F &&Functor,
                           R &&...Reducers) {
   if constexpr (N == 1) {
      const auto Policy = Bounds1D(0, UpperBounds[0]);
      Kokkos::parallel_reduce(Label, Policy, std::forward<F>(Functor),
                              std::forward<R>(Reducers)...);

   } else {

#ifdef OMEGA_TARGET_DEVICE
      // On device convert the functor to use one dimensional indexing and use
      // 1D RangePolicy
      auto LinFunctor = LinearIdxWrapper{std::forward<F>(Functor), UpperBounds};
      int LinBound    = 1;
      for (int Rank = 0; Rank < N; ++Rank) {
         LinBound *= UpperBounds[Rank];
      }
      const auto Policy = Bounds1D(0, LinBound);
      Kokkos::parallel_reduce(Label, Policy, std::move(LinFunctor),
                              std::forward<R>(Reducers)...);
#else
      // On host use MDRangePolicy
      const int LowerBounds[N] = {0};
      const auto Policy        = Bounds<N>(LowerBounds, UpperBounds);
      Kokkos::parallel_reduce(Label, Policy, std::forward<F>(Functor),
                              std::forward<R>(Reducers)...);
#endif
   }
}

// parallelReduce: without label
template <int N, class F, class... R>
inline void parallelReduce(const int (&UpperBounds)[N], F &&Functor,
                           R &&...Reducers) {
   parallelReduce("", UpperBounds, std::forward<F>(Functor),
                  std::forward<R>(Reducers)...);
}

/// Hierarchical parallelism wrappers

#define INNER_LAMBDA [=]
// #define INNER_LAMBDA [&]

// parallelForOuter: with label
template <int N, class F>
inline void parallelForOuter(const std::string &Label,
                             const LaunchConfig<N> &Config, F &&Functor) {

   auto LinFunctor =
       LinearIdxWrapper{std::forward<F>(Functor), Config.UpperBounds};
   int LinBound = 1;
   for (int Rank = 0; Rank < N; ++Rank) {
      LinBound *= Config.UpperBounds[Rank];
   }

   const int TeamSize   = Config.TeamCfg.TeamSize;
   const int VectorSize = Config.TeamCfg.VectorSize;

   const int NTeams = (LinBound + TeamSize - 1) / TeamSize;

   auto Policy = TeamPolicy(NTeams, TeamSize, VectorSize);

   if (Config.ScratchBytesPerThread > 0) {
      Policy.set_scratch_size(0,
                              Kokkos::PerThread(Config.ScratchBytesPerThread));
   }

   Kokkos::parallel_for(
       Label, Policy, KOKKOS_LAMBDA(const TeamMember &Team) {
          const int TeamId   = Team.league_rank();
          const int ThreadId = Team.team_rank();
          const int Id       = TeamId * Team.team_size() + ThreadId;
          if (Id < LinBound) {
             LinFunctor(Id, Team);
          }
       });
}

// parallelForOuter: with label and with array bounds
template <int N, class F>
inline void parallelForOuter(const std::string &Label,
                             const int (&UpperBounds)[N], F &&Functor) {
   parallelForOuter(Label, LaunchConfig(UpperBounds), std::forward<F>(Functor));
}

// parallelForOuter: without label and with launch config
template <int N, class F>
inline void parallelForOuter(const LaunchConfig<N> &Config, F &&Functor) {
   parallelForOuter("", Config, std::forward<F>(Functor));
}

// parallelForOuter: without label and with array bounds
template <int N, class F>
inline void parallelForOuter(const int (&UpperBounds)[N], F &&Functor) {
   parallelForOuter("", LaunchConfig(UpperBounds), std::forward<F>(Functor));
}

template <int N, class F, class... R>
inline void parallelReduceOuterImpl(const std::string &Label,
                                    const LaunchConfig<N> &Config, F &&Functor,
                                    R &&...Reducers) {
#ifdef KOKKOS_ENABLE_CUDA
   using cuda::std::apply;
   using cuda::std::tuple;
#else
   using std::apply;
   using std::tuple;
#endif

   auto LinFunctor =
       LinearIdxWrapper{std::forward<F>(Functor), Config.UpperBounds};
   int LinBound = 1;
   for (int Rank = 0; Rank < N; ++Rank) {
      LinBound *= Config.UpperBounds[Rank];
   }

   const int TeamSize   = Config.TeamCfg.TeamSize;
   const int VectorSize = Config.TeamCfg.VectorSize;

   const int NTeams = (LinBound + TeamSize - 1) / TeamSize;

   auto Policy = TeamPolicy(NTeams, TeamSize, VectorSize);

   if (Config.ScratchBytesPerThread > 0) {
      Policy.set_scratch_size(0,
                              Kokkos::PerThread(Config.ScratchBytesPerThread));
   }

   Kokkos::parallel_reduce(
       Label, Policy,
       KOKKOS_LAMBDA(
           const TeamMember &Team,
           typename std::remove_reference_t<R>::value_type &...TeamAccums) {
          const int TeamId = Team.league_rank();

          tuple<typename std::remove_reference_t<R>::value_type...>
              TeamValsTuple;

          auto TeamReducersTuple = apply(
              [&](auto &...TeamVals) {
                 return tuple<std::remove_reference_t<R>...>(TeamVals...);
              },
              TeamValsTuple);

          apply(
              [&](auto &...TeamReducers) {
                 Kokkos::parallel_reduce(
                     TeamThreadRange(Team, Team.team_size()),
                     INNER_LAMBDA(int ThreadId, auto &...ThreadAccums) {
                        const int Id = TeamId * Team.team_size() + ThreadId;

                        if (Id < LinBound) {
                           LinFunctor(Id, Team, ThreadAccums...);
                        }
                     },
                     TeamReducers...);
              },
              TeamReducersTuple);

          Kokkos::single(Kokkos::PerTeam(Team), [&]() {
             apply(
                 [&](auto &...TeamReducers) {
                    (TeamReducers.join(TeamAccums, TeamReducers.reference()),
                     ...);
                 },
                 TeamReducersTuple);
          });
       },
       std::forward<R>(Reducers)...);
}

template <class T> auto forwardReductionArg(T &&arg) {
   using DerefT = std::remove_reference_t<T>;
   if constexpr (Kokkos::is_reducer_v<DerefT>) {
      return std::forward<T>(arg);
   } else {
      static_assert(std::is_arithmetic_v<DerefT>);
      return Kokkos::Sum<DerefT>(arg);
   }
}

// parallelReduceOuter: with label and with launch config
template <int N, class F, class... R>
inline void parallelReduceOuter(const std::string &Label,
                                const LaunchConfig<N> &Config, F &&Functor,
                                R &&...Reducers) {
   parallelReduceOuterImpl(Label, Config, std::forward<F>(Functor),
                           forwardReductionArg(Reducers)...);
}

// parallelReduceOuter: with label and with array bounds
template <int N, class F, class... R>
inline void parallelReduceOuter(const std::string &Label,
                                const int (&UpperBounds)[N], F &&Functor,
                                R &&...Reducers) {
   parallelReduceOuterImpl(Label, LaunchConfig(UpperBounds),
                           std::forward<F>(Functor),
                           forwardReductionArg(Reducers)...);
}

// parallelReduceOuter: without label and with launch config
template <int N, class F, class... R>
inline void parallelReduceOuter(const LaunchConfig<N> &Config, F &&Functor,
                                R &&...Reducers) {
   parallelReduceOuter("", Config, std::forward<F>(Functor),
                       std::forward<R>(Reducers)...);
}

// parallelReduceOuter: without label and with array bounds
template <int N, class F, class... R>
inline void parallelReduceOuter(const int (&UpperBounds)[N], F &&Functor,
                                R &&...Reducers) {
   parallelReduceOuter("", UpperBounds, std::forward<F>(Functor),
                       std::forward<R>(Reducers)...);
}

// parallelForInner
template <class F>
KOKKOS_FUNCTION void parallelForInner(const TeamMember &Team, int UpperBound,
                                      F &&Functor) {
   const auto Policy = ThreadVectorRange(Team, UpperBound);
   Kokkos::parallel_for(Policy, std::forward<F>(Functor));
}

// parallelReduceInner
template <class F, class... R>
KOKKOS_FUNCTION void parallelReduceInner(const TeamMember &Team, int UpperBound,
                                         F &&Functor, R &&...Reducers) {
   const auto Policy = ThreadVectorRange(Team, UpperBound);
   Kokkos::parallel_reduce(Policy, std::forward<F>(Functor),
                           std::forward<R>(Reducers)...);
}

// parallelScanInner
template <class F, class... R>
KOKKOS_FUNCTION void parallelScanInner(const TeamMember &Team, int UpperBound,
                                       F &&Functor, R &&...Reducers) {
   const auto Policy = ThreadVectorRange(Team, UpperBound);
   Kokkos::parallel_scan(Policy, std::forward<F>(Functor),
                         std::forward<R>(Reducers)...);
}

} // end namespace OMEGA

//===----------------------------------------------------------------------===//
#endif
