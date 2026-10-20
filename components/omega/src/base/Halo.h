#ifndef OMEGA_HALO_H
#define OMEGA_HALO_H
//===-- base/Halo.h - halo exchange definitions -----------------*- C++ -*-===//
//
/// \file
/// \brief Defines classes and methods for performing halo exchanges
///
/// This header defines classes and methods needed for exchanging the halo
/// elements of every supported array type in OMEGA with elements of the array
/// on neighboring mesh partitions. The Halo class constructor creates a Halo
/// object based on a given machine environment and parallel decomposition.
/// The Halo object contains all the info needed for performing a halo
/// exchange The halo exchanges are carried out via non-blocking MPI library
/// routines. The Halo class public member function exchangeFullArrayHalo
/// which is called by the user to perform halo exchanges is a template
/// function and thus is fully defined in this header.
///
//
//===----------------------------------------------------------------------===//

#include "DataTypes.h"
#include "Decomp.h"
#include "Logging.h"
#include "MachEnv.h"
#include "OmegaKokkos.h"
#include "Pacer.h"
#include "mpi.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

namespace OMEGA {

// Set the default MPI real data type as single or double precision based on
// the default real data type used by OMEGA.
#ifdef OMEGA_SINGLE_PRECISION
static const MPI_Datatype MPI_RealKind = MPI_FLOAT;
#else
static const MPI_Datatype MPI_RealKind = MPI_DOUBLE;
#endif

/// The MeshElement enum identifies the index space to use for a halo exchange.
enum MeshElement { OnCell, OnEdge, OnVertex };

// Conditionally resize MPI buffer if it is not large enough
template <class BufferType>
void expandBuffer(BufferType &Buffer, int BufferSize) {
   if (Buffer.extent_int(0) < BufferSize) {
      Buffer = BufferType(Buffer.label(), BufferSize);
   }
}

// Determine the number of array elements per cell, edge, or vertex
// in the input array
template <class ArrayType>
KOKKOS_FUNCTION I4 totalSize(const ArrayType &Array) {
   I4 TotSize;
   I4 NDims = Array.rank();

   if (NDims == 1) {
      TotSize = 1;
   } else if (NDims == 2) {
      TotSize = Array.extent(1);
   } else {
      TotSize = 1;
      for (int I = 0; I < NDims - 2; ++I) {
         TotSize *= Array.extent(I);
      }
      TotSize *= Array.extent(NDims - 1);
   }
   return TotSize;
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION auto makeArraySubview(const ViewType &Array, I4 Offset,
                                             I4 Size) {
   return Kokkos::subview(Array, Kokkos::make_pair(Offset, Offset + Size));
}

/// The Halo class contains two nested classes, ExchList and Neighbor classes,
/// defined below. The Neighbor class stores the data needed to perform a full
/// halo exchange with every neighboring task for any array defined on the mesh.
/// The local task ID and the MPI communicator handle are also stored here.
class Halo {
 private:
   /// Flag to allow passing device arrays to MPI_Irecv and MPI_Isend in
   /// startReceives and startSends, determined by pre-processing parameter
#ifdef OMEGA_MPI_ON_DEVICE
   const bool ExchOnDev = true;
#else
   const bool ExchOnDev = false;
#endif

   /// The default Halo handles halo exchanges for arrays defined on the mesh
   /// with the default decomposition. A pointer is stored here for easy
   /// retrieval.
   static Halo *DefaultHalo;

   /// All halos are tracked/stored within the class as a map paired with a
   /// name for later retrieval.
   static std::map<std::string, std::unique_ptr<Halo>> AllHalos;

   const Decomp *MyDecomp{nullptr}; /// Pointer to decomposition object

   I4 NNghbr;       /// number of neighboring tasks
   I4 MyTask;       /// local MPI Task ID
   I4 HaloWidth;    /// cell width of halo
   MPI_Comm MyComm; /// MPI communicator handle

   /// Forward Declaration of Neighbor class, defined below
   class Neighbor;

   /// Sorted list of Task IDs of all neighboring tasks in ascending order.
   /// Another task is considered a neighbor if a mesh element owned by that
   /// task is in the halo of the local task in at least one index space, or if
   /// an owned element is in the halo of that task in at least one index space.
   std::vector<I4> NeighborList;

   /// Flags to control which tasks in NeighborList the local task needs to
   /// send elements to and receive elements from for each index space.
   HostArray1DI4 SendFlagsH[3], RecvFlagsH[3];
   Array1DI4 SendFlags[3], RecvFlags[3];

   /// The ExchList class contains the information needed to pack values from
   /// an array into a buffer or unpack values from a buffer into an array
   /// for a particular index space and all neighbors.
   class ExchList {
    private:
      /// Total number of elements in each neighbor's exchange list
      HostArray1DI4 NTotH;
      Array1DI4 NTot;
      /// Backing arrays for all neighbors' local indices
      HostArray1DI4 IndexHBacking;
      Array1DI4 IndexBacking;
      /// Offsets for each neighbor's indices in the backing arrays
      HostArray1DI4 IndexOffsetH;
      Array1DI4 IndexOffset;
      /// Create a neighbor's host index subview on demand

      /// The input contains a list of indices for each neighbor and halo layer
      ExchList(const std::vector<std::vector<std::vector<I4>>> &Lists);

      HostArray1DI4 hostIndices(const I4 INeighbor) const {
         const I4 Offset = IndexOffsetH(INeighbor);
         return Kokkos::subview(
             IndexHBacking,
             Kokkos::make_pair(Offset, Offset + NTotH[INeighbor]));
      }

      /// Empty constructor for ExchList class
      ExchList() = default;

    public:
      /// Halo and Neighbor are friend class to allow access to private
      /// members of the class
      friend class Halo;
      friend class Neighbor;
   }; // end class ExchList

   /// The Neighbor class stores each kind of halo-exchange data for all
   /// neighboring tasks in separate arrays.
   class Neighbor {
    private:
      std::vector<I4> TaskID; /// IDs of neighboring tasks

      /// Arrays of ExchList objects for sends and recieves for each
      /// index space. 0 = OnCell, 1 = OnEdge, 2 = OnVertex
      ExchList SendLists[3], RecvLists[3];
      /// Buffers for MPI communication on host and device
      HostArray1DR8 SendBufferH, RecvBufferH;
      Array1DR8 SendBuffer, RecvBuffer;
      /// Offsets for each neighbor in the shared communication buffers
      HostArray1DI4 SendBufferOffsetH, RecvBufferOffsetH;
      Array1DI4 SendBufferOffset, RecvBufferOffset;
      /// MPI request handles for non-blocking MPI communication
      std::vector<MPI_Request> RReq, SReq;

      // Flags to track whether a message has been received and whether
      // the buffer has been unpacked yet. The MPI_Test routine requires an
      // integer, so Received is stored as an I4.
      std::vector<I4> Received;
      std::vector<I4> Unpacked;

      Neighbor() = default;

      /// Construct exchange data for all neighboring tasks.
      Neighbor(const std::vector<std::vector<std::vector<I4>>> &SendCell,
               const std::vector<std::vector<std::vector<I4>>> &SendEdge,
               const std::vector<std::vector<std::vector<I4>>> &SendVert,
               const std::vector<std::vector<std::vector<I4>>> &RecvCell,
               const std::vector<std::vector<std::vector<I4>>> &RecvEdge,
               const std::vector<std::vector<std::vector<I4>>> &RecvVert,
               const std::vector<I4> &NeighborIDs);

    public:
      /// Halo is a friend class to allow access to private members
      /// of the class
      friend class Halo;

   }; // end class Neighbor

   /// All exchange lists, buffers, and MPI state for neighboring tasks.
   Neighbor Neighbors;

   // Private methods

   /// Uses info from Decomp to generate a sorted list of tasks that own
   /// elements in the the Halo of the local task for a particular index space.
   /// Utilized only during halo construction
   int generateListOfTasksInHalo(const I4 NOwned, const I4 NAll,
                                 HostArray2DI4 Locs,
                                 std::vector<I4> &ListOfTasks);

   /// Set SendFlags and RecvFlags for the input index space. Utilized only
   /// during halo construction
   int setNeighborFlags(std::vector<I4> NeighborElem,
                        const MeshElement IdxSpace);

   /// Uses info from Decomp to determine all tasks which own elements in the
   /// halo of the local task or need locally owned elements for their halo.
   /// Utilized only during halo construction
   int determineNeighbors(const I4 NumTasks);

   /// Send a vector of integers to each neighboring task and receive a vector
   /// of integers from each neighboring task. The first dimension of each
   /// input 2D vector represents the task in the order they appear in
   /// NeighborList. Utilized only during halo construction
   int exchangeVectorInt(const std::vector<std::vector<I4>> &SendVec,
                         std::vector<std::vector<I4>> &RecvVec);

   /// Generate the lists of indices to send to and receive from each
   /// neighboring task for the input IndexSpace and the Decomp pointed to
   /// by member variable MyDecomp, and save the lists in the input 3D
   /// vectors SendLists and RecvLists. The first dimension of these vectors
   /// represent the task in the order they appear in NeighborList, and the
   /// remaining 2D vector is used in constructing a Neighbor object for
   /// that task. Utilized only during halo construction
   int
   generateExchangeLists(std::vector<std::vector<std::vector<I4>>> &SendLists,
                         std::vector<std::vector<std::vector<I4>>> &RecvLists,
                         const MeshElement IndexSpace);

   /// Allocate the recieve buffers and call MPI_Irecv for each Neighbor.
   /// The input bool UseDevBuffer specifies whether or not the device buffer
   /// will be used in the unpackBuffer functionfor unpacking into the array.
   int startReceives(bool UseDevBuffer, I4 TotSize, I4 CurElem);

   /// Call MPI_Isend for each Neighbor to send the packed buffers to the
   /// neighboring tasks. The input bool UseDevBuffer specifies whether or not
   /// the device buffer was packed in the packBuffer function.
   int startSends(bool UseDevBuffer, I4 TotSize, I4 CurElem);

   /// Function template that returns a bool that is true if the Array is
   /// on the device, or if the device and host memory spaces are the same
   /// space. Used to determine if buffer packs and unpacks are done within
   /// Kokkos parallelFor kernels.
   template <typename T> bool devBufferPUP(const T &Array) {
      bool OnDev = findArrayMemLoc<T>() == ArrayMemLoc::Both ||
                   findArrayMemLoc<T>() == ArrayMemLoc::Device;
      return OnDev;
   }

   /// Construct a new halo labeled Name for the input MachEnv and Decomp
   Halo(const std::string &Name, const MachEnv *InEnv, const Decomp *InDecomp);

   // Forbid copy and move construction
   Halo(const Halo &) = delete;
   Halo(Halo &&)      = delete;

 public:
   // Methods

   /// initialize default Halo
   static int init();

   /// Creates a new halo by calling the constructor and puts it in the AllHalos
   /// map
   static Halo *create(const std::string &Name, const MachEnv *Env,
                       const Decomp *Decomp);

   /// Destructor
   ~Halo();

   /// Erase - removes Halo by name
   static void erase(std::string InName ///< [in] name of Halo to remove
   );

   /// Clear - removes all defined Halo instances
   static void clear();

   /// Retrieves a pointer to the default Halo object.
   static Halo *getDefault();

   /// Retrieves a pointer to a Halo object by Name
   static Halo *get(std::string Name);

   /// Retrieves MPI communicator from a Halo object
   MPI_Comm getComm() const;

   /// Buffer pack specialized function templates for supported Kokkos array
   /// ranks. Select out the proper elements from the input Array to send to a
   /// neighboring task and pack them into the proper send buffer for
   /// that Neighbor.
   template <typename T>
   std::enable_if_t<ArrayRank<T>::Is1D>
   packBuffer(const T &Array,       // 1D Kokkos array of any type
              const I4 CurNeighbor, // current neighbor
              const I4 CurElem      // current element
   ) {
      using ValType = typename T::non_const_value_type;

      OMEGA_SCOPE(LocList, Neighbors.SendLists[CurElem]);

      const I4 NTotList     = LocList.NTotH[CurNeighbor];
      const I4 BufferSize   = NTotList;
      const I4 BufferOffset = Neighbors.SendBufferOffsetH(CurNeighbor);

      if (devBufferPUP(Array)) {
         const auto IndexBacking  = LocList.IndexBacking;
         const auto IndexOffset   = LocList.IndexOffset;
         const auto Buffer        = Neighbors.SendBuffer;
         const auto BufferOffsets = Neighbors.SendBufferOffset;
         parallelFor(
             {NTotList}, KOKKOS_LAMBDA(int IExch) {
                const auto LocIndex = makeArraySubview(
                    IndexBacking, IndexOffset(CurNeighbor), NTotList);
                const auto LocBuff = makeArraySubview(
                    Buffer, BufferOffsets(CurNeighbor), BufferSize);
                const ValType Val = Array(LocIndex(IExch));
                R8 RVal;
                memcpy(&RVal, &Val, sizeof(ValType));
                LocBuff(IExch) = RVal;
             });
      } else {
         auto LocIndexH = LocList.hostIndices(CurNeighbor);
         auto LocBuffH  = Kokkos::subview(
             Neighbors.SendBufferH,
             Kokkos::make_pair(BufferOffset, BufferOffset + BufferSize));
         for (int IExch = 0; IExch < LocList.NTotH[CurNeighbor]; ++IExch) {
            const ValType Val = Array(LocIndexH(IExch));
            R8 RVal;
            memcpy(&RVal, &Val, sizeof(ValType));
            LocBuffH(IExch) = RVal;
         }
      }
   }

   template <typename T>
   std::enable_if_t<ArrayRank<T>::Is2D>
   packBuffer(const T &Array,       // 2D Kokkos array of any type
              const I4 CurNeighbor, // current neighbor
              const I4 CurElem      // current element
   ) {

      using ValType = typename T::non_const_value_type;

      OMEGA_SCOPE(LocList, Neighbors.SendLists[CurElem]);

      const I4 NJ = Array.extent(1);

      const I4 NTotList     = LocList.NTotH[CurNeighbor];
      const I4 BufferSize   = NTotList * totalSize(Array);
      const I4 BufferOffset = Neighbors.SendBufferOffsetH(CurNeighbor);

      if (devBufferPUP(Array)) {
         const auto IndexBacking  = LocList.IndexBacking;
         const auto IndexOffset   = LocList.IndexOffset;
         const auto Buffer        = Neighbors.SendBuffer;
         const auto BufferOffsets = Neighbors.SendBufferOffset;

         parallelFor(
             {NTotList, NJ}, KOKKOS_LAMBDA(int IExch, int J) {
                const auto LocIndex = makeArraySubview(
                    IndexBacking, IndexOffset(CurNeighbor), NTotList);
                const auto LocBuff = makeArraySubview(
                    Buffer, BufferOffsets(CurNeighbor), BufferSize);
                const ValType Val = Array(LocIndex(IExch), J);
                R8 RVal;
                memcpy(&RVal, &Val, sizeof(ValType));
                const I4 IBuff = IExch * NJ + J;
                LocBuff(IBuff) = RVal;
             });
      } else {
         auto LocIndexH = LocList.hostIndices(CurNeighbor);
         auto LocBuffH  = Kokkos::subview(
             Neighbors.SendBufferH,
             Kokkos::make_pair(BufferOffset, BufferOffset + BufferSize));
         for (int IExch = 0; IExch < LocList.NTotH[CurNeighbor]; ++IExch) {
            for (int J = 0; J < NJ; ++J) {
               const I4 IBuff    = IExch * NJ + J;
               const ValType Val = Array(LocIndexH(IExch), J);
               R8 RVal;
               memcpy(&RVal, &Val, sizeof(ValType));
               LocBuffH(IBuff) = RVal;
            }
         }
      }
   }

   template <typename T>
   std::enable_if_t<ArrayRank<T>::Is3D>
   packBuffer(const T &Array,       // 3D Kokkos array of any type
              const I4 CurNeighbor, // current neighbor
              const I4 CurElem      // current element
   ) {

      using ValType = typename T::non_const_value_type;

      OMEGA_SCOPE(LocList, Neighbors.SendLists[CurElem]);

      const I4 NJ = Array.extent(2);
      const I4 NK = Array.extent(0);

      const I4 NTotList     = LocList.NTotH[CurNeighbor];
      const I4 BufferSize   = LocList.NTotH[CurNeighbor] * totalSize(Array);
      const I4 BufferOffset = Neighbors.SendBufferOffsetH(CurNeighbor);

      if (devBufferPUP(Array)) {
         const auto IndexBacking  = LocList.IndexBacking;
         const auto IndexOffset   = LocList.IndexOffset;
         const auto Buffer        = Neighbors.SendBuffer;
         const auto BufferOffsets = Neighbors.SendBufferOffset;

         parallelFor(
             {NK, NTotList, NJ}, KOKKOS_LAMBDA(int K, int IExch, int J) {
                const auto LocIndex = makeArraySubview(
                    IndexBacking, IndexOffset(CurNeighbor), NTotList);
                const auto LocBuff = makeArraySubview(
                    Buffer, BufferOffsets(CurNeighbor), BufferSize);
                const ValType Val = Array(K, LocIndex(IExch), J);
                R8 RVal;
                memcpy(&RVal, &Val, sizeof(ValType));
                const I4 IBuff = (K * NTotList + IExch) * NJ + J;
                LocBuff(IBuff) = RVal;
             });
      } else {
         auto LocIndexH = LocList.hostIndices(CurNeighbor);
         auto LocBuffH  = Kokkos::subview(
             Neighbors.SendBufferH,
             Kokkos::make_pair(BufferOffset, BufferOffset + BufferSize));
         for (int K = 0; K < NK; ++K) {
            for (int IExch = 0; IExch < LocList.NTotH[CurNeighbor]; ++IExch) {
               for (int J = 0; J < NJ; ++J) {
                  const I4 IBuff =
                      (K * LocList.NTotH[CurNeighbor] + IExch) * NJ + J;
                  const ValType Val = Array(K, LocIndexH(IExch), J);
                  R8 RVal;
                  memcpy(&RVal, &Val, sizeof(ValType));
                  LocBuffH(IBuff) = RVal;
               }
            }
         }
      }
   }

   template <typename T>
   std::enable_if_t<ArrayRank<T>::Is4D>
   packBuffer(const T &Array,       // 4D Kokkos array of any type
              const I4 CurNeighbor, // current neighbor
              const I4 CurElem      // current element
   ) {

      using ValType = typename T::non_const_value_type;

      OMEGA_SCOPE(LocList, Neighbors.SendLists[CurElem]);

      const I4 NJ = Array.extent(3);
      const I4 NK = Array.extent(1);
      const I4 NL = Array.extent(0);

      const I4 NTotList     = LocList.NTotH[CurNeighbor];
      const I4 BufferSize   = LocList.NTotH[CurNeighbor] * totalSize(Array);
      const I4 BufferOffset = Neighbors.SendBufferOffsetH(CurNeighbor);

      if (devBufferPUP(Array)) {
         const auto IndexBacking  = LocList.IndexBacking;
         const auto IndexOffset   = LocList.IndexOffset;
         const auto Buffer        = Neighbors.SendBuffer;
         const auto BufferOffsets = Neighbors.SendBufferOffset;

         parallelFor(
             {NL, NK, NTotList, NJ},
             KOKKOS_LAMBDA(int L, int K, int IExch, int J) {
                const auto LocIndex = makeArraySubview(
                    IndexBacking, IndexOffset(CurNeighbor), NTotList);
                const auto LocBuff = makeArraySubview(
                    Buffer, BufferOffsets(CurNeighbor), BufferSize);
                const ValType Val = Array(L, K, LocIndex(IExch), J);
                R8 RVal;
                memcpy(&RVal, &Val, sizeof(ValType));
                const I4 IBuff = ((L * NK + K) * NTotList + IExch) * NJ + J;
                LocBuff(IBuff) = RVal;
             });
      } else {
         auto LocIndexH = LocList.hostIndices(CurNeighbor);
         auto LocBuffH  = Kokkos::subview(
             Neighbors.SendBufferH,
             Kokkos::make_pair(BufferOffset, BufferOffset + BufferSize));
         for (int L = 0; L < NL; ++L) {
            for (int K = 0; K < NK; ++K) {
               for (int IExch = 0; IExch < LocList.NTotH[CurNeighbor];
                    ++IExch) {
                  for (int J = 0; J < NJ; ++J) {
                     const I4 IBuff =
                         ((L * NK + K) * NTotList + IExch) * NJ + J;
                     const ValType Val = Array(L, K, LocIndexH(IExch), J);
                     R8 RVal;
                     memcpy(&RVal, &Val, sizeof(ValType));
                     LocBuffH(IBuff) = RVal;
                  }
               }
            }
         }
      }
   }

   template <typename T>
   std::enable_if_t<ArrayRank<T>::Is5D>
   packBuffer(const T &Array,       // 5D Kokkos array of any type
              const I4 CurNeighbor, // current neighbor
              const I4 CurElem      // current element
   ) {

      using ValType = typename T::non_const_value_type;

      OMEGA_SCOPE(LocList, Neighbors.SendLists[CurElem]);

      const I4 NJ = Array.extent(4);
      const I4 NK = Array.extent(2);
      const I4 NL = Array.extent(1);
      const I4 NM = Array.extent(0);

      const I4 NTotList     = LocList.NTotH[CurNeighbor];
      const I4 BufferSize   = LocList.NTotH[CurNeighbor] * totalSize(Array);
      const I4 BufferOffset = Neighbors.SendBufferOffsetH(CurNeighbor);

      if (devBufferPUP(Array)) {
         const auto IndexBacking  = LocList.IndexBacking;
         const auto IndexOffset   = LocList.IndexOffset;
         const auto Buffer        = Neighbors.SendBuffer;
         const auto BufferOffsets = Neighbors.SendBufferOffset;

         parallelFor(
             {NM, NL, NK, NTotList, NJ},
             KOKKOS_LAMBDA(int M, int L, int K, int IExch, int J) {
                const auto LocIndex = makeArraySubview(
                    IndexBacking, IndexOffset(CurNeighbor), NTotList);
                const auto LocBuff = makeArraySubview(
                    Buffer, BufferOffsets(CurNeighbor), BufferSize);
                const ValType Val = Array(M, L, K, LocIndex(IExch), J);
                R8 RVal;
                memcpy(&RVal, &Val, sizeof(ValType));
                const I4 IBuff =
                    (((M * NL + L) * NK + K) * NTotList + IExch) * NJ + J;
                LocBuff(IBuff) = RVal;
             });
      } else {
         auto LocIndexH = LocList.hostIndices(CurNeighbor);
         auto LocBuffH  = Kokkos::subview(
             Neighbors.SendBufferH,
             Kokkos::make_pair(BufferOffset, BufferOffset + BufferSize));
         for (int M = 0; M < NM; ++M) {
            for (int L = 0; L < NL; ++L) {
               for (int K = 0; K < NK; ++K) {
                  for (int IExch = 0; IExch < LocList.NTotH[CurNeighbor];
                       ++IExch) {
                     for (int J = 0; J < NJ; ++J) {
                        const I4 IBuff =
                            (((M * NL + L) * NK + K) * NTotList + IExch) * NJ +
                            J;
                        const ValType Val = Array(M, L, K, LocIndexH(IExch), J);
                        R8 RVal;
                        memcpy(&RVal, &Val, sizeof(ValType));
                        LocBuffH(IBuff) = RVal;
                     }
                  }
               }
            }
         }
      }
   }

   /// Pack all neighbors and exchange indices in parallel when the input array
   /// is on the device. This overload is not used by the halo exchange yet.
   template <typename T>
   void packBufferNbPar(const T &Array, const I4 CurElem, const I4 MaxNTot) {
      static_assert(ArrayRank<T>::Is1D || ArrayRank<T>::Is2D ||
                        ArrayRank<T>::Is3D || ArrayRank<T>::Is4D ||
                        ArrayRank<T>::Is5D,
                    "Unsupported array rank for Halo::packBuffer");

      if (!devBufferPUP(Array)) {
         for (I4 INghbr = 0; INghbr < NNghbr; ++INghbr) {
            if (SendFlagsH[CurElem](INghbr)) {
               packBuffer(Array, INghbr, CurElem);
            }
         }
         return;
      }

      OMEGA_SCOPE(LocList, Neighbors.SendLists[CurElem]);

      if (NNghbr == 0 || MaxNTot == 0) {
         return;
      }

      using ValType            = typename T::non_const_value_type;
      const auto IndexBacking  = LocList.IndexBacking;
      const auto IndexOffsets  = LocList.IndexOffset;
      const auto NTot          = LocList.NTot;
      const auto Buffer        = Neighbors.SendBuffer;
      const auto BufferOffsets = Neighbors.SendBufferOffset;
      const auto SendActive    = SendFlags[CurElem];
      if constexpr (ArrayRank<T>::Is1D) {
         parallelFor(
             {NNghbr, MaxNTot}, KOKKOS_LAMBDA(int INghbr, int IExch) {
                const I4 NTotList = NTot(INghbr);
                if (SendActive(INghbr) == 0 || IExch >= NTotList) {
                   return;
                }
                const auto LocIndex = makeArraySubview(
                    IndexBacking, IndexOffsets(INghbr), NTotList);
                const auto LocBuff =
                    makeArraySubview(Buffer, BufferOffsets(INghbr), NTotList);
                const ValType Val = Array(LocIndex(IExch));
                R8 RVal;
                memcpy(&RVal, &Val, sizeof(ValType));
                LocBuff(IExch) = RVal;
             });
      } else if constexpr (ArrayRank<T>::Is2D) {
         const I4 NJ = Array.extent(1);
         parallelFor(
             {NNghbr, MaxNTot, NJ},
             KOKKOS_LAMBDA(int INghbr, int IExch, int J) {
                const I4 NTotList = NTot(INghbr);
                if (SendActive(INghbr) == 0 || IExch >= NTotList) {
                   return;
                }
                const auto LocIndex = makeArraySubview(
                    IndexBacking, IndexOffsets(INghbr), NTotList);
                const auto LocBuff = makeArraySubview(
                    Buffer, BufferOffsets(INghbr), NTotList * NJ);
                const ValType Val = Array(LocIndex(IExch), J);
                R8 RVal;
                memcpy(&RVal, &Val, sizeof(ValType));
                LocBuff(IExch * NJ + J) = RVal;
             });
      } else if constexpr (ArrayRank<T>::Is3D) {
         const I4 NK = Array.extent(0);
         const I4 NJ = Array.extent(2);
         parallelFor(
             {NNghbr, NK, MaxNTot, NJ},
             KOKKOS_LAMBDA(int INghbr, int K, int IExch, int J) {
                const I4 NTotList = NTot(INghbr);
                if (SendActive(INghbr) == 0 || IExch >= NTotList) {
                   return;
                }
                const auto LocIndex = makeArraySubview(
                    IndexBacking, IndexOffsets(INghbr), NTotList);
                const auto LocBuff = makeArraySubview(
                    Buffer, BufferOffsets(INghbr), NTotList * NK * NJ);
                const ValType Val = Array(K, LocIndex(IExch), J);
                R8 RVal;
                memcpy(&RVal, &Val, sizeof(ValType));
                LocBuff((K * NTotList + IExch) * NJ + J) = RVal;
             });
      } else if constexpr (ArrayRank<T>::Is4D) {
         const I4 NL = Array.extent(0);
         const I4 NK = Array.extent(1);
         const I4 NJ = Array.extent(3);
         parallelFor(
             {NNghbr, NL, NK, MaxNTot, NJ},
             KOKKOS_LAMBDA(int INghbr, int L, int K, int IExch, int J) {
                const I4 NTotList = NTot(INghbr);
                if (SendActive(INghbr) == 0 || IExch >= NTotList) {
                   return;
                }
                const auto LocIndex = makeArraySubview(
                    IndexBacking, IndexOffsets(INghbr), NTotList);
                const auto LocBuff = makeArraySubview(
                    Buffer, BufferOffsets(INghbr), NTotList * NL * NK * NJ);
                const ValType Val = Array(L, K, LocIndex(IExch), J);
                R8 RVal;
                memcpy(&RVal, &Val, sizeof(ValType));
                LocBuff(((L * NK + K) * NTotList + IExch) * NJ + J) = RVal;
             });
      } else if constexpr (ArrayRank<T>::Is5D) {
         const I4 NM = Array.extent(0);
         const I4 NL = Array.extent(1);
         const I4 NK = Array.extent(2);
         const I4 NJ = Array.extent(4);
         parallelFor(
             {NNghbr, NM, NL, NK, MaxNTot, NJ},
             KOKKOS_LAMBDA(int INghbr, int M, int L, int K, int IExch, int J) {
                const I4 NTotList = NTot(INghbr);
                if (SendActive(INghbr) == 0 || IExch >= NTotList) {
                   return;
                }
                const auto LocIndex = makeArraySubview(
                    IndexBacking, IndexOffsets(INghbr), NTotList);
                const auto LocBuff =
                    makeArraySubview(Buffer, BufferOffsets(INghbr),
                                     NTotList * NM * NL * NK * NJ);
                const ValType Val = Array(M, L, K, LocIndex(IExch), J);
                R8 RVal;
                memcpy(&RVal, &Val, sizeof(ValType));
                LocBuff((((M * NL + L) * NK + K) * NTotList + IExch) * NJ + J) =
                    RVal;
             });
      }
   }

   /// Buffer unpack specialized function templates for supported Kokkos array
   /// ranks. After receiving a message from a neighboring task, save the
   /// elements of the proper receive buffer for that Neighbor into the
   /// corresponding halo elements of the input Array
   template <typename T>
   std::enable_if_t<ArrayRank<T>::Is1D>
   unpackBuffer(const T &Array,       // 1D Kokkos array of any type
                const I4 CurNeighbor, // current neighbor
                const I4 CurElem      // current element
   ) {

      using ValType = typename T::non_const_value_type;

      OMEGA_SCOPE(LocList, Neighbors.RecvLists[CurElem]);
      const I4 NTotList     = LocList.NTotH[CurNeighbor];
      const I4 BufferSize   = NTotList;
      const I4 BufferOffset = Neighbors.RecvBufferOffsetH(CurNeighbor);

      if (devBufferPUP(Array)) {
         const auto IndexBacking  = LocList.IndexBacking;
         const auto IndexOffset   = LocList.IndexOffset;
         const auto Buffer        = Neighbors.RecvBuffer;
         const auto BufferOffsets = Neighbors.RecvBufferOffset;
         parallelFor(
             {NTotList}, KOKKOS_LAMBDA(int IExch) {
                const auto LocIndex = makeArraySubview(
                    IndexBacking, IndexOffset(CurNeighbor), NTotList);
                const auto LocBuff = makeArraySubview(
                    Buffer, BufferOffsets(CurNeighbor), BufferSize);
                const I4 IArr = LocIndex(IExch);
                const R8 RVal = LocBuff(IExch);
                ValType Val;
                memcpy(&Val, &RVal, sizeof(ValType));
                Array(IArr) = Val;
             });
      } else {
         auto LocIndexH = LocList.hostIndices(CurNeighbor);
         auto LocBuffH  = Kokkos::subview(
             Neighbors.RecvBufferH,
             Kokkos::make_pair(BufferOffset, BufferOffset + BufferSize));
         for (int IExch = 0; IExch < LocList.NTotH[CurNeighbor]; ++IExch) {
            const I4 IArr = LocIndexH(IExch);
            const R8 RVal = LocBuffH(IExch);
            ValType Val;
            memcpy(&Val, &RVal, sizeof(ValType));
            Array(IArr) = Val;
         }
      }
   }

   template <typename T>
   std::enable_if_t<ArrayRank<T>::Is2D>
   unpackBuffer(const T &Array,       // 2D Kokkos array of any type
                const I4 CurNeighbor, // current neighbor
                const I4 CurElem      // current element
   ) {

      using ValType = typename T::non_const_value_type;

      OMEGA_SCOPE(LocList, Neighbors.RecvLists[CurElem]);

      const I4 NJ           = Array.extent(1);
      const I4 NTotList     = LocList.NTotH[CurNeighbor];
      const I4 BufferSize   = NTotList * totalSize(Array);
      const I4 BufferOffset = Neighbors.RecvBufferOffsetH(CurNeighbor);

      if (devBufferPUP(Array)) {
         const auto IndexBacking  = LocList.IndexBacking;
         const auto IndexOffset   = LocList.IndexOffset;
         const auto Buffer        = Neighbors.RecvBuffer;
         const auto BufferOffsets = Neighbors.RecvBufferOffset;

         parallelFor(
             {NTotList, NJ}, KOKKOS_LAMBDA(int IExch, int J) {
                const auto LocIndex = makeArraySubview(
                    IndexBacking, IndexOffset(CurNeighbor), NTotList);
                const auto LocBuff = makeArraySubview(
                    Buffer, BufferOffsets(CurNeighbor), BufferSize);
                const I4 IBuff = IExch * NJ + J;
                const I4 IArr  = LocIndex(IExch);
                const R8 RVal  = LocBuff(IBuff);
                ValType Val;
                memcpy(&Val, &RVal, sizeof(ValType));
                Array(IArr, J) = Val;
             });
      } else {
         auto LocIndexH = LocList.hostIndices(CurNeighbor);
         auto LocBuffH  = Kokkos::subview(
             Neighbors.RecvBufferH,
             Kokkos::make_pair(BufferOffset, BufferOffset + BufferSize));
         for (int IExch = 0; IExch < LocList.NTotH[CurNeighbor]; ++IExch) {
            for (int J = 0; J < NJ; ++J) {
               const I4 IBuff = IExch * NJ + J;
               const I4 IArr  = LocIndexH(IExch);
               const R8 RVal  = LocBuffH(IBuff);
               ValType Val;
               memcpy(&Val, &RVal, sizeof(ValType));
               Array(IArr, J) = Val;
            }
         }
      }
   }

   template <typename T>
   std::enable_if_t<ArrayRank<T>::Is3D>
   unpackBuffer(const T &Array,       // 3D Kokkos array of any type
                const I4 CurNeighbor, // current neighbor
                const I4 CurElem      // current element
   ) {

      using ValType = typename T::non_const_value_type;

      OMEGA_SCOPE(LocList, Neighbors.RecvLists[CurElem]);

      const I4 NJ = Array.extent(2);
      const I4 NK = Array.extent(0);

      const I4 NTotList     = LocList.NTotH[CurNeighbor];
      const I4 BufferSize   = NTotList * totalSize(Array);
      const I4 BufferOffset = Neighbors.RecvBufferOffsetH(CurNeighbor);

      if (devBufferPUP(Array)) {
         const auto IndexBacking  = LocList.IndexBacking;
         const auto IndexOffset   = LocList.IndexOffset;
         const auto Buffer        = Neighbors.RecvBuffer;
         const auto BufferOffsets = Neighbors.RecvBufferOffset;

         parallelFor(
             {NK, NTotList, NJ}, KOKKOS_LAMBDA(int K, int IExch, int J) {
                const auto LocIndex = makeArraySubview(
                    IndexBacking, IndexOffset(CurNeighbor), NTotList);
                const auto LocBuff = makeArraySubview(
                    Buffer, BufferOffsets(CurNeighbor), BufferSize);
                const I4 IBuff = (K * NTotList + IExch) * NJ + J;
                const I4 IArr  = LocIndex(IExch);
                const R8 RVal  = LocBuff(IBuff);
                ValType Val;
                memcpy(&Val, &RVal, sizeof(ValType));
                Array(K, IArr, J) = Val;
             });
      } else {
         auto LocIndexH = LocList.hostIndices(CurNeighbor);
         auto LocBuffH  = Kokkos::subview(
             Neighbors.RecvBufferH,
             Kokkos::make_pair(BufferOffset, BufferOffset + BufferSize));
         for (int K = 0; K < NK; ++K) {
            for (int IExch = 0; IExch < LocList.NTotH[CurNeighbor]; ++IExch) {
               for (int J = 0; J < NJ; ++J) {
                  const I4 IBuff =
                      (K * LocList.NTotH[CurNeighbor] + IExch) * NJ + J;
                  const I4 IArr = LocIndexH(IExch);
                  const R8 RVal = LocBuffH(IBuff);
                  ValType Val;
                  memcpy(&Val, &RVal, sizeof(ValType));
                  Array(K, IArr, J) = Val;
               }
            }
         }
      }
   }

   template <typename T>
   std::enable_if_t<ArrayRank<T>::Is4D>
   unpackBuffer(const T &Array,       // 4D Kokkos array of any type
                const I4 CurNeighbor, // current neighbor
                const I4 CurElem      // current element
   ) {

      using ValType = typename T::non_const_value_type;

      OMEGA_SCOPE(LocList, Neighbors.RecvLists[CurElem]);

      const I4 NJ = Array.extent(3);
      const I4 NK = Array.extent(1);
      const I4 NL = Array.extent(0);

      const I4 NTotList     = LocList.NTotH[CurNeighbor];
      const I4 BufferSize   = NTotList * totalSize(Array);
      const I4 BufferOffset = Neighbors.RecvBufferOffsetH(CurNeighbor);

      if (devBufferPUP(Array)) {
         const auto IndexBacking  = LocList.IndexBacking;
         const auto IndexOffset   = LocList.IndexOffset;
         const auto Buffer        = Neighbors.RecvBuffer;
         const auto BufferOffsets = Neighbors.RecvBufferOffset;

         parallelFor(
             {NL, NK, NTotList, NJ},
             KOKKOS_LAMBDA(int L, int K, int IExch, int J) {
                const auto LocIndex = makeArraySubview(
                    IndexBacking, IndexOffset(CurNeighbor), NTotList);
                const auto LocBuff = makeArraySubview(
                    Buffer, BufferOffsets(CurNeighbor), BufferSize);
                const I4 IBuff = ((L * NK + K) * NTotList + IExch) * NJ + J;
                const I4 IArr  = LocIndex(IExch);
                const R8 RVal  = LocBuff(IBuff);
                ValType Val;
                memcpy(&Val, &RVal, sizeof(ValType));
                Array(L, K, IArr, J) = Val;
             });
      } else {
         auto LocIndexH = LocList.hostIndices(CurNeighbor);
         auto LocBuffH  = Kokkos::subview(
             Neighbors.RecvBufferH,
             Kokkos::make_pair(BufferOffset, BufferOffset + BufferSize));
         for (int L = 0; L < NL; ++L) {
            for (int K = 0; K < NK; ++K) {
               for (int IExch = 0; IExch < LocList.NTotH[CurNeighbor];
                    ++IExch) {
                  for (int J = 0; J < NJ; ++J) {
                     const I4 IBuff =
                         ((L * NK + K) * NTotList + IExch) * NJ + J;
                     const I4 IArr = LocIndexH(IExch);
                     const R8 RVal = LocBuffH(IBuff);
                     ValType Val;
                     memcpy(&Val, &RVal, sizeof(ValType));
                     Array(L, K, IArr, J) = Val;
                  }
               }
            }
         }
      }
   }

   template <typename T>
   std::enable_if_t<ArrayRank<T>::Is5D>
   unpackBuffer(const T &Array,       // 5D Kokkos array of any type
                const I4 CurNeighbor, // current neighbor
                const I4 CurElem      // current element
   ) {

      using ValType = typename T::non_const_value_type;

      OMEGA_SCOPE(LocList, Neighbors.RecvLists[CurElem]);

      const I4 NJ = Array.extent(4);
      const I4 NK = Array.extent(2);
      const I4 NL = Array.extent(1);
      const I4 NM = Array.extent(0);

      const I4 NTotList     = LocList.NTotH[CurNeighbor];
      const I4 BufferSize   = NTotList * totalSize(Array);
      const I4 BufferOffset = Neighbors.RecvBufferOffsetH(CurNeighbor);

      if (devBufferPUP(Array)) {
         const auto IndexBacking  = LocList.IndexBacking;
         const auto IndexOffset   = LocList.IndexOffset;
         const auto Buffer        = Neighbors.RecvBuffer;
         const auto BufferOffsets = Neighbors.RecvBufferOffset;

         parallelFor(
             {NM, NL, NK, NTotList, NJ},
             KOKKOS_LAMBDA(int M, int L, int K, int IExch, int J) {
                const auto LocIndex = makeArraySubview(
                    IndexBacking, IndexOffset(CurNeighbor), NTotList);
                const auto LocBuff = makeArraySubview(
                    Buffer, BufferOffsets(CurNeighbor), BufferSize);
                const I4 IBuff =
                    (((M * NL + L) * NK + K) * NTotList + IExch) * NJ + J;
                const I4 IArr = LocIndex(IExch);
                const R8 RVal = LocBuff(IBuff);
                ValType Val;
                memcpy(&Val, &RVal, sizeof(ValType));
                Array(M, L, K, IArr, J) = Val;
             });
      } else {
         auto LocIndexH = LocList.hostIndices(CurNeighbor);
         auto LocBuffH  = Kokkos::subview(
             Neighbors.RecvBufferH,
             Kokkos::make_pair(BufferOffset, BufferOffset + BufferSize));
         for (int M = 0; M < NM; ++M) {
            for (int L = 0; L < NL; ++L) {
               for (int K = 0; K < NK; ++K) {
                  for (int IExch = 0; IExch < LocList.NTotH[CurNeighbor];
                       ++IExch) {
                     for (int J = 0; J < NJ; ++J) {
                        const I4 IBuff =
                            (((M * NL + L) * NK + K) * NTotList + IExch) * NJ +
                            J;
                        const I4 IArr = LocIndexH(IExch);
                        const R8 RVal = LocBuffH(IBuff);
                        ValType Val;
                        memcpy(&Val, &RVal, sizeof(ValType));
                        Array(M, L, K, IArr, J) = Val;
                     }
                  }
               }
            }
         }
      }
   }

   template <typename T>
   void packAllBuffers(T &Array,           // Kokkos array of any type
                       MeshElement CurElem // index space Array is defined on
   ) {
      Pacer::start("Halo:packBuffers", 4);
      for (int INghbr = 0; INghbr < NNghbr; ++INghbr) {
         Neighbors.Received[INghbr] = false;
         Neighbors.Unpacked[INghbr] = false;
         if (SendFlagsH[CurElem](INghbr)) {
            packBuffer(Array, INghbr, CurElem);
         }
      }
      Pacer::stop("Halo:packBuffers", 4);
   }

   template <typename T>
   void
   packAllBuffersNbPar(T &Array,           // Kokkos array of any type
                       MeshElement CurElem // index space Array is defined on
   ) {
      Pacer::start("Halo:packBuffers", 4);

      I4 MaxNTot = 0;
      for (int INghbr = 0; INghbr < NNghbr; ++INghbr) {
         Neighbors.Received[INghbr] = false;
         Neighbors.Unpacked[INghbr] = false;

         const auto &SendList = Neighbors.SendLists[CurElem];
         MaxNTot              = std::max(MaxNTot, SendList.NTotH[INghbr]);
      }
      packBufferNbPar(Array, CurElem, MaxNTot);
      Pacer::stop("Halo:packBuffers", 4);
   }

   //---------------------------------------------------------------------------
   // Function template to perform a full halo exchange on the input Kokkos
   // array of any supported type defined on the input index space ThisElem
   template <typename T>
   int
   exchangeFullArrayHalo(T &Array,           // Kokkos array of any type
                         MeshElement CurElem // index space Array is defined on
   ) {
      // Add more context for timers in this function
      // BUG: fails when there is no parent timer
      // Uncomment after fixing in Pacer
      // Pacer::addParentPrefix();

      I4 IErr{0}; // error code

      // Logical flag to track if all messages have been received
      bool AllReceived{false};

      // If the Array is in device memory space, the buffer pack and unpack
      // functions will use device buffers
      bool UseDevBuffer = devBufferPUP(Array);

      // Determine the contiguous buffer offsets and sizes for all neighbors
      I4 SendBufferSize = 0;
      I4 RecvBufferSize = 0;
      for (int INghbr = 0; INghbr < NNghbr; ++INghbr) {
         Neighbors.SendBufferOffsetH(INghbr) = SendBufferSize;
         Neighbors.RecvBufferOffsetH(INghbr) = RecvBufferSize;
         if (SendFlagsH[CurElem](INghbr)) {
            SendBufferSize +=
                totalSize(Array) * Neighbors.SendLists[CurElem].NTotH[INghbr];
         }
         if (RecvFlagsH[CurElem](INghbr)) {
            RecvBufferSize +=
                totalSize(Array) * Neighbors.RecvLists[CurElem].NTotH[INghbr];
         }
      }
      deepCopy(Neighbors.SendBufferOffset, Neighbors.SendBufferOffsetH);
      deepCopy(Neighbors.RecvBufferOffset, Neighbors.RecvBufferOffsetH);

      // Grow all buffers before starting communication or packing
      expandBuffer(Neighbors.SendBuffer, SendBufferSize);
      expandBuffer(Neighbors.SendBufferH, SendBufferSize);
      expandBuffer(Neighbors.RecvBuffer, RecvBufferSize);
      expandBuffer(Neighbors.RecvBufferH, RecvBufferSize);

      // Call MPI_Irecv for each Neighbor
      // so the local task is ready to accept messages from each
      // neighboring task
      startReceives(UseDevBuffer, totalSize(Array), CurElem);

      // packAllBuffers(Array, CurElem);
      packAllBuffersNbPar(Array, CurElem);

      // Call MPI_Isend for each Neighbor to send the packed buffers
      startSends(UseDevBuffer, totalSize(Array), CurElem);

      // Collect all send requests
      std::vector<MPI_Request> SendReqs;
      for (int INghbr = 0; INghbr < NNghbr; ++INghbr) {
         if (SendFlagsH[CurElem](INghbr)) {
            SendReqs.push_back(Neighbors.SReq[INghbr]);
         }
      }

      I4 MaxIter = 1000000000; // Large integer to prevent infinite loop
      I4 IPass   = 0;          // Number of passes through while loop
      I4 NRcvd   = 0;          // Integer to track number of messages received

      // Total number of messages the local task will receive
      I4 NMessages = 0;
      for (I4 INghbr = 0; INghbr < NNghbr; ++INghbr) {
         NMessages += RecvFlagsH[CurElem](INghbr);
      }

      // Until all messages from neighboring tasks are received, loop
      // through Neighbor objects and use MPI_Test to check if the message
      // has been received. Unpack buffers upon receipt of each message
      Pacer::start("Halo:receiveUnpack", 4);
      while (!AllReceived) {
         for (int INghbr = 0; INghbr < NNghbr; ++INghbr) {
            if (RecvFlagsH[CurElem](INghbr)) {
               if (!Neighbors.Received[INghbr]) {
                  MPI_Test(&Neighbors.RReq[INghbr], &Neighbors.Received[INghbr],
                           MPI_STATUS_IGNORE);
                  if (Neighbors.Received[INghbr]) {
                     ++NRcvd;
                  }
               }
               if (Neighbors.Received[INghbr] && !Neighbors.Unpacked[INghbr]) {
                  // If the device buffer will be used in unpackBuffer, but the
                  // exchange was done with the host buffer, a deep copy from
                  // host to device is needed.
                  if (UseDevBuffer && !ExchOnDev) {
                     // The number of elements we need to copy is different
                     // than the buffer allocation size
                     const I4 BufferSize =
                         totalSize(Array) *
                         Neighbors.RecvLists[CurElem].NTotH[INghbr];
                     const I4 Offset = Neighbors.RecvBufferOffsetH(INghbr);
                     const auto CopyRange =
                         Kokkos::make_pair(Offset, Offset + BufferSize);

                     deepCopy(
                         Kokkos::subview(Neighbors.RecvBuffer, CopyRange),
                         Kokkos::subview(Neighbors.RecvBufferH, CopyRange));
                  }
                  unpackBuffer(Array, INghbr, CurElem);
                  Neighbors.Unpacked[INghbr] = true;
               }
            }
         }

         if (NRcvd == NMessages) {
            AllReceived = true;
         }
         ++IPass;
         if (IPass == MaxIter) {
            LOG_ERROR("Halo: Maximum iterations reached during halo exchange");
            IErr = -1;
            break;
         }
      }
      Pacer::stop("Halo:receiveUnpack", 4);

      // We need to fence here because we are reusing buffers and GPU work is
      // async
      if (UseDevBuffer) {
         Kokkos::fence();
      }

      Pacer::start("Halo:waitSends", 4);
      // Wait for all sends to complete before proceeding
      MPI_Waitall(SendReqs.size(), SendReqs.data(), MPI_STATUS_IGNORE);
      Pacer::stop("Halo:waitSends", 4);

      // BUG: fails when there is no parent timer
      // Uncomment after fixing in Pacer
      // Pacer::removeParentPrefix();

      return IErr;
   } // end exchangeFullArrayHalo

}; // end class Halo

} // end namespace OMEGA

//===----------------------------------------------------------------------===//
#endif // defined OMEGA_HALO_H
