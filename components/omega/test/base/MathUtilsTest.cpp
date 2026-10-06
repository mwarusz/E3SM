//===-- Test driver for OMEGA math utilities -------------------------*- C++
//-*-===//
///
/// \file
/// \brief Unit tests for mathematical functions and utilities
///
//===------------------------------------------------------------------===//

#include "MathUtils.h"
#include "Error.h"
#include "Logging.h"
#include "MachEnv.h"
#include "OmegaKokkos.h"

#include "mpi.h"

using namespace OMEGA;

// We can test some stuff at compile time

// pow tests
static_assert(Math::pow<0>(2) == 1);
static_assert(Math::pow<1>(2) == 2);
static_assert(Math::pow<2>(2) == 4);
static_assert(Math::pow<10>(2) == 1024);

static_assert(Math::pow<2>(0.5_Real) == 0.25_Real);
static_assert(Math::pow<3>(0.5_Real) == 0.125_Real);
static_assert(Math::pow<-1>(0.5_Real) == 2.0_Real);

// min tests
static_assert(Math::min(1, 2) == 1);
static_assert(Math::min(2, 1) == 1);
static_assert(Math::min(2, -1, 0) == -1);

// max tests
static_assert(Math::max(1, 2) == 2);
static_assert(Math::max(2, 1) == 2);
static_assert(Math::max(2, -1, 0, 3) == 3);

Error testMathUtilsHost() {
   Error Err;

   if (Math::pow<2>(3) != 9) {
      Err += Error(ErrorCode::Fail, "host pow<2>(3) FAIL");
   }

   if (Math::pow<3>(0.25_Real) != 0.015625_Real) {
      Err += Error(ErrorCode::Fail, "host pow<3>(0.25_Real) FAIL");
   }

   if (Math::pow<-2>(2.0_Real) != 0.25_Real) {
      Err += Error(ErrorCode::Fail, "host pow<-2>(2.0_Real) FAIL");
   }

   if (Math::min(2, 3, 1) != 1) {
      Err += Error(ErrorCode::Fail, "host min(2, 3, 1) FAIL");
   }

   if (Math::max(2, 3, 1) != 3) {
      Err += Error(ErrorCode::Fail, "host max(2, 3, 1) FAIL");
   }

   return Err;
}

Error testMathUtilsDevice() {
   Error Err;

   // Bit mask used to communicate device results to the host
   // A bit is set when the corresponding test passed
   I8 DeviceResults;

   parallelReduce(
       {1},
       KOKKOS_LAMBDA(int, I8 &Accum) {
          if (Math::pow<2>(3) != 9) {
             Accum |= (1 << 0);
          }

          if (Math::pow<3>(0.25_Real) != 0.015625_Real) {
             Accum |= (1 << 1);
          }

          if (Math::pow<-2>(2.0_Real) != 0.25_Real) {
             Accum |= (1 << 2);
          }

          if (Math::min(2, 3, 1) != 1) {
             Accum |= (1 << 3);
          }

          if (Math::max(2, 3, 1) != 3) {
             Accum |= (1 << 4);
          }
       },
       DeviceResults);

   if (DeviceResults & (1 << 0)) {
      Err += Error(ErrorCode::Fail, "device pow<2>(3) FAIL");
   }

   if (DeviceResults & (1 << 1)) {
      Err += Error(ErrorCode::Fail, "device pow<3>(0.25_Real) FAIL");
   }

   if (DeviceResults & (1 << 2)) {
      Err += Error(ErrorCode::Fail, "device pow<-2>(2.0_Real) FAIL");
   }

   if (DeviceResults & (1 << 3)) {
      Err += Error(ErrorCode::Fail, "device min(2, 3, 1) FAIL");
   }

   if (DeviceResults & (1 << 4)) {
      Err += Error(ErrorCode::Fail, "device max(2, 3, 1) FAIL");
   }

   return Err;
}

int main(int argc, char *argv[]) {
   Error Err;

   MPI_Init(&argc, &argv);
   MachEnv::init(MPI_COMM_WORLD);
   MachEnv *DefEnv = MachEnv::getDefault();
   initLogging(DefEnv);

   try {
      Kokkos::initialize(argc, argv);

      Err += testMathUtilsHost();
      Err += testMathUtilsDevice();

      Kokkos::finalize();
   } catch (const std::exception &Ex) {
      Err += Error(ErrorCode::Fail, Ex.what() + std::string(": FAIL"));
   } catch (...) {
      Err += Error(ErrorCode::Fail, "Unknown: FAIL");
   }

   CHECK_ERROR_ABORT(Err, "Kokkos Wrappers Unit Tests FAIL");

   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();

   return 0;
}
