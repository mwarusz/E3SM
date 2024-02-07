#include "Operators.h"
#include "DataTypes.h"
#include "Decomp.h"
#include "Halo.h"
#include "HorzMesh.h"
#include "IO.h"
#include "Logging.h"
#include "MachEnv.h"
#include "mpi.h"

#include <cmath>
#include <iostream>

using namespace OMEGA;
using yakl::c::parallel_for;

bool isApprox(Real X, Real Y, Real RTol) {
   return std::abs(X - Y) <= RTol * std::max(std::abs(X), std::abs(Y));
}

struct ExactFunctions {
   Real PI = M_PI;
   // TODO: get this from the mesh once we support periodic planar meshes
   Real Lx = 1;
   Real Ly = std::sqrt(3) / 2;

   YAKL_INLINE Real exactScalar(Real X, Real Y) const {
      return std::sin(2 * PI * X / Lx) * std::sin(2 * PI * Y / Ly);
   }

   YAKL_INLINE Real exactGradScalarX(Real X, Real Y) const {
      return 2 * PI / Lx * std::cos(2 * PI * X / Lx) *
             std::sin(2 * PI * Y / Ly);
   }

   YAKL_INLINE Real exactGradScalarY(Real X, Real Y) const {
      return 2 * PI / Ly * std::sin(2 * PI * X / Lx) *
             std::cos(2 * PI * Y / Ly);
   }

   YAKL_INLINE Real exactVecX(Real X, Real Y) const {
      return std::sin(2 * PI * X / Lx) * std::cos(2 * PI * Y / Ly);
   }

   YAKL_INLINE Real exactVecY(Real X, Real Y) const {
      return std::cos(2 * PI * X / Lx) * std::sin(2 * PI * Y / Ly);
   }

   YAKL_INLINE Real exactDivVec(Real X, Real Y) const {
      return 2 * PI * (1. / Lx + 1. / Ly) * std::cos(2 * PI * X / Lx) *
             std::cos(2 * PI * Y / Ly);
   }

   YAKL_INLINE Real exactCurlVec(Real X, Real Y) const {
      return 2 * PI * (-1. / Lx + 1. / Ly) * std::sin(2 * PI * X / Lx) *
             std::sin(2 * PI * Y / Ly);
   }
};

int testDivergence(Real RTol) {
   int Err;
   ExactFunctions EF;

   const auto &mesh = HorzMesh::getDefault();
   auto XEdge       = mesh->XEdgeH.createDeviceCopy();
   auto YEdge       = mesh->YEdgeH.createDeviceCopy();
   auto &AngleEdge  = mesh->AngleEdge;

   // Prepare operator input
   Array1DReal VecEdge("VecEdge", mesh->NEdgesSize);
   parallel_for(
       mesh->NEdgesOwned, YAKL_LAMBDA(int IEdge) {
          const Real X = XEdge(IEdge);
          const Real Y = YEdge(IEdge);

          const Real VecX = EF.exactVecX(X, Y);
          const Real VecY = EF.exactVecY(X, Y);

          const Real EdgeNormalX = std::cos(AngleEdge(IEdge));
          const Real EdgeNormalY = std::sin(AngleEdge(IEdge));

          VecEdge(IEdge) = EdgeNormalX * VecX + EdgeNormalY * VecY;
       });

   // Perform halo exchange
   Halo MyHalo(MachEnv::getDefaultEnv(), Decomp::getDefault());
   auto VecEdgeH = VecEdge.createHostCopy();
   MyHalo.exchangeFullArrayHalo(VecEdgeH, OnEdge);
   VecEdgeH.deep_copy_to(VecEdge);

   auto XCell     = mesh->XCellH.createDeviceCopy();
   auto YCell     = mesh->YCellH.createDeviceCopy();
   auto &AreaCell = mesh->AreaCell;

   // Compute element-wise errors
   Array1DReal LInfCell("LInfCell", mesh->NCellsOwned);
   Array1DReal L2Cell("L2Cell", mesh->NCellsOwned);
   DivergenceOnCell DivergenceCell(mesh);
   parallel_for(
       mesh->NCellsOwned, YAKL_LAMBDA(int ICell) {
          // Numerical result
          const Real DivCellNum = DivergenceCell(ICell, VecEdge);

          // Exact result
          const Real X            = XCell(ICell);
          const Real Y            = YCell(ICell);
          const Real DivCellExact = EF.exactDivVec(X, Y);

          // Errors
          LInfCell(ICell) = std::abs(DivCellNum - DivCellExact);
          L2Cell(ICell)   = AreaCell(ICell) * LInfCell(ICell) * LInfCell(ICell);
       });

   // Compute global error norms
   const Real LInfErrorLoc = yakl::intrinsics::maxval(LInfCell);
   const Real L2ErrorLoc   = yakl::intrinsics::sum(L2Cell);

   MPI_Comm Comm = MachEnv::getDefaultEnv()->getComm();
   Real LInfError;
   Err =
       MPI_Allreduce(&LInfErrorLoc, &LInfError, 1, MPI_RealKind, MPI_MAX, Comm);

   Real L2Error;
   Err = MPI_Allreduce(&L2ErrorLoc, &L2Error, 1, MPI_RealKind, MPI_SUM, Comm);
   L2Error = std::sqrt(L2Error);

   // Check error values
   const Real ExpectedLInfError = 0.016907664729341576;
   const Real ExpectedL2Error   = 0.00786717747637947704;

   if (Err == 0 && isApprox(LInfError, ExpectedLInfError, RTol) &&
       isApprox(L2Error, ExpectedL2Error, RTol)) {
      return 0;
   } else {
      return 1;
   }
}

int testGradient(Real RTol) {
   int Err;
   ExactFunctions EF;

   const auto &mesh = HorzMesh::getDefault();
   const auto XCell = mesh->XCellH.createDeviceCopy();
   const auto YCell = mesh->YCellH.createDeviceCopy();

   // Prepare operator input
   Array1DReal ScalarCell("ScalarCell", mesh->NCellsSize);
   parallel_for(
       mesh->NCellsOwned, YAKL_LAMBDA(int ICell) {
          const Real X      = XCell(ICell);
          const Real Y      = YCell(ICell);
          ScalarCell(ICell) = EF.exactScalar(X, Y);
       });

   // Perform halo exchange
   Halo MyHalo(MachEnv::getDefaultEnv(), Decomp::getDefault());
   auto ScalarCellH = ScalarCell.createHostCopy();
   MyHalo.exchangeFullArrayHalo(ScalarCellH, OnCell);
   ScalarCellH.deep_copy_to(ScalarCell);

   const auto XEdge      = mesh->XEdgeH.createDeviceCopy();
   const auto YEdge      = mesh->YEdgeH.createDeviceCopy();
   const auto &AngleEdge = mesh->AngleEdge;
   const auto &DcEdge    = mesh->DcEdge;
   const auto &DvEdge    = mesh->DvEdge;

   // Compute element-wise errors
   Array1DReal LInfEdge("LInfEdge", mesh->NEdgesOwned);
   Array1DReal L2Edge("L2Edge", mesh->NEdgesOwned);
   GradientOnEdge GradientEdge(mesh);
   parallel_for(
       mesh->NEdgesOwned, YAKL_LAMBDA(int IEdge) {
          // Numerical result
          const Real GradScalarNum = GradientEdge(IEdge, ScalarCell);

          // Exact result
          const Real X                = XEdge(IEdge);
          const Real Y                = YEdge(IEdge);
          const Real GradScalarExactX = EF.exactGradScalarX(X, Y);
          const Real GradScalarExactY = EF.exactGradScalarY(X, Y);
          const Real EdgeNormalX      = std::cos(AngleEdge(IEdge));
          const Real EdgeNormalY      = std::sin(AngleEdge(IEdge));
          const Real GradScalarExact =
              EdgeNormalX * GradScalarExactX + EdgeNormalY * GradScalarExactY;

          // Errors
          LInfEdge(IEdge)     = std::abs(GradScalarNum - GradScalarExact);
          const Real AreaEdge = DcEdge(IEdge) * DvEdge(IEdge) / 2;
          L2Edge(IEdge)       = AreaEdge * LInfEdge(IEdge) * LInfEdge(IEdge);
       });

   // Compute global error norms
   const Real LInfErrorLoc = yakl::intrinsics::maxval(LInfEdge);
   const Real L2ErrorLoc   = yakl::intrinsics::sum(L2Edge);

   MPI_Comm Comm = MachEnv::getDefaultEnv()->getComm();
   Real LInfError;
   Err =
       MPI_Allreduce(&LInfErrorLoc, &LInfError, 1, MPI_RealKind, MPI_MAX, Comm);

   Real L2Error;
   Err = MPI_Allreduce(&L2ErrorLoc, &L2Error, 1, MPI_RealKind, MPI_SUM, Comm);
   L2Error = std::sqrt(L2Error);

   // Check error values
   const Real ExpectedLInfError = 0.0078388002934621781;
   const Real ExpectedL2Error   = 0.00424268862440643248;

   if (Err == 0 && isApprox(LInfError, ExpectedLInfError, RTol) &&
       isApprox(L2Error, ExpectedL2Error, RTol)) {
      return 0;
   } else {
      return 1;
   }
}

int testCurl(Real RTol) {
   int Err;
   ExactFunctions EF;

   const auto &mesh      = HorzMesh::getDefault();
   const auto XEdge      = mesh->XEdgeH.createDeviceCopy();
   const auto YEdge      = mesh->YEdgeH.createDeviceCopy();
   const auto &AngleEdge = mesh->AngleEdge;

   // Prepare operator input
   Array1DReal VecEdge("VecEdge", mesh->NEdgesSize);
   parallel_for(
       mesh->NEdgesOwned, YAKL_LAMBDA(int IEdge) {
          const Real X = XEdge(IEdge);
          const Real Y = YEdge(IEdge);

          const Real VecExactX   = EF.exactVecX(X, Y);
          const Real VecExactY   = EF.exactVecY(X, Y);
          const Real EdgeNormalX = std::cos(AngleEdge(IEdge));
          const Real EdgeNormalY = std::sin(AngleEdge(IEdge));
          VecEdge(IEdge) = EdgeNormalX * VecExactX + EdgeNormalY * VecExactY;
       });

   // Perform halo exchange
   Halo MyHalo(MachEnv::getDefaultEnv(), Decomp::getDefault());
   auto VecEdgeH = VecEdge.createHostCopy();
   MyHalo.exchangeFullArrayHalo(VecEdgeH, OnEdge);
   VecEdgeH.deep_copy_to(VecEdge);

   const auto XVertex       = mesh->XVertexH.createDeviceCopy();
   const auto YVertex       = mesh->YVertexH.createDeviceCopy();
   const auto &AreaTriangle = mesh->AreaTriangle;

   // Compute element-wise errors
   Array1DReal LInfVertex("LInfVertex", mesh->NVerticesOwned);
   Array1DReal L2Vertex("L2Vertex", mesh->NVerticesOwned);
   CurlOnVertex CurlVertex(mesh);
   parallel_for(
       mesh->NVerticesOwned, YAKL_LAMBDA(int IVertex) {
          // Numerical result
          const Real CurlNum = CurlVertex(IVertex, VecEdge);

          // Exact result
          const Real X         = XVertex(IVertex);
          const Real Y         = YVertex(IVertex);
          const Real CurlExact = EF.exactCurlVec(X, Y);

          // Errors
          LInfVertex(IVertex) = std::abs(CurlNum - CurlExact);
          L2Vertex(IVertex) =
              AreaTriangle(IVertex) * LInfVertex(IVertex) * LInfVertex(IVertex);
       });

   // Compute global error norms
   const Real LInfErrorLoc = yakl::intrinsics::maxval(LInfVertex);
   const Real L2ErrorLoc   = yakl::intrinsics::sum(L2Vertex);

   MPI_Comm Comm = MachEnv::getDefaultEnv()->getComm();
   Real LInfError;
   Err =
       MPI_Allreduce(&LInfErrorLoc, &LInfError, 1, MPI_RealKind, MPI_MAX, Comm);

   Real L2Error;
   Err = MPI_Allreduce(&L2ErrorLoc, &L2Error, 1, MPI_RealKind, MPI_SUM, Comm);
   L2Error = std::sqrt(L2Error);

   // Check error values
   const Real ExpectedLInfError = 0.156364592741396718;
   const Real ExpectedL2Error   = 0.0729744189366629548;

   if (Err == 0 && isApprox(LInfError, ExpectedLInfError, RTol) &&
       isApprox(L2Error, ExpectedL2Error, RTol)) {
      return 0;
   } else {
      return 1;
   }
}

int testRecon(Real RTol) {
   int Err;
   ExactFunctions EF;

   const auto &mesh      = HorzMesh::getDefault();
   const auto XEdge      = mesh->XEdgeH.createDeviceCopy();
   const auto YEdge      = mesh->YEdgeH.createDeviceCopy();
   const auto &AngleEdge = mesh->AngleEdge;

   // Prepare operator input
   Array1DReal VecEdge("VecEdge", mesh->NEdgesSize);
   parallel_for(
       mesh->NEdgesOwned, YAKL_LAMBDA(int IEdge) {
          const Real X = XEdge(IEdge);
          const Real Y = YEdge(IEdge);

          const Real VecExactX   = EF.exactVecX(X, Y);
          const Real VecExactY   = EF.exactVecY(X, Y);
          const Real EdgeNormalX = std::cos(AngleEdge(IEdge));
          const Real EdgeNormalY = std::sin(AngleEdge(IEdge));
          VecEdge(IEdge) = EdgeNormalX * VecExactX + EdgeNormalY * VecExactY;
       });

   // Perform halo exchange
   Halo MyHalo(MachEnv::getDefaultEnv(), Decomp::getDefault());
   auto VecEdgeH = VecEdge.createHostCopy();
   MyHalo.exchangeFullArrayHalo(VecEdgeH, OnEdge);
   VecEdgeH.deep_copy_to(VecEdge);

   const auto &DcEdge = mesh->DcEdge;
   const auto &DvEdge = mesh->DvEdge;

   // Compute element-wise errors
   Array1DReal LInfEdge("LInfEdge", mesh->NEdgesOwned);
   Array1DReal L2Edge("L2Edge", mesh->NEdgesOwned);
   TangentialReconOnEdge TanReconEdge(mesh);
   parallel_for(
       mesh->NEdgesOwned, YAKL_LAMBDA(int IEdge) {
          // Numerical result
          const Real VecReconNum = TanReconEdge(IEdge, VecEdge);

          // Exact result
          const Real X             = XEdge(IEdge);
          const Real Y             = YEdge(IEdge);
          const Real VecX          = EF.exactVecX(X, Y);
          const Real VecY          = EF.exactVecY(X, Y);
          const Real EdgeTangentX  = -std::sin(AngleEdge(IEdge));
          const Real EdgeTangentY  = std::cos(AngleEdge(IEdge));
          const Real VecReconExact = EdgeTangentX * VecX + EdgeTangentY * VecY;

          // Errors
          LInfEdge(IEdge)     = std::abs(VecReconNum - VecReconExact);
          const Real AreaEdge = DcEdge(IEdge) * DvEdge(IEdge) / 2;
          L2Edge(IEdge)       = AreaEdge * LInfEdge(IEdge) * LInfEdge(IEdge);
       });

   // Compute global error norms
   const Real LInfErrorLoc = yakl::intrinsics::maxval(LInfEdge);
   const Real L2ErrorLoc   = yakl::intrinsics::sum(L2Edge);

   MPI_Comm Comm = MachEnv::getDefaultEnv()->getComm();
   Real LInfError;
   Err =
       MPI_Allreduce(&LInfErrorLoc, &LInfError, 1, MPI_RealKind, MPI_MAX, Comm);

   Real L2Error;
   Err = MPI_Allreduce(&L2ErrorLoc, &L2Error, 1, MPI_RealKind, MPI_SUM, Comm);
   L2Error = std::sqrt(L2Error);

   // Check error values
   const Real ExpectedLInfError = 0.00449932090822358077;
   const Real ExpectedL2Error   = 0.00194202022746063673;

   if (Err == 0 && isApprox(LInfError, ExpectedLInfError, RTol) &&
       isApprox(L2Error, ExpectedL2Error, RTol)) {
      return 0;
   } else {
      return 1;
   }
}

//------------------------------------------------------------------------------
// The initialization routine for Operators testing
int initOperatorsTest(int argc, char *argv[]) {

   MPI_Init(&argc, &argv);
   yakl::init();

   int Err = 0;

   MachEnv::init(MPI_COMM_WORLD);
   MachEnv *DefEnv  = MachEnv::getDefaultEnv();
   MPI_Comm DefComm = DefEnv->getComm();

   Err = IO::init(DefComm);
   if (Err != 0)
      LOG_ERROR("HorzMeshTest: error initializing parallel IO");

   Err = Decomp::init();
   if (Err != 0)
      LOG_ERROR("HorzMeshTest: error initializing default decomposition");

   Err = HorzMesh::init();
   if (Err != 0)
      LOG_ERROR("HorzMeshTest: error initializing default mesh");

   return Err;
}

void finalizeOperatorsTest() {
   HorzMesh::clear();
   Decomp::clear();
   MachEnv::removeAll();
   yakl::finalize();
   MPI_Finalize();
}

int main(int argc, char *argv[]) {
   int Err = initOperatorsTest(argc, argv);
   if (Err != 0)
      LOG_CRITICAL("OperatorsTest: Error initializing");

   const Real RTol = sizeof(Real) == 4 ? 1e-2 : 1e-10;

   int DivErr = testDivergence(RTol);
   if (DivErr == 0) {
      LOG_INFO("OperatorsTest: Divergence PASS");
   } else {
      Err = DivErr;
      LOG_INFO("OperatorsTest: Divergence FAIL");
   }

   int GradErr = testGradient(RTol);
   if (GradErr == 0) {
      LOG_INFO("OperatorsTest: Gradient PASS");
   } else {
      Err = GradErr;
      LOG_INFO("OperatorsTest: Gradient FAIL");
   }

   int CurlErr = testCurl(RTol);
   if (CurlErr == 0) {
      LOG_INFO("OperatorsTest: Curl PASS");
   } else {
      Err = CurlErr;
      LOG_INFO("OperatorsTest: Curl FAIL");
   }

   int ReconErr = testRecon(RTol);
   if (Err == 0) {
      LOG_INFO("OperatorsTest: Recon PASS");
   } else {
      Err = ReconErr;
      LOG_INFO("OperatorsTest: Recon FAIL");
   }

   if (Err == 0) {
      LOG_INFO("OperatorsTest: Successful completion");
   }

   finalizeOperatorsTest();
} // end of main
//===-----------------------------------------------------------------------===/
