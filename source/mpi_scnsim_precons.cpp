#include "mpi_scnsim_precons.h"

BlockPreconditioner::SchurComplementTpp::SchurComplementTpp(
  TimerOutput &timer2,
  const std::vector<IndexSet> &owned_partitioning,
  const PETScWrappers::MPI::BlockSparseMatrix &system,
  const PETScWrappers::PreconditionerBase &Pvvinv)
  : timer2(timer2), system_matrix(&system), Pvv_inverse(&Pvvinv)
{
  dumb_vector.reinit(owned_partitioning, system_matrix->get_mpi_communicator());
}

void BlockPreconditioner::SchurComplementTpp::vmult(
  PETScWrappers::MPI::Vector &dst, const PETScWrappers::MPI::Vector &src) const
{
  // this is the exact representation of Tpp = App - Apv * Pvv * Avp.
  PETScWrappers::MPI::Vector tmp1(dumb_vector.block(0)),
    tmp2(dumb_vector.block(0)), tmp3(src);
  system_matrix->block(0, 1).vmult(tmp1, src);
  Pvv_inverse->vmult(tmp2, tmp1);
  system_matrix->block(1, 0).vmult(tmp3, tmp2);
  system_matrix->block(1, 1).vmult(dst, src);
  dst -= tmp3;
}

BlockPreconditioner::BlockPreconditioner(
  TimerOutput &timer2,
  const std::vector<IndexSet> &owned_partitioning,
  const PETScWrappers::MPI::BlockSparseMatrix &system,
  PETScWrappers::MPI::SparseMatrix &schur,
  PvvType pvv_type)
  : timer2(timer2),
    system_matrix(&system),
    schur_matrix(&schur),
    pvv_type(pvv_type),
    Tpp_itr(0)
{
  // Initialize the Pvv inverse (the ILU(0) factorization of Avv)
  Pvv_inverse_blockJacobi.initialize(system_matrix->block(0, 0));
  Pvv_inverse_ilu.initialize(system_matrix->block(0, 0));
  // Initialize Tpp
  if (pvv_type == PvvType::block_jacobi)
    {
      Tpp.reset(new SchurComplementTpp(
        timer2, owned_partitioning, *system_matrix, Pvv_inverse_blockJacobi));
    }
  else
    {
      Tpp.reset(new SchurComplementTpp(
        timer2, owned_partitioning, *system_matrix, Pvv_inverse_ilu));
    }
}

/**
 * The vmult operation strictly follows the definition of
 * BlockSchurPreconditioner. Conceptually it computes \f$u = P^{-1}v\f$.
 */
void BlockPreconditioner::vmult(
  PETScWrappers::MPI::BlockVector &dst,
  const PETScWrappers::MPI::BlockVector &src) const
{
  if (pvv_type == PvvType::none)
    {
      dst = src;
      return;
    }
  // Compute the intermediate vector:
  //      |I           0|*|src(0)| = |src(0)|
  //      |-ApvPvv^-1  I| |src(1)|   |ptmp  |
  /////////////////////////////////////////
  PETScWrappers::MPI::Vector ptmp1(src.block(0)), ptmp(src.block(1));
  if (pvv_type == PvvType::ilu)
    {
      Pvv_inverse_ilu.vmult(ptmp1, src.block(0));
    }
  else // block jacobi
    {
      Pvv_inverse_blockJacobi.vmult(ptmp1, src.block(0));
    }
  this->Apv().vmult(ptmp, ptmp1);
  ptmp *= -1.0;
  ptmp += src.block(1);

  // Compute the final vector:
  //      |Pvv^-1     -Pvv^-1*Avp*Tpp^-1|*|src(0)|
  //      |0           Tpp^-1           | |ptmp  |
  //                        =   |Pvv^-1*src(0) - Pvv^-1*Avp*Tpp^-1*ptmp|
  //                            |Tpp^-1 * ptmp                         |
  //////////////////////////////////////////
  // Compute Tpp^-1 * ptmp first, which is equal to the problem Tpp*x = ptmp
  // Set up initial guess first
  {
    PETScWrappers::MPI::Vector c(ptmp), Sc(ptmp);
    Tpp->vmult(Sc, c);
    double alpha = (ptmp * c) / (Sc * c);
    c *= alpha;
    dst.block(1) = c;
  }

  auto inner_solver = [&]() {
    SolverControl solver_control(
      ptmp.size(), 1e-3 * ptmp.l2_norm(), true, true);
    GrowingVectorMemory<PETScWrappers::MPI::Vector> vector_memory;
    SolverGMRES<PETScWrappers::MPI::Vector> gmres(
      solver_control,
      vector_memory,
      SolverGMRES<PETScWrappers::MPI::Vector>::AdditionalData(200));
    gmres.solve(
      *Tpp, dst.block(1), ptmp, PETScWrappers::PreconditionNone(this->App()));
    // Count iterations for this solver solving Tpp inverse
    Tpp_itr += solver_control.last_step();
  };
  // Compute the multiplication
  if (pvv_type == PvvType::ilu)
    {
      timer2.enter_subsection("Solving Tpp with ilu pvv");
      inner_solver();
      timer2.leave_subsection("Solving Tpp with ilu pvv");
    }
  else // blockjacobi
    {
      timer2.enter_subsection("Solving Tpp with block Jacobi pvv");
      inner_solver();
      timer2.leave_subsection("Solving Tpp with block Jacobi pvv");
    }

  // Compute Pvv^-1*src(0) - Pvv^-1*Avp*dst(1)
  PETScWrappers::MPI::Vector utmp1(src.block(0)), utmp2(src.block(0));
  this->Avp().vmult(utmp1, dst.block(1));
  if (pvv_type == PvvType::ilu)
    {
      Pvv_inverse_ilu.vmult(utmp2, utmp1);
      Pvv_inverse_ilu.vmult(dst.block(0), src.block(0));
    }
  else
    {

      Pvv_inverse_blockJacobi.vmult(utmp2, utmp1);
      Pvv_inverse_blockJacobi.vmult(dst.block(0), src.block(0));
    }
  dst.block(0) -= utmp2;
}