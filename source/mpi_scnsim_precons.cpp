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
  PETScWrappers::MPI::SparseMatrix &absA,
  PETScWrappers::MPI::SparseMatrix &schur,
  PETScWrappers::MPI::SparseMatrix &B2pp)
  : timer2(timer2),
    system_matrix(&system),
    Abs_A_matrix(&absA),
    schur_matrix(&schur),
    B2pp_matrix(&B2pp),
    Tpp_itr(0)
{
  // Initialize the Pvv inverse (the ILU(0) factorization of Avv)
  Pvv_inverse.initialize(system_matrix->block(0, 0));
  // Initialize Tpp
  Tpp.reset(new SchurComplementTpp(
    timer2, owned_partitioning, *system_matrix, Pvv_inverse));

  // Compute B2pp matrix App - Apv*rowsum(|Avv|)^(-1)*Avp
  // as the preconditioner to solve Tpp^-1
  PETScWrappers::MPI::BlockVector IdentityVector, RowSumAvv, ReverseRowSum;
  IdentityVector.reinit(owned_partitioning,
                        system_matrix->get_mpi_communicator());
  RowSumAvv.reinit(owned_partitioning, system_matrix->get_mpi_communicator());
  ReverseRowSum.reinit(owned_partitioning,
                       system_matrix->get_mpi_communicator());
  // Want to set ReverseRowSum to 1 to calculate the Rowsum first
  IdentityVector.block(0) = 1;
  // iterate the Avv matrix to set everything to positive.
  Abs_A_matrix->add(1, system_matrix->block(0, 0));
  Abs_A_matrix->compress(VectorOperation::add);

  // local information of the matrix is in unit of row, so we want to know
  // the range of global row indices that the local rank has.
  unsigned int row_start = Abs_A_matrix->local_range().first;
  unsigned int row_end = Abs_A_matrix->local_range().second;
  unsigned int row_range = row_end - row_start;
  // A temporal vector to cache the columns and values to be set.
  std::vector<std::vector<unsigned int>> cache_columns;
  std::vector<std::vector<double>> cache_values;
  cache_columns.resize(row_range);
  cache_values.resize(row_range);
  for (auto r = Abs_A_matrix->local_range().first;
       r < Abs_A_matrix->local_range().second;
       ++r)
    {
      // Allocation of memory for the input values
      cache_columns[r - row_start].resize(Abs_A_matrix->row_length(r));
      cache_values[r - row_start].resize(Abs_A_matrix->row_length(r));
      unsigned int col_count = 0;
      auto itr = Abs_A_matrix->begin(r);
      while (col_count < Abs_A_matrix->row_length(r))
        {
          cache_columns[r - row_start].push_back(itr->column());
          cache_values[r - row_start].push_back(std::abs(itr->value()));
          ++col_count;
          if (col_count == Abs_A_matrix->row_length(r))
            break;
          ++itr;
        }
    }
  for (auto r = Abs_A_matrix->local_range().first;
       r < Abs_A_matrix->local_range().second;
       ++r)
    {
      Abs_A_matrix->set(
        r, cache_columns[r - row_start], cache_values[r - row_start], true);
    }
  Abs_A_matrix->compress(VectorOperation::insert);

  // Compute the diag vector rowsum(|Avv|)^(-1)
  Abs_A_matrix->vmult(RowSumAvv.block(0), IdentityVector.block(0));
  // Reverse the vector and store in ReverseRowSum
  std::vector<double> cache_vector(ReverseRowSum.block(0).local_size());
  std::vector<unsigned int> cache_rows(ReverseRowSum.block(0).local_size());
  for (auto r = ReverseRowSum.block(0).local_range().first;
       r < ReverseRowSum.block(0).local_range().second;
       ++r)
    {
      cache_vector.push_back(1 / (RowSumAvv.block(0)(r)));
      cache_rows.push_back(r);
    }
  ReverseRowSum.block(0).set(cache_rows, cache_vector);
  ReverseRowSum.compress(VectorOperation::insert);

  // Compute Schur matrix Apv*rowsum(|Avv|)^(-1)*Avp
  system_matrix->block(1, 0).mmult(
    *schur_matrix, system_matrix->block(0, 1), ReverseRowSum.block(0));
  // Add in numbers to B2pp
  B2pp_matrix->add(-1, *schur_matrix);
  B2pp_matrix->add(1, system_matrix->block(1, 1));
  B2pp_matrix->compress(VectorOperation::add);
  B2pp_inverse.initialize(*B2pp_matrix);
}

/**
 * The vmult operation strictly follows the definition of
 * BlockSchurPreconditioner. Conceptually it computes \f$u = P^{-1}v\f$.
 */
void BlockPreconditioner::vmult(
  PETScWrappers::MPI::BlockVector &dst,
  const PETScWrappers::MPI::BlockVector &src) const
{
  // Compute the intermediate vector:
  //      |I           0|*|src(0)| = |src(0)|
  //      |-ApvPvv^-1  I| |src(1)|   |ptmp  |
  /////////////////////////////////////////
  PETScWrappers::MPI::Vector ptmp1(src.block(0)), ptmp(src.block(1));
  Pvv_inverse.vmult(ptmp1, src.block(0));
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
  // Compute the multiplication
  timer2.enter_subsection("Solving Tpp");
  SolverControl solver_control(ptmp.size(), 1e-3 * ptmp.l2_norm(), true, true);
  GrowingVectorMemory<PETScWrappers::MPI::Vector> vector_memory;
  SolverGMRES<PETScWrappers::MPI::Vector> gmres(
    solver_control,
    vector_memory,
    SolverGMRES<PETScWrappers::MPI::Vector>::AdditionalData(200));
  gmres.solve(*Tpp, dst.block(1), ptmp, B2pp_inverse);
  // B2pp_inverse.vmult(dst.block(1), ptmp);
  // Count iterations for this solver solving Tpp inverse
  Tpp_itr += solver_control.last_step();

  timer2.leave_subsection("Solving Tpp");

  // Compute Pvv^-1*src(0) - Pvv^-1*Avp*dst(1)
  PETScWrappers::MPI::Vector utmp1(src.block(0)), utmp2(src.block(0));
  this->Avp().vmult(utmp1, dst.block(1));
  Pvv_inverse.vmult(utmp2, utmp1);
  Pvv_inverse.vmult(dst.block(0), src.block(0));
  dst.block(0) -= utmp2;
}