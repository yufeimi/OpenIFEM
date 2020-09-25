#ifndef MPI_SCNSIM_PRECONS
#define MPI_SCNSIM_PRECONS

#include "mpi_fluid_solver.h"
#include "preconditioner_pilut.h"

class BlockPreconditioner : public Subscriptor
{
public:
  enum inner_type
  {
    none = 0,
    block_jacobi,
    mumps
  };
  /// Constructor.
  BlockPreconditioner(TimerOutput &timer2,
                      const std::vector<IndexSet> &owned_partitioning,
                      const PETScWrappers::MPI::BlockSparseMatrix &system,
                      PETScWrappers::MPI::SparseMatrix &absA,
                      PETScWrappers::MPI::SparseMatrix &schur,
                      PETScWrappers::MPI::SparseMatrix &B2pp);

  /// The matrix-vector multiplication must be defined.
  void vmult(PETScWrappers::MPI::BlockVector &dst,
             const PETScWrappers::MPI::BlockVector &src) const;
  /// Accessors for the blocks of the system matrix for clearer
  /// representation
  const PETScWrappers::MPI::SparseMatrix &Avv() const
  {
    return system_matrix->block(0, 0);
  }
  const PETScWrappers::MPI::SparseMatrix &Avp() const
  {
    return system_matrix->block(0, 1);
  }
  const PETScWrappers::MPI::SparseMatrix &Apv() const
  {
    return system_matrix->block(1, 0);
  }
  const PETScWrappers::MPI::SparseMatrix &App() const
  {
    return system_matrix->block(1, 1);
  }
  int get_Tpp_itr_count() const { return Tpp_itr; }
  void Erase_Tpp_count() { Tpp_itr = 0; }

private:
  class SchurComplementTpp;

  /// We would like to time the BlockSchuPreconditioner in detail.
  TimerOutput &timer2;

  /// dealii smart pointer checks if an object is still being referenced
  /// when it is destructed therefore is safer than plain reference.
  const SmartPointer<const PETScWrappers::MPI::BlockSparseMatrix> system_matrix;
  const SmartPointer<PETScWrappers::MPI::SparseMatrix> Abs_A_matrix;
  const SmartPointer<PETScWrappers::MPI::SparseMatrix> schur_matrix;
  const SmartPointer<PETScWrappers::MPI::SparseMatrix> B2pp_matrix;

  PreconditionEuclid Pvv_inverse;
  PreconditionEuclid B2pp_inverse;

  std::shared_ptr<SchurComplementTpp> Tpp;
  // iteration counter for solving Tpp
  mutable int Tpp_itr;
  class SchurComplementTpp : public Subscriptor
  {
  public:
    SchurComplementTpp(TimerOutput &timer2,
                       const std::vector<IndexSet> &owned_partitioning,
                       const PETScWrappers::MPI::BlockSparseMatrix &system,
                       const PETScWrappers::PreconditionerBase &Pvvinv);
    void vmult(PETScWrappers::MPI::Vector &dst,
               const PETScWrappers::MPI::Vector &src) const;

  private:
    TimerOutput &timer2;
    const SmartPointer<const PETScWrappers::MPI::BlockSparseMatrix>
      system_matrix;
    const PETScWrappers::PreconditionerBase *Pvv_inverse;
    PETScWrappers::MPI::BlockVector dumb_vector;
  };
};

#endif