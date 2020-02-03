#ifndef MPI_FSI
#define MPI_FSI

#include <deal.II/base/table_indices.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include "mpi_fluid_solver.h"
#include "mpi_shared_solid_solver.h"

using namespace dealii;

extern template class Fluid::MPI::FluidSolver<2>;
extern template class Fluid::MPI::FluidSolver<3>;
extern template class Solid::MPI::SharedSolidSolver<2>;
extern template class Solid::MPI::SharedSolidSolver<3>;
extern template class Utils::GridInterpolator<2, Vector<double>>;
extern template class Utils::GridInterpolator<3, Vector<double>>;
extern template class Utils::GridInterpolator<2,
                                              PETScWrappers::MPI::BlockVector>;
extern template class Utils::GridInterpolator<3,
                                              PETScWrappers::MPI::BlockVector>;
extern template class Utils::SPHInterpolator<2, Vector<double>>;
extern template class Utils::SPHInterpolator<3, Vector<double>>;
extern template class Utils::SPHInterpolator<2,
                                             PETScWrappers::MPI::BlockVector>;
extern template class Utils::SPHInterpolator<3,
                                             PETScWrappers::MPI::BlockVector>;
extern template class Utils::CellLocator<2, DoFHandler<2, 2>>;
extern template class Utils::CellLocator<3, DoFHandler<3, 3>>;

namespace MPI
{
  template <int dim>
  class FSI
  {
  public:
    FSI(Fluid::MPI::FluidSolver<dim> &,
        Solid::MPI::SharedSolidSolver<dim> &,
        const Parameters::AllParameters &,
        bool use_dirichlet_bc = false);
    void run();

    //! Destructor
    ~FSI();

  private:
    /// Collect all the boundary lines in solid triangulation.
    void collect_solid_boundaries();

    /// Setup the hints for searching for each fluid cell.
    void setup_cell_hints();

    /// Define a smallest rectangle (or hex in 3d) that contains the solid.
    void update_solid_box();

    /// Find the vertices that are onwed by the local process.
    void update_vertices_mask();

    /// Check if a point is inside a mesh.
    bool point_in_solid(const DoFHandler<dim> &, const Point<dim> &);

    /*! \brief Update the indicator field of the fluid solver.
     *
     *  Although the indicator field is defined at quadrature points in order
     * to cache the fsi force, the quadrature points in the same cell are
     * updated as a whole: they are either all 1 or all 0. The criteria is
     * that whether all of the vertices are in solid mesh (because later on
     * Dirichlet BCs obtained from the solid will be applied).
     */
    void update_indicator();

    /*! \brief Update the FSI mapping between solid and fluid.
     *
     *  For energy convervation purpose, the same shape functions should be
     * used for both interpolation (fluid to solid) and distribution
     * (solid to fluid) process. In this member function, a solid node is
     * mapped with a fluid cell. The information on fluid nodes (vel, acc) are
     * interpolated onto solid nodes, then the FE integration is performed on
     * solid nodes to compute the integrated FSI force. After the integrated
     * FSI force is computed, it is distributed onto the nodes of the
     * surrounding fluid cell, and directly added onto the RHS to avoid
     * double-counted integration.
     */
    void update_distributors();

    /// Move solid triangulation either forward or backward using
    /// displacements,
    void move_solid_mesh(bool);

    /*! \brief Compute the fluid traction on solid boundaries.
     *
     *  The implementation is straight-forward: loop over the faces on the
     *  solid Neumann boundary, find the quadrature points and normals,
     *  then interpolate the fluid pressure and symmetric gradient of velocity
     * at those points, based on which the fluid traction is calculated.
     */
    void find_solid_bc();

    /*! \brief Interpolate the fluid velocity to solid vertices.
     *
     *  This is IFEM, not mIFEM.
     */
    void update_solid_displacement();

    /*! \brief Compute the Dirichlet BCs on the artificial fluid using solid
     * velocity,
     *         as well as the fsi stress and acceleration terms at the
     * artificial fluid quadrature points.
     *
     *  The Dirichlet BCs are obtained by interpolating solid velocity to the
     * fluid
     *  vertices and the FSI force is defined as:
     *  \f$F^{\text{FSI}} = \frac{Dv^f_i}{Dt}) - \sigma^f_{ij,j})\f$.
     *  In practice, we avoid directly evaluating stress divergence, so the
     * stress itself and the acceleration are separately cached onto the fluid
     * quadrature
     *  points to be used by the fluid solver.
     */
    void find_fluid_bc();

    /// Mesh adaption.
    void refine_mesh(const unsigned int, const unsigned int);

    struct DistributorStorage;

    // For MPI FSI, the solid solver uses shared trianulation. i.e.,
    // each process has the entire graph, for the ease of looping.
    Fluid::MPI::FluidSolver<dim> &fluid_solver;
    Solid::MPI::SharedSolidSolver<dim> &solid_solver;
    Parameters::AllParameters parameters;
    MPI_Comm mpi_communicator;
    ConditionalOStream pcout;
    Utils::Time time;
    mutable TimerOutput timer;

    // This vector represents the smallest box that contains the solid.
    // The point stored is in the order of:
    // (x_min, x_max, y_min, y_max, z_min, z_max)
    Vector<double> solid_box;

    // This vector collects the solid boundaries for computing thw winding
    // number.
    std::list<typename Triangulation<dim>::face_iterator> solid_boundaries;

    // A mask that marks local fluid vertices for solid bc interpolation
    // searching.
    std::vector<bool> vertices_mask;

    // A vector for solid vertices that stores the fluid cells that surrounds
    // them and the corresponding shape functions for each dof.
    std::vector<std::unique_ptr<DistributorStorage>> distributor_storage;

    std::vector<Point<dim>> solid_support_points;

    // Cell storage that stores hints of cell searching from last time step.
    CellDataStorage<
      typename parallel::distributed::Triangulation<dim>::active_cell_iterator,
      typename DoFHandler<dim>::active_cell_iterator>
      cell_hints;

    bool use_dirichlet_bc;

    struct DistributorStorage
    {
      // Methods
      DistributorStorage(typename DoFHandler<dim>::active_cell_iterator f_cell,
                         MappingQGeneric<dim> mapping)
        : f_cell(f_cell), mapping(mapping){};
      void compute_shape_values(FESystem<dim> &, const Point<dim> &);
      // Attributes
      typename DoFHandler<dim>::active_cell_iterator f_cell;
      MappingQGeneric<dim> mapping;
      std::list<std::pair<unsigned, double>> shape_value;
    };
  };
} // namespace MPI

#endif
