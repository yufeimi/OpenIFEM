#include "mpi_fsi.h"
#include <iostream>

namespace MPI
{
  template <int dim>
  FSI<dim>::~FSI()
  {
    timer.print_summary();
  }

  template <int dim>
  FSI<dim>::FSI(Fluid::MPI::FluidSolver<dim> &f,
                Solid::MPI::SharedSolidSolver<dim> &s,
                const Parameters::AllParameters &p,
                bool use_dirichlet_bc)
    : fluid_solver(f),
      solid_solver(s),
      parameters(p),
      mpi_communicator(MPI_COMM_WORLD),
      pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0),
      time(parameters.end_time,
           parameters.time_step,
           parameters.output_interval,
           parameters.refinement_interval,
           parameters.save_interval),
      timer(
        mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times),
      use_dirichlet_bc(use_dirichlet_bc)
  {
    solid_box.reinit(2 * dim);
  }

  template <int dim>
  void FSI<dim>::move_solid_mesh(bool move_forward)
  {
    TimerOutput::Scope timer_section(timer, "Move solid mesh");
    // All gather the information so each process has the entire solution.
    Vector<double> localized_displacement(solid_solver.current_displacement);
    // Exactly the same as the serial version, since we must update the
    // entire graph on every process.
    std::vector<bool> vertex_touched(solid_solver.triangulation.n_vertices(),
                                     false);
    for (auto cell = solid_solver.dof_handler.begin_active();
         cell != solid_solver.dof_handler.end();
         ++cell)
      {
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            if (!vertex_touched[cell->vertex_index(v)])
              {
                vertex_touched[cell->vertex_index(v)] = true;
                Point<dim> vertex_displacement;
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    vertex_displacement[d] =
                      localized_displacement(cell->vertex_dof_index(v, d));
                  }
                if (move_forward)
                  {
                    cell->vertex(v) += vertex_displacement;
                  }
                else
                  {
                    cell->vertex(v) -= vertex_displacement;
                  }
              }
          }
      }
  }

  template <int dim>
  void FSI<dim>::collect_solid_boundaries()
  {
    if (dim == 2)
      for (auto cell = solid_solver.triangulation.begin_active();
           cell != solid_solver.triangulation.end();
           ++cell)
        {
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            {
              if (cell->face(f)->at_boundary())
                {
                  solid_boundaries.push_back(cell->face(f));
                }
            }
        }
  }

  template <int dim>
  void FSI<dim>::update_solid_box()
  {
    move_solid_mesh(true);
    solid_box = 0;
    for (unsigned int i = 0; i < dim; ++i)
      {
        solid_box(2 * i) =
          solid_solver.triangulation.get_vertices().begin()->operator()(i);
        solid_box(2 * i + 1) =
          solid_solver.triangulation.get_vertices().begin()->operator()(i);
      }
    for (auto v = solid_solver.triangulation.get_vertices().begin();
         v != solid_solver.triangulation.get_vertices().end();
         ++v)
      {
        for (unsigned int i = 0; i < dim; ++i)
          {
            if ((*v)(i) < solid_box(2 * i))
              solid_box(2 * i) = (*v)(i);
            else if ((*v)(i) > solid_box(2 * i + 1))
              solid_box(2 * i + 1) = (*v)(i);
          }
      }
    move_solid_mesh(false);
  }

  template <int dim>
  void FSI<dim>::update_vertices_mask()
  {
    // Initilize vertices mask
    vertices_mask.resize(fluid_solver.triangulation.n_vertices(), false);
    for (auto cell = fluid_solver.triangulation.begin_active();
         cell != fluid_solver.triangulation.end();
         ++cell)
      {
        if (!cell->is_locally_owned())
          {
            continue;
          }
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            vertices_mask[cell->vertex_index(v)] = true;
          }
      }
  }

  template <int dim>
  bool FSI<dim>::point_in_solid(const DoFHandler<dim> &df,
                                const Point<dim> &point)
  {
    // Check whether the point is in the solid box first.
    for (unsigned int i = 0; i < dim; ++i)
      {
        if (point(i) < solid_box(2 * i) || point(i) > solid_box(2 * i + 1))
          return false;
      }

    // Compute its angle to each boundary face
    if (dim == 2)
      {
        unsigned int cross_number = 0;
        unsigned int half_cross_number = 0;
        for (auto f = solid_boundaries.begin(); f != solid_boundaries.end();
             ++f)
          {
            Point<dim> p1 = (*f)->vertex(0), p2 = (*f)->vertex(1);
            double y_diff1 = p1(1) - point(1);
            double y_diff2 = p2(1) - point(1);
            double x_diff1 = p1(0) - point(0);
            double x_diff2 = p2(0) - point(0);
            Tensor<1, dim> r1 = p1 - p2;
            Tensor<1, dim> r2;
            // r1[1] == 0 if the boundary is horizontal
            if (r1[1] != 0.0)
              r2 = r1 * (point(1) - p2(1)) / r1[1];
            if (y_diff1 * y_diff2 < 0)
              {
                // Point is on the left of the boundary
                if (r2[0] + p2(0) > point(0))
                  {
                    ++cross_number;
                  }
                // Point is on the boundary
                else if (r2[0] + p2(0) == point(0))
                  {
                    return true;
                  }
              }
            // Point is on the same horizontal line with one of the vertices
            else if (y_diff1 * y_diff2 == 0)
              {
                // The boundary is horizontal
                if (y_diff1 == 0 && y_diff2 == 0)
                  // The point is on it
                  if (x_diff1 * x_diff2 < 0)
                    {
                      return true;
                    }
                  // The point is not on it
                  else
                    continue;
                // On the left of the boundary
                else if (r2[0] + p2(0) > point(0))
                  { // The point must not be on the top or bottom of the box
                    // (because it can be tangential)
                    if (point(1) != solid_box(2) && point(1) != solid_box(3))
                      ++half_cross_number;
                  }
                // Point overlaps with the vertex
                else if (point == p1 || point == p2)
                  {
                    return true;
                  }
              }
          }
        cross_number += half_cross_number / 2;
        if (cross_number % 2 == 0)
          return false;
        return true;
      }
    for (auto cell = df.begin_active(); cell != df.end(); ++cell)
      {
        if (cell->point_inside(point))
          {
            return true;
          }
      }
    return false;
  }

  template <int dim>
  void FSI<dim>::setup_cell_hints()
  {
    unsigned int n_unit_points =
      fluid_solver.fe.get_unit_support_points().size();
    for (auto cell = fluid_solver.triangulation.begin_active();
         cell != fluid_solver.triangulation.end();
         ++cell)
      {
        if (!cell->is_artificial())
          {
            cell_hints.initialize(cell, n_unit_points);
            const std::vector<
              std::shared_ptr<typename DoFHandler<dim>::active_cell_iterator>>
              hints = cell_hints.get_data(cell);
            Assert(hints.size() == n_unit_points,
                   ExcMessage("Wrong number of cell hints!"));
            for (unsigned int v = 0; v < n_unit_points; ++v)
              {
                // Initialize the hints with the begin iterators!
                *(hints[v]) = solid_solver.dof_handler.begin_active();
              }
          }
      }
  }

  template <int dim>
  void FSI<dim>::update_solid_displacement()
  {
    move_solid_mesh(true);
    Vector<double> localized_solid_displacement(
      solid_solver.current_displacement);
    std::vector<bool> vertex_touched(solid_solver.dof_handler.n_dofs(), false);
    for (auto cell : solid_solver.dof_handler.active_cell_iterators())
      {
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            if (!vertex_touched[cell->vertex_index(v)] &&
                !solid_solver.constraints.is_constrained(cell->vertex_index(v)))
              {
                vertex_touched[cell->vertex_index(v)] = true;
                Point<dim> point = cell->vertex(v);
                Vector<double> tmp(dim + 1);
                VectorTools::point_value(fluid_solver.dof_handler,
                                         fluid_solver.present_solution,
                                         point,
                                         tmp);
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    localized_solid_displacement[cell->vertex_dof_index(
                      v, d)] += tmp[d] * time.get_delta_t();
                  }
              }
          }
      }
    move_solid_mesh(false);
    solid_solver.current_displacement = localized_solid_displacement;
  }

  // Dirichlet bcs are applied to artificial fluid cells, so fluid nodes
  // should be marked as artificial or real. Meanwhile, additional body force
  // is acted at the artificial fluid quadrature points. To accomodate these
  // two settings, we define indicator at quadrature points, but only when all
  // of the vertices of a fluid cell are found to be in solid domain,
  // set the indicators at all quadrature points to be 1.
  template <int dim>
  void FSI<dim>::update_indicator()
  {
    TimerOutput::Scope timer_section(timer, "Update indicator");
    move_solid_mesh(true);
    AffineConstraints<double> indicator_constraint;
    indicator_constraint.clear();
    indicator_constraint.reinit(fluid_solver.locally_relevant_scalar_dofs);

    const std::vector<Point<dim>> &unit_points =
      fluid_solver.scalar_fe.get_unit_support_points();
    std::vector<types::global_dof_index> scalar_dof_indices(
      fluid_solver.scalar_fe.dofs_per_cell);

    MappingQGeneric<dim> mapping(parameters.fluid_pressure_degree);
    Quadrature<dim> dummy_q(unit_points);
    FEValues<dim> dummy_fe_values(
      mapping, fluid_solver.scalar_fe, dummy_q, update_quadrature_points);
    for (auto scalar_f_cell = fluid_solver.scalar_dof_handler.begin_active();
         scalar_f_cell != fluid_solver.scalar_dof_handler.end();
         ++scalar_f_cell)
      {
        if (scalar_f_cell->is_artificial())
          {
            continue;
          }
        dummy_fe_values.reinit(scalar_f_cell);
        scalar_f_cell->get_dof_indices(scalar_dof_indices);
        auto support_points = dummy_fe_values.get_quadrature_points();
        bool artificial_fluid_cell = false;
        for (unsigned int i = 0; i < unit_points.size(); ++i)
          {
            if (point_in_solid(solid_solver.dof_handler, support_points[i]))
              {
                artificial_fluid_cell = true;
                break;
              }
          }
        if (artificial_fluid_cell)
          for (unsigned int i = 0; i < unit_points.size(); ++i)
            {
              auto scalar_line = scalar_dof_indices[i];
              indicator_constraint.add_line(scalar_line);
              indicator_constraint.set_inhomogeneity(scalar_line, 1);
            }
      }
    indicator_constraint.close();
    PETScWrappers::MPI::Vector tmp;
    tmp.reinit(fluid_solver.locally_owned_scalar_dofs,
               fluid_solver.mpi_communicator);
    indicator_constraint.distribute(tmp);
    fluid_solver.indicator = tmp;
    move_solid_mesh(false);
  }

  // Loop over the fluid cells, and find solid cells with their vertices inside
  // this fluid cell.
  template <int dim>
  void FSI<dim>::update_distributors()
  {
    move_solid_mesh(true);
    AssertThrow(parameters.solid_degree == 1, ExcNotImplemented());
    std::vector<int> vertices_used(solid_solver.triangulation.n_vertices(), 0);
    // Re-initialize the distributor storage
    distributor_storage.clear();
    distributor_storage.resize(solid_support_points.size());
    std::list<typename DoFHandler<dim>::active_cell_iterator> in_solid_f_cells;
    // Pre-filter in-solid fluid cells
    for (auto f_cell : fluid_solver.dof_handler.active_cell_iterators())
      {
        if (!f_cell->is_locally_owned())
          continue;
        for (unsigned v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            if (point_in_solid(solid_solver.dof_handler, f_cell->vertex(v)))
              {
                in_solid_f_cells.push_back(f_cell);
                break;
              }
          }
      }
    // Find surrounding fluid cells for each solid vertex
    Vector<double> localized_solid_displacement(
      solid_solver.current_displacement);
    for (unsigned i = 0; i < solid_support_points.size(); ++i)
      {
        // Move the support point to current location
        const Point<dim> &support_point = solid_support_points[i];
        Point<dim> point_displacement;
        for (unsigned int j = 0; j < dim; ++j)
          {
            point_displacement(j) = localized_solid_displacement[i + j];
          }
        Point<dim> moved_support_point = support_point + point_displacement;
        // Loop over fluid cells to find the surrounding one
        for (auto f_cell : in_solid_f_cells)
          {
            if (f_cell->point_inside(moved_support_point))
              {
                distributor_storage[i] = std::make_unique<DistributorStorage>(
                  f_cell,
                  MappingQGeneric<dim>(parameters.fluid_velocity_degree));
                distributor_storage[i]->compute_shape_values(
                  fluid_solver.fe, moved_support_point);
              }
          }
        i = i + dim - 1;
      }
    move_solid_mesh(false);
  }

  // This function interpolates the solid velocity into the fluid solver,
  // as the Dirichlet boundary conditions for artificial fluid vertices
  template <int dim>
  void FSI<dim>::find_fluid_bc()
  {
    TimerOutput::Scope timer_section(timer, "Find fluid BC");
    move_solid_mesh(true);

    // Do not use dirichlete bc for fluid
    if (!use_dirichlet_bc)
      {
        // Update the distributors
        update_distributors();

        // Interpolate the fluid velocity onto solid vertices
        fluid_solver.fsi_acceleration = 0;
        Vector<double> localized_solid_displacement(
          solid_solver.current_displacement);
        Vector<double> local_velocity(solid_support_points.size());
        for (unsigned i = 0; i < solid_support_points.size(); ++i)
          {
            if (!distributor_storage[i])
              continue;
            // Move the support point to current location
            const Point<dim> &support_point = solid_support_points[i];
            Point<dim> point_displacement;
            for (unsigned int j = 0; j < dim; ++j)
              {
                point_displacement(j) = localized_solid_displacement[i + j];
              }
            Point<dim> moved_support_point = support_point + point_displacement;

            Utils::GridInterpolator<dim, PETScWrappers::MPI::BlockVector>
              interpolator(fluid_solver.dof_handler,
                           moved_support_point,
                           {},
                           distributor_storage[i]->f_cell);
            // Check if the vertex is in the fluid cell
            if (!interpolator.found_cell())
              {
                std::stringstream message;
                message << "Interpolation for solid vertex "
                        << moved_support_point
                        << " failed due to coordinates mismatch." << std::endl;
                AssertThrow(interpolator.found_cell(),
                            ExcMessage(message.str()));
              }
            // Start interpolation
            Vector<double> fluid_vel(dim + 1);
            interpolator.point_value(fluid_solver.present_solution, fluid_vel);
            // Jump through the support points on the same node
            for (unsigned j = 0; j < dim; ++j)
              {
                local_velocity[i + j] = fluid_vel[j];
              }
            i = i + dim - 1;
          }
        Utilities::MPI::sum(local_velocity, mpi_communicator, local_velocity);
        solid_solver.fluid_velocity = local_velocity;

        // Let the solid solver assemble the FSI force
        solid_solver.assemble_fsi_force();

        // Distribute the FSI force back onto the fluid
        Vector<double> localized_solid_fsi_force(solid_solver.fsi_force);
        for (unsigned i = 0; i < solid_support_points.size(); ++i)
          {
            if (!distributor_storage[i])
              continue;
            auto f_cell = distributor_storage[i]->f_cell;
            // Stored fluid element must be local
            Assert(
              f_cell->is_locally_owned(),
              ExcMessage("Fluid cell on distribution storage is not local!"));
            std::vector<types::global_dof_index> dof_indices(
              fluid_solver.fe.dofs_per_cell);

            const unsigned int dofs_per_cell = fluid_solver.fe.dofs_per_cell;
            Vector<double> local_distributed_force(dofs_per_cell);
            /* Iterate over the shape functions. As the shape functions are
            based on the fluid support points, for the same point there would
            be (dim + 1) shape functions. We will just use the velocity ones
            and skip the pressure dof.
            */
            for (auto itr = distributor_storage[i]->shape_value.begin();
                 itr != distributor_storage[i]->shape_value.end();
                 ++itr)
              { // Outer loop on nodes
                for (unsigned j = 0; j < dim; ++j)
                  { // Inner loop on DoFs
                    local_distributed_force[itr->first] =
                      itr->second * localized_solid_fsi_force[i + j];
                    ++itr;
                  }
                // Skip the pressure dof
              }

            f_cell->get_dof_indices(dof_indices);

            fluid_solver.zero_constraints.distribute_local_to_global(
              local_distributed_force,
              dof_indices,
              fluid_solver.fsi_acceleration);
            i = i + dim - 1;
          }
        fluid_solver.fsi_acceleration.compress(VectorOperation::add);
      }
    // Use Dirichlet BC for fluid
    else
      {
        // The nonzero Dirichlet BCs (to set the velocity) and zero Dirichlet
        // BCs (to set the velocity increment) for the artificial fluid domain.
        AffineConstraints<double> inner_nonzero, inner_zero;
        inner_nonzero.clear();
        inner_zero.clear();
        inner_nonzero.reinit(fluid_solver.locally_relevant_dofs);
        inner_zero.reinit(fluid_solver.locally_relevant_dofs);

        // Cell center in unit coordinate system
        Point<dim> unit_center;
        for (unsigned int i = 0; i < dim; ++i)
          {
            unit_center[i] = 0.5;
          }
        Quadrature<dim> quad(unit_center);
        MappingQGeneric<dim> mapping(parameters.fluid_velocity_degree);
        FEValues<dim> fe_values(mapping,
                                fluid_solver.fe,
                                quad,
                                update_quadrature_points | update_values |
                                  update_gradients);
        Vector<double> localized_solid_velocity(solid_solver.current_velocity);

        const std::vector<Point<dim>> &unit_points =
          fluid_solver.fe.get_unit_support_points();
        Quadrature<dim> dummy_q(unit_points);
        FEValues<dim> dummy_fe_values(
          mapping, fluid_solver.fe, dummy_q, update_quadrature_points);
        std::vector<types::global_dof_index> dof_indices(
          fluid_solver.fe.dofs_per_cell);
        std::vector<unsigned int> dof_touched(fluid_solver.dof_handler.n_dofs(),
                                              0);

        for (auto f_cell = fluid_solver.dof_handler.begin_active();
             f_cell != fluid_solver.dof_handler.end();
             ++f_cell)
          {
            // Use is_artificial() instead of !is_locally_owned() because ghost
            // elements must be taken care of to set correct Dirichlet BCs!
            if (f_cell->is_artificial())
              {
                continue;
              }
            // Dirichlet BCs
            dummy_fe_values.reinit(f_cell);
            f_cell->get_dof_indices(dof_indices);
            auto support_points = dummy_fe_values.get_quadrature_points();
            auto hints = cell_hints.get_data(f_cell);
            // Declare the fluid velocity for interpolating BC
            Vector<double> fluid_velocity(dim);
            // Loop over the support points to set Dirichlet BCs.
            for (unsigned int i = 0; i < unit_points.size(); ++i)
              {
                // Skip the already-set dofs.
                if (dof_touched[dof_indices[i]] != 0)
                  continue;
                auto base_index = fluid_solver.fe.system_to_base_index(i);
                const unsigned int i_group = base_index.first.first;
                Assert(i_group < 2,
                       ExcMessage(
                         "There should be only 2 groups of finite element!"));
                if (i_group == 1)
                  continue; // skip the pressure dofs
                bool inside = true;
                for (unsigned int d = 0; d < dim; ++d)
                  if (std::abs(unit_points[i][d]) < 1e-5)
                    {
                      inside = false;
                      break;
                    }
                if (inside)
                  continue; // skip the in-cell support point
                // Same as fluid_solver.fe.system_to_base_index(i).first.second;
                const unsigned int index =
                  fluid_solver.fe.system_to_component_index(i).first;
                Assert(index < dim,
                       ExcMessage("Vector component should be less than dim!"));
                dof_touched[dof_indices[i]] = 1;
                if (!point_in_solid(solid_solver.dof_handler,
                                    support_points[i]))
                  continue;
                Utils::CellLocator<dim, DoFHandler<dim>> locator(
                  solid_solver.dof_handler, support_points[i], *(hints[i]));
                *(hints[i]) = locator.search();
                Utils::GridInterpolator<dim, Vector<double>> interpolator(
                  solid_solver.dof_handler, support_points[i], {}, *(hints[i]));
                if (!interpolator.found_cell())
                  {
                    std::stringstream message;
                    message
                      << "Cannot find point in solid: " << support_points[i]
                      << std::endl;
                    AssertThrow(interpolator.found_cell(),
                                ExcMessage(message.str()));
                  }
                interpolator.point_value(localized_solid_velocity,
                                         fluid_velocity);
                auto line = dof_indices[i];
                inner_nonzero.add_line(line);
                inner_zero.add_line(line);
                // Note that we are setting the value of the constraint to the
                // velocity delta!
                inner_nonzero.set_inhomogeneity(
                  line,
                  fluid_velocity[index] - fluid_solver.present_solution(line));
              }
          }
        inner_nonzero.close();
        inner_zero.close();
        fluid_solver.nonzero_constraints.merge(
          inner_nonzero,
          AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
        fluid_solver.zero_constraints.merge(
          inner_zero,
          AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
      }

    move_solid_mesh(false);
  }

  template <int dim>
  void FSI<dim>::find_solid_bc()
  {
    TimerOutput::Scope timer_section(timer, "Find solid BC");
    // Must use the updated solid coordinates
    move_solid_mesh(true);
    // Fluid FEValues to do interpolation
    FEValues<dim> fe_values(
      fluid_solver.fe, fluid_solver.volume_quad_formula, update_values);
    // Solid FEFaceValues to get the normal at face center
    Point<dim - 1> unit_face_center;
    for (unsigned int i = 0; i < dim - 1; ++i)
      {
        unit_face_center[i] = 0.5;
      }
    Quadrature<dim - 1> center_quad(unit_face_center);
    FEFaceValues<dim> fe_face_values(solid_solver.fe,
                                     unit_face_center,
                                     update_quadrature_points |
                                       update_normal_vectors);

    for (auto s_cell = solid_solver.dof_handler.begin_active();
         s_cell != solid_solver.dof_handler.end();
         ++s_cell)
      {
        auto ptr = solid_solver.cell_property.get_data(s_cell);
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            // Current face is at boundary and without Dirichlet bc.
            if (s_cell->face(f)->at_boundary())
              {
                fe_face_values.reinit(s_cell, f);
                // face center
                Point<dim> q_point = fe_face_values.quadrature_point(0);
                Tensor<1, dim> normal = fe_face_values.normal_vector(0);
                Vector<double> value(dim + 1);
                Utils::GridInterpolator<dim, PETScWrappers::MPI::BlockVector>
                  interpolator(
                    fluid_solver.dof_handler, q_point, vertices_mask);
                interpolator.point_value(fluid_solver.present_solution, value);
                std::vector<Tensor<1, dim>> gradient(dim + 1, Tensor<1, dim>());
                interpolator.point_gradient(fluid_solver.present_solution,
                                            gradient);
                Vector<double> global_value(dim + 1);
                std::vector<Tensor<1, dim>> global_gradient(dim + 1,
                                                            Tensor<1, dim>());
                for (unsigned int i = 0; i < dim + 1; ++i)
                  {
                    global_value[i] =
                      Utilities::MPI::sum(value[i], mpi_communicator);
                    global_gradient[i] =
                      Utilities::MPI::sum(gradient[i], mpi_communicator);
                  }
                // Compute stress
                SymmetricTensor<2, dim> sym_deformation;
                for (unsigned int i = 0; i < dim; ++i)
                  {
                    for (unsigned int j = 0; j < dim; ++j)
                      {
                        sym_deformation[i][j] =
                          (global_gradient[i][j] + global_gradient[j][i]) / 2;
                      }
                  }
                // \f$ \sigma = -p\bold{I} + \mu\nabla^S v\f$
                SymmetricTensor<2, dim> stress =
                  -global_value[dim] *
                    Physics::Elasticity::StandardTensors<dim>::I +
                  2 * parameters.viscosity * sym_deformation;
                ptr[f]->fsi_traction = stress * normal;
              }
          }
      }
    move_solid_mesh(false);
  }

  template <int dim>
  void FSI<dim>::refine_mesh(const unsigned int min_grid_level,
                             const unsigned int max_grid_level)
  {
    TimerOutput::Scope timer_section(timer, "Refine mesh");
    move_solid_mesh(true);
    std::vector<Point<dim>> solid_boundary_points;
    for (auto s_cell : solid_solver.dof_handler.active_cell_iterators())
      {
        bool is_boundary = false;
        Point<dim> point;
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            if (s_cell->face(face)->at_boundary())
              {
                point = s_cell->face(face)->center();
                is_boundary = true;
                break;
              }
          }
        if (is_boundary)
          {
            solid_boundary_points.push_back(point);
          }
      }
    for (auto f_cell : fluid_solver.dof_handler.active_cell_iterators())
      {
        auto center = f_cell->center();
        double dist = 1000;
        for (auto point : solid_boundary_points)
          {
            dist = std::min(center.distance(point), dist);
          }
        if (dist < f_cell->diameter())
          f_cell->set_refine_flag();
        else
          f_cell->set_coarsen_flag();
      }
    move_solid_mesh(false);
    if (fluid_solver.triangulation.n_levels() > max_grid_level)
      {
        for (auto cell =
               fluid_solver.triangulation.begin_active(max_grid_level);
             cell != fluid_solver.triangulation.end();
             ++cell)
          {
            cell->clear_refine_flag();
          }
      }

    for (auto cell = fluid_solver.triangulation.begin_active(min_grid_level);
         cell != fluid_solver.triangulation.end_active(min_grid_level);
         ++cell)
      {
        cell->clear_coarsen_flag();
      }

    parallel::distributed::SolutionTransfer<dim,
                                            PETScWrappers::MPI::BlockVector>
      solution_transfer(fluid_solver.dof_handler);

    fluid_solver.triangulation.prepare_coarsening_and_refinement();
    solution_transfer.prepare_for_coarsening_and_refinement(
      fluid_solver.present_solution);

    fluid_solver.triangulation.execute_coarsening_and_refinement();

    fluid_solver.setup_dofs();
    fluid_solver.make_constraints();
    fluid_solver.initialize_system();

    PETScWrappers::MPI::BlockVector buffer;
    buffer.reinit(fluid_solver.owned_partitioning,
                  fluid_solver.mpi_communicator);
    buffer = 0;
    solution_transfer.interpolate(buffer);
    fluid_solver.nonzero_constraints.distribute(buffer);
    fluid_solver.present_solution = buffer;
    update_vertices_mask();
  }

  template <int dim>
  void FSI<dim>::run()
  {
    pcout << "Running with PETSc on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    solid_solver.triangulation.refine_global(parameters.global_refinements[1]);
    // Try load from previous computation.
    bool success_load =
      solid_solver.load_checkpoint() && fluid_solver.load_checkpoint();
    AssertThrow(
      solid_solver.time.current() == fluid_solver.time.current(),
      ExcMessage("Solid and fluid restart files have different time steps. "
                 "Check and remove inconsistent restart files!"));
    if (!success_load)
      {
        solid_solver.setup_dofs();
        solid_solver.initialize_system();
        fluid_solver.triangulation.refine_global(
          parameters.global_refinements[0]);
        fluid_solver.setup_dofs();
        fluid_solver.make_constraints();
        fluid_solver.initialize_system();
      }
    else
      {
        while (time.get_timestep() < solid_solver.time.get_timestep())
          {
            time.increment();
          }
      }

    collect_solid_boundaries();
    update_solid_box();
    setup_cell_hints();
    update_vertices_mask();
    solid_support_points.resize(solid_solver.dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(
      MappingQGeneric<dim>(parameters.solid_degree),
      solid_solver.dof_handler,
      solid_support_points);

    pcout << "Number of fluid active cells and dofs: ["
          << fluid_solver.triangulation.n_active_cells() << ", "
          << fluid_solver.dof_handler.n_dofs() << "]" << std::endl
          << "Number of solid active cells and dofs: ["
          << solid_solver.triangulation.n_active_cells() << ", "
          << solid_solver.dof_handler.n_dofs() << "]" << std::endl;
    bool first_step = !success_load;
    if (parameters.refinement_interval < parameters.end_time)
      {
        refine_mesh(parameters.global_refinements[0],
                    parameters.global_refinements[0] + 3);
        refine_mesh(parameters.global_refinements[0],
                    parameters.global_refinements[0] + 3);
        setup_cell_hints();
        solid_support_points.resize(solid_solver.dof_handler.n_dofs());
        DoFTools::map_dofs_to_support_points(
          MappingQGeneric<dim>(parameters.solid_degree),
          solid_solver.dof_handler,
          solid_support_points);
      }
    while (time.end() - time.current() > 1e-12)
      {
        find_solid_bc();
        if (success_load)
          {
            solid_solver.assemble_system(true);
          }
        {
          TimerOutput::Scope timer_section(timer, "Run solid solver");
          solid_solver.run_one_step(first_step);
        }
        update_solid_box();
        update_indicator();
        fluid_solver.make_constraints();
        if (!first_step)
          {
            fluid_solver.nonzero_constraints.clear();
            fluid_solver.nonzero_constraints.copy_from(
              fluid_solver.zero_constraints);
          }
        find_fluid_bc();
        {
          TimerOutput::Scope timer_section(timer, "Run fluid solver");
          fluid_solver.run_one_step(true);
        }
        first_step = false;
        time.increment();
        if (time.time_to_refine())
          {
            refine_mesh(parameters.global_refinements[0],
                        parameters.global_refinements[0] + 3);
            setup_cell_hints();
          }
        if (time.time_to_save())
          {
            solid_solver.save_checkpoint(time.get_timestep());
            fluid_solver.save_checkpoint(time.get_timestep());
          }
      }
  }

  template <int dim>
  void FSI<dim>::DistributorStorage::compute_shape_values(FESystem<dim> &fe,
                                                          const Point<dim> &p)
  {
    Point<dim> unit_p = mapping.transform_real_to_unit_cell(f_cell, p);
    for (unsigned i = 0; i < fe.dofs_per_cell; ++i)
      {
        this->shape_value.push_back(
          std::make_pair(i, fe.shape_value(i, unit_p)));
      }
  }

  template class FSI<2>;
  template class FSI<3>;
} // namespace MPI
