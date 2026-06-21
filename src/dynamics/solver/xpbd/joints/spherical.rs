use super::PointConstraintShared;
use crate::{
    dynamics::{
        joints::MotorModel,
        solver::{
            solver_body::{SolverBody, SolverBodyInertia},
            xpbd::*,
        },
    },
    prelude::*,
};
use bevy::prelude::*;

/// Constraint data required by the XPBD constraint solver for a [`SphericalJoint`].
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Component, Debug, PartialEq)]
pub struct SphericalJointSolverData {
    pub(super) point_constraint: PointConstraintShared,
    pub(super) swing_axis1: Vector,
    pub(super) swing_axis2: Vector,
    pub(super) twist_axis1: Vector,
    pub(super) twist_axis2: Vector,
    pub(super) total_swing_lagrange: Vector,
    pub(super) total_twist_lagrange: Vector,
    /// World-space orientation of the first body's joint frame (`rotation1 * basis1`), captured at
    /// prepare time. The motor reconstructs the live frames each substep via the bodies' `delta_rotation`.
    pub(super) frame1: Quaternion,
    /// World-space orientation of the second body's joint frame (`rotation2 * basis2`) at prepare time.
    pub(super) frame2: Quaternion,
    /// Accumulated motor Lagrange multiplier (world-space rotation vector) for this frame.
    pub(super) total_motor_lagrange: Vector,
    /// Motor Lagrange multiplier from the previous frame, used for warm starting.
    /// Zeroed after being applied in the first substep.
    pub(super) warm_start_motor_lagrange: Vector,
}

impl XpbdConstraintSolverData for SphericalJointSolverData {
    fn clear_lagrange_multipliers(&mut self) {
        self.point_constraint.clear_lagrange_multipliers();
        self.total_swing_lagrange = Vector::ZERO;
        self.total_twist_lagrange = Vector::ZERO;
        // Save motor lagrange for warm starting before clearing.
        self.warm_start_motor_lagrange = self.total_motor_lagrange;
        self.total_motor_lagrange = Vector::ZERO;
    }

    fn total_position_lagrange(&self) -> Vector {
        self.point_constraint.total_position_lagrange()
    }

    fn total_rotation_lagrange(&self) -> AngularVector {
        self.total_swing_lagrange + self.total_twist_lagrange + self.total_motor_lagrange
    }

    fn total_motor_lagrange(&self) -> Scalar {
        self.total_motor_lagrange.length()
    }
}

impl XpbdConstraint<2> for SphericalJoint {
    type SolverData = SphericalJointSolverData;

    fn prepare(
        &mut self,
        bodies: [&RigidBodyQueryReadOnlyItem; 2],
        solver_data: &mut SphericalJointSolverData,
    ) {
        let [body1, body2] = bodies;

        let Some(local_anchor1) = self.local_anchor1() else {
            return;
        };
        let Some(local_anchor2) = self.local_anchor2() else {
            return;
        };
        let Some(local_basis1) = self.local_basis1() else {
            return;
        };
        let Some(local_basis2) = self.local_basis2() else {
            return;
        };

        // Compute the rotation matrices since we're performing so many rotations.
        let rot1_mat = Matrix::from_quat(body1.rotation.0);
        let rot2_mat = Matrix::from_quat(body2.rotation.0);

        // Prepare the point-to-point constraint.
        let point_constraint = &mut solver_data.point_constraint;
        point_constraint.world_r1 = rot1_mat * (local_anchor1 - body1.center_of_mass.0);
        point_constraint.world_r2 = rot2_mat * (local_anchor2 - body2.center_of_mass.0);
        point_constraint.center_difference = (body2.position.0 - body1.position.0)
            + (body2.rotation * body2.center_of_mass.0 - body1.rotation * body1.center_of_mass.0);

        // Prepare the base swing and twist axes.
        let swing_axis = self.twist_axis.any_orthonormal_vector();
        solver_data.swing_axis1 = rot1_mat * (local_basis1 * swing_axis);
        solver_data.swing_axis2 = rot2_mat * (local_basis2 * swing_axis);
        solver_data.twist_axis1 = rot1_mat * (local_basis1 * self.twist_axis);
        solver_data.twist_axis2 = rot2_mat * (local_basis2 * self.twist_axis);

        // Prepare the world-space joint frame orientations for the motor. The live frames are
        // reconstructed each substep by premultiplying with the bodies' `delta_rotation`.
        solver_data.frame1 = body1.rotation.0 * local_basis1;
        solver_data.frame2 = body2.rotation.0 * local_basis2;
    }

    fn solve(
        &mut self,
        bodies: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        solver_data: &mut SphericalJointSolverData,
        dt: Scalar,
    ) {
        let [body1, body2] = bodies;
        let [inertia1, inertia2] = inertias;

        // Align positions
        solver_data.point_constraint.solve(
            [body1, body2],
            [inertia1, inertia2],
            self.point_compliance,
            dt,
        );

        // Solve the motor before limits to give limits higher priority.
        let inv_inertia1 = inertia1.effective_inv_angular_inertia();
        let inv_inertia2 = inertia2.effective_inv_angular_inertia();
        self.apply_motor(
            body1,
            body2,
            inv_inertia1,
            inv_inertia2,
            solver_data,
            dt,
        );

        // Apply swing limits
        self.apply_swing_limits(body1, body2, inertia1, inertia2, solver_data, dt);

        // Apply twist limits
        self.apply_twist_limits(body1, body2, inertia1, inertia2, solver_data, dt);
    }

    fn warm_start_motors(
        &self,
        bodies: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        solver_data: &mut SphericalJointSolverData,
        _dt: Scalar,
        warm_start_coefficient: Scalar,
    ) {
        if !self.motor.enabled {
            return;
        }

        let [body1, body2] = bodies;
        let [inertia1, inertia2] = inertias;

        let inv_inertia1 = inertia1.effective_inv_angular_inertia();
        let inv_inertia2 = inertia2.effective_inv_angular_inertia();

        let impulse = warm_start_coefficient * solver_data.warm_start_motor_lagrange;

        body1.angular_velocity -= inv_inertia1 * impulse;
        body2.angular_velocity += inv_inertia2 * impulse;

        solver_data.warm_start_motor_lagrange = Vector::ZERO;
    }
}

impl SphericalJoint {
    /// Applies the 3-DOF angular motor, driving the relative rotation of the joint frames towards
    /// [`target_orientation`](SphericalJoint::target_orientation) (and the relative angular velocity
    /// towards zero).
    ///
    /// This is the 3-DOF generalization of the [`RevoluteJoint`] motor: instead of a scalar angle
    /// about the hinge axis, the position error is the shortest-arc world-space rotation vector from
    /// the current relative orientation to the target, and the velocity error is the full relative
    /// angular velocity. The combined PD response is applied as a soft positional correction along
    /// its own axis, mirroring [`align_orientation`](crate::dynamics::solver::xpbd::AngularConstraint::align_orientation).
    fn apply_motor(
        &self,
        body1: &mut SolverBody,
        body2: &mut SolverBody,
        inv_angular_inertia1: SymmetricTensor,
        inv_angular_inertia2: SymmetricTensor,
        solver_data: &mut SphericalJointSolverData,
        dt: Scalar,
    ) {
        let motor = &self.motor;

        if !motor.enabled {
            return;
        }

        // Reconstruct the live world-space joint frames from the prepared base + accumulated deltas.
        let frame1 = body1.delta_rotation.0 * solver_data.frame1;
        let frame2 = body2.delta_rotation.0 * solver_data.frame2;

        // Position error: the world-space rotation that should be applied to body2 (and the opposite
        // to body1) so that `frame1⁻¹ * frame2 == target_orientation`.
        let target_frame2 = frame1 * self.target_orientation;
        let mut error = target_frame2 * frame2.conjugate();
        // Choose the shortest arc.
        if error.w < 0.0 {
            error = -error;
        }
        let position_error = error.to_scaled_axis();

        // Velocity error: drive the relative angular velocity (body2 - body1) towards zero.
        let velocity_error = body1.angular_velocity - body2.angular_velocity;

        // The raw PD drive vector. Its *direction* is the correction axis; its magnitude is scaled
        // per the motor model below. For `ForceBased` this is a torque-like quantity (needs `w_sum`
        // to become a velocity change); for the others it is already a velocity change. This mirrors
        // the scalar revolute motor, but in 3-DOF along the instantaneous error axis.
        let drive = match motor.motor_model {
            MotorModel::SpringDamper {
                frequency,
                damping_ratio,
            } => {
                // Implicit Euler formulation for stable spring-damper behavior.
                let omega = TAU * frequency;
                let omega_sq = omega * omega;
                let two_zeta_omega = 2.0 * damping_ratio * omega;
                let inv_denominator = 1.0 / (1.0 + two_zeta_omega * dt + omega_sq * dt * dt);
                (omega_sq * position_error + two_zeta_omega * velocity_error) * dt * inv_denominator
            }
            MotorModel::AccelerationBased { stiffness, damping } => {
                damping * velocity_error + stiffness * position_error * dt
            }
            MotorModel::ForceBased { stiffness, damping } => {
                stiffness * position_error + damping * velocity_error
            }
        };

        let drive_magnitude = drive.length();
        if drive_magnitude <= Scalar::EPSILON {
            return;
        }
        let axis = drive / drive_magnitude;

        // Generalized inverse mass of the two bodies about the correction axis.
        let w1 = AngularConstraint::compute_generalized_inverse_mass(
            self,
            inv_angular_inertia1,
            axis,
        );
        let w2 = AngularConstraint::compute_generalized_inverse_mass(
            self,
            inv_angular_inertia2,
            axis,
        );
        let w_sum = w1 + w2;
        if w_sum <= Scalar::EPSILON {
            return;
        }

        // Target velocity change along `axis`. `ForceBased` converts force -> velocity change via
        // `w_sum` (which then cancels in `delta_lagrange`, matching the revolute motor); the other
        // models already express a velocity change.
        let target_velocity_change = match motor.motor_model {
            MotorModel::ForceBased { .. } => drive_magnitude * w_sum,
            _ => drive_magnitude,
        };

        let correction = target_velocity_change * dt;
        let mut delta_lagrange = correction / w_sum;

        // Clamp to limit the instantaneous torque per substep.
        if motor.max_torque < Scalar::MAX && motor.max_torque > 0.0 {
            let max_delta = motor.max_torque * dt * dt;
            delta_lagrange = delta_lagrange.clamp(-max_delta, max_delta);
        }

        solver_data.total_motor_lagrange += delta_lagrange * axis;

        // Positive `delta_lagrange` rotates body2 towards the target (and body1 away), reducing the error.
        self.apply_angular_lagrange_update(
            body1,
            body2,
            inv_angular_inertia1,
            inv_angular_inertia2,
            delta_lagrange,
            axis,
        );
    }

    /// Applies angle limits to limit the relative rotation of the bodies around the `swing_axis`.
    fn apply_swing_limits(
        &self,
        body1: &mut SolverBody,
        body2: &mut SolverBody,
        inertia1: &SolverBodyInertia,
        inertia2: &SolverBodyInertia,
        solver_data: &mut SphericalJointSolverData,
        dt: Scalar,
    ) {
        if let Some(joint_limit) = self.swing_limit {
            let a1 = body1.delta_rotation * solver_data.swing_axis1;
            let a2 = body2.delta_rotation * solver_data.swing_axis2;

            let n = a1.cross(a2);
            let n_magnitude = n.length();

            if n_magnitude <= Scalar::EPSILON {
                return;
            }

            let n = n / n_magnitude;

            if let Some(correction) = joint_limit.compute_correction(n, a1, a2, PI) {
                let inv_inertia1 = inertia1.effective_inv_angular_inertia();
                let inv_inertia2 = inertia2.effective_inv_angular_inertia();

                solver_data.total_swing_lagrange += self.align_orientation(
                    body1,
                    body2,
                    inv_inertia1,
                    inv_inertia2,
                    correction,
                    0.0,
                    self.swing_compliance,
                    dt,
                );
            }
        }
    }

    /// Applies angle limits to limit the relative rotation of the bodies around the `twist_axis`.
    fn apply_twist_limits(
        &self,
        body1: &mut SolverBody,
        body2: &mut SolverBody,
        inertia1: &SolverBodyInertia,
        inertia2: &SolverBodyInertia,
        solver_data: &mut SphericalJointSolverData,
        dt: Scalar,
    ) {
        if let Some(joint_limit) = self.twist_limit {
            let a1 = body1.delta_rotation * solver_data.swing_axis1;
            let a2 = body2.delta_rotation * solver_data.swing_axis2;

            let n = a1 + a2;
            let n_magnitude = n.length();

            if n_magnitude <= Scalar::EPSILON {
                return;
            }

            let b1 = body1.delta_rotation * solver_data.twist_axis1;
            let b2 = body2.delta_rotation * solver_data.twist_axis2;

            let n = n / n_magnitude;

            let n1 = b1 - n.dot(b1) * n;
            let n2 = b2 - n.dot(b2) * n;
            let n1_magnitude = n1.length();
            let n2_magnitude = n2.length();

            if n1_magnitude <= Scalar::EPSILON || n2_magnitude <= Scalar::EPSILON {
                return;
            }

            let n1 = n1 / n1_magnitude;
            let n2 = n2 / n2_magnitude;

            let max_correction = if a1.dot(a2) > -0.5 { 2.0 * PI } else { dt };

            if let Some(correction) = joint_limit.compute_correction(n, n1, n2, max_correction) {
                let inv_inertia1 = inertia1.effective_inv_angular_inertia();
                let inv_inertia2 = inertia2.effective_inv_angular_inertia();

                solver_data.total_twist_lagrange += self.align_orientation(
                    body1,
                    body2,
                    inv_inertia1,
                    inv_inertia2,
                    correction,
                    0.0,
                    self.twist_compliance,
                    dt,
                );
            }
        }
    }
}

impl PositionConstraint for SphericalJoint {}

impl AngularConstraint for SphericalJoint {}
