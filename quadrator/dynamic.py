# quadrotor_6dof/dynamics.py

import jax.numpy as jnp

# -----------------------------
# Physical parameters
# -----------------------------
m = 1.5  # mass (kg)
g = 9.81
J = jnp.diag(jnp.array([0.03, 0.03, 0.05]))  # inertia matrix
J_inv = jnp.linalg.inv(J)

e3 = jnp.array([0.0, 0.0, 1.0])


# -----------------------------
# Helper: skew-symmetric matrix
def skew(omega):
    """
    Convert angular velocity vector into skew-symmetric matrix
    """
    wx, wy, wz = omega
    return jnp.array([
        [0.0, -wz,  wy],
        [wz,  0.0, -wx],
        [-wy, wx,  0.0]
    ])


def quadrotor_step(state, control, wind, dt=0.01):
    """
    One integration step of 6DOF quadrotor dynamics

    state = (p, v, R, omega)
    control = (T, tau)
    wind = wind disturbance vector
    """

    p, v, R, omega = state
    T, tau = control

    # Translational dynamics
    dp = v
    dv = (T / m) * (R @ e3) - g * e3 + wind

    # Rotational dynamics
    dR = R @ skew(omega)
    domega = J_inv @ (tau - jnp.cross(omega, J @ omega))

    # Euler integration
    p_next = p + dt * dp
    v_next = v + dt * dv
    R_next = R + dt * dR
    omega_next = omega + dt * domega

    return (p_next, v_next, R_next, omega_next)
