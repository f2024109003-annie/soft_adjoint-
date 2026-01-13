import jax.numpy as jnp
from jax import grad, jit
from dynamics import quadrotor_step

# ----------------------------
# Simulation parameters
# ----------------------------
dt = 0.01
T_final = 20.0
steps = int(T_final / dt)

# ----------------------------
# Reference trajectory (hover)
# ----------------------------
def reference_position(t):
    return jnp.array([0.0, 0.0, 10.0])  # 10m hover


# ----------------------------
# Single rollout cost
# ----------------------------
def single_rollout_cost(u, wind, state0):
    """
    Cost for one wind realization
    """
    state = state0
    cost = 0.0

    for k in range(steps):
        t = k * dt
        p_ref = reference_position(t)

        p, v, R, omega = state
        cost += jnp.sum((p - p_ref) ** 2)

        state = quadrotor_step(state, u, wind, dt)

    return cost / steps


# ----------------------------
# Soft adjoint loss (ensemble)
# ----------------------------
def soft_loss(u, wind_ensemble, state0, weights=None):
    """
    Soft adjoint loss: weighted ensemble average
    """
    E = len(wind_ensemble)

    if weights is None:
        weights = jnp.ones(E) / E

    losses = jnp.array([
        single_rollout_cost(u, wind_ensemble[e], state0)
        for e in range(E)
    ])

    return jnp.sum(weights * losses)


# ----------------------------
# Soft adjoint gradient
# ----------------------------
soft_grad = jit(grad(soft_loss))
