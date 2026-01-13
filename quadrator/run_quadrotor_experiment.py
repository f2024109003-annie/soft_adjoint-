import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from soft_adjoint_quadrotor import soft_loss, soft_grad
from dynamics import quadrotor_step

# ----------------------------
# Experiment configuration
# ----------------------------
E = 40          # ensemble size
runs = 128      # number of trials
dt = 0.01
T = 20.0
steps = int(T / dt)

# Initial state: hover
p0 = jnp.array([0.0, 0.0, 10.0])
v0 = jnp.zeros(3)
R0 = jnp.eye(3)
omega0 = jnp.zeros(3)
state0 = (p0, v0, R0, omega0)

# Control (constant thrust)
u0 = jnp.array([14.7])  # hover thrust

# Reference
p_ref = jnp.array([0.0, 0.0, 10.0])

# ----------------------------
# Synthetic ECMWF-style winds
# ----------------------------
def generate_wind_ensemble(E):
    return jnp.array([
        jnp.array([0.0, 0.0, np.random.normal(0, 2.0)])
        for _ in range(E)
    ])

# ----------------------------
# Classical baseline
# ----------------------------
def classical_run(wind, state):
    traj = []
    for _ in range(steps):
        traj.append(state[0])
        state = quadrotor_step(state, u0, wind, dt)
    traj = jnp.array(traj)
    rmse = jnp.sqrt(jnp.mean((traj - p_ref) ** 2))
    return rmse, traj

# ----------------------------
# Soft adjoint run
# ----------------------------
def soft_run(wind_ensemble, state):
    u = u0
    for _ in range(20):  # gradient descent steps
        grad_u = soft_grad(u, wind_ensemble, state)
        u = u - 0.1 * grad_u
    traj = []
    state_sim = state
    for _ in range(steps):
        traj.append(state_sim[0])
        state_sim = quadrotor_step(state_sim, u, wind_ensemble[0], dt)
    traj = jnp.array(traj)
    rmse = jnp.sqrt(jnp.mean((traj - p_ref) ** 2))
    return rmse, traj

# ----------------------------
# Run experiments
# ----------------------------
rmse_classical = []
rmse_soft = []

for _ in range(runs):
    winds = generate_wind_ensemble(E)

    r_c, traj_c = classical_run(winds[0], state0)
    r_s, traj_s = soft_run(winds, state0)

    rmse_classical.append(r_c)
    rmse_soft.append(r_s)

# ----------------------------
# Results (Table)
# ----------------------------
print("Classical RMSE:", np.mean(rmse_classical), "+-", np.std(rmse_classical))
print("Soft RMSE:", np.mean(rmse_soft), "+-", np.std(rmse_soft))

# ----------------------------
# Figure (Trajectory)
# ----------------------------
plt.figure(figsize=(8,5))
plt.plot(traj_c[:,2], label="Classical")
plt.plot(traj_s[:,2], label="Soft Adjoint")
plt.axhline(10.0, linestyle="--", color="k", label="Reference")
plt.xlabel("Time step")
plt.ylabel("Altitude (m)")
plt.legend()
plt.title("Quadrotor Hover under Ensemble Wind")
plt.tight_layout()
plt.show()

