import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
import gymnasium as gym
from gymnasium import spaces

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# KS Equation Solver using Spectral Methods
class KSSolver:
    def __init__(self, L=100, nx=512, nu=1, dt=0.002):  # Further reduced dt for stability
        self.L = L
        self.nx = nx
        self.nu = nu
        self.dx = L / nx
        self.x = np.linspace(0, L, nx, endpoint=False)
        self.k = np.fft.fftfreq(nx, d=self.dx) * 2 * np.pi
        self.FL = (self.k**2 - nu * self.k**4)
        self.FN = -0.5 * 1j * self.k
        self.dt = dt

    def step(self, u, control=0):
        u = np.clip(u, -8, 8)  # Stricter clipping to prevent blow-up
        u_hat = np.fft.fft(u)
        u2 = u**2
        u2_hat = np.fft.fft(u2)
        u_hat_next = (u_hat + self.dt * (1.5 * self.FN * u2_hat - 0.5 * self.FN * u2_hat)) / (1 - self.dt * self.FL)
        u_hat_next += control
        u_next = np.real(np.fft.ifft(u_hat_next))
        return u_next

    def initial_condition(self):
        return np.cos(2 * np.pi * self.x / self.L) + 0.1 * np.cos(4 * np.pi * self.x / self.L)

    def compute_gradient(self, u):
        u_hat = np.fft.fft(u)
        u_grad_hat = 1j * self.k * u_hat
        u_grad = np.real(np.fft.ifft(u_grad_hat))
        return u_grad

    def compute_time_derivative(self, u, control=0):
        u_next = self.step(u, control)
        u_dot = (u_next - u) / self.dt
        return u_dot

# Traditional Feedback Control
class TraditionalControl:
    def __init__(self, solver, gain=1.5, grad_gain=0.1):
        self.solver = solver
        self.gain = gain
        self.grad_gain = grad_gain

    def compute_control(self, u):
        u_grad = self.solver.compute_gradient(u)
        return -self.gain * u - self.grad_gain * u_grad

# Enhanced DeepONet Implementation
class DeepONet(nn.Module):
    def __init__(self, branch_input_dim=512, trunk_input_dim=1, hidden_dim=256, output_dim=512):
        super(DeepONet, self).__init__()
        # Deeper and wider branch network
        self.branch_net = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Deeper trunk network
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, branch_input, trunk_input):
        branch_out = self.branch_net(branch_input)
        trunk_out = self.trunk_net(trunk_input)
        combined = branch_out * trunk_out
        output = self.output_layer(combined)
        return output

# Generate more diverse training data for DeepONet
def generate_deeponet_data(solver, num_trajectories=200, t_max=15):
    branch_inputs = []
    trunk_inputs = []
    outputs = []
    for _ in range(num_trajectories):
        u = solver.initial_condition() + np.random.randn(solver.nx) * 0.2  # Increased noise
        u_grad = solver.compute_gradient(u)
        branch_inputs.append(u)
        t = np.random.uniform(0, t_max)
        trunk_inputs.append(np.array([t]))
        control = -u - 0.1 * u_grad
        outputs.append(control)
    return (np.array(branch_inputs), np.array(trunk_inputs), np.array(outputs))

# Train DeepONet with more epochs
def train_deeponet(deeponet, branch_inputs, trunk_inputs, outputs, epochs=1500):
    optimizer = torch.optim.Adam(deeponet.parameters(), lr=0.0005)  # Reduced learning rate
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        branch_tensor = torch.FloatTensor(branch_inputs)
        trunk_tensor = torch.FloatTensor(trunk_inputs)
        output_tensor = torch.FloatTensor(outputs)
        pred = deeponet(branch_tensor, trunk_tensor)
        loss = criterion(pred, output_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Custom Gym Environment for KS Control
class KSEnv(gym.Env):
    def __init__(self, solver, deeponet):
        super(KSEnv, self).__init__()
        self.solver = solver
        self.deeponet = deeponet
        self.action_space = spaces.Box(low=-1, high=1, shape=(solver.nx,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5, high=5, shape=(solver.nx,), dtype=np.float32)
        self.state = None
        self.prev_state = None
        self.step_count = 0
        self.max_steps = 200
        # Compute typical gradient and time derivative magnitudes
        grad_mags = []
        u_dot_mags = []
        for _ in range(20):  # More samples for better statistics
            u_init = solver.initial_condition() + np.random.randn(solver.nx) * 0.2
            grad_init = solver.compute_gradient(u_init)
            grad_mags.append(np.mean(grad_init**2))
            u_dot_init = solver.compute_time_derivative(u_init)
            u_dot_mags.append(np.mean(u_dot_init**2))
        self.grad_scale = np.mean(grad_mags) + np.std(grad_mags)  # Add std for robustness
        self.u_dot_scale = np.mean(u_dot_mags) + np.std(u_dot_mags)
        print(f"Computed grad_scale: {self.grad_scale:.4f}, u_dot_scale: {self.u_dot_scale:.4f}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.solver.initial_condition()
        self.prev_state = self.state.copy()
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        with torch.no_grad():
            branch_input = torch.FloatTensor(self.state).unsqueeze(0)
            trunk_input = torch.FloatTensor([self.step_count * self.solver.dt]).unsqueeze(0)
            deeponet_action = self.deeponet(branch_input, trunk_input).numpy().squeeze()
        action = deeponet_action + action
        action = np.clip(action, -1, 1)
        self.prev_state = self.state.copy()
        self.state = self.solver.step(self.state, control=action)
        state_energy = np.mean(self.state**2)
        action_magnitude = np.mean(action**2)
        grad = self.solver.compute_gradient(self.state)
        grad_magnitude = np.mean(grad**2)
        u_dot = self.solver.compute_time_derivative(self.state, control=action)
        u_dot_magnitude = np.mean(u_dot**2)
        # Adjusted reward weights
        state_term = state_energy / 0.5
        action_term = action_magnitude / 1.0
        grad_term = grad_magnitude / self.grad_scale
        u_dot_term = u_dot_magnitude / self.u_dot_scale
        reward = -state_term - 0.005 * action_term - 0.002 * grad_term - 0.001 * u_dot_term  # Reduced penalties
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps
        info = {
            "action_magnitude": np.mean(np.abs(action)),
            "state_energy": state_energy,
            "action_penalty": 0.005 * action_magnitude,
            "grad_penalty": 0.002 * grad_magnitude,
            "u_dot_penalty": 0.001 * u_dot_magnitude,
            "reward": reward,
        }
        return self.state, reward, terminated, truncated, info

    def render(self):
        pass

# Performance Comparison
def compare_control_strategies(solver, deeponet, rl_model, traditional_control, num_steps=200):
    initial_state = solver.initial_condition()
    rl_states = [initial_state.copy()]
    trad_states = [initial_state.copy()]
    rl_energy = [np.mean(initial_state**2)]
    trad_energy = [np.mean(initial_state**2)]
    rl_action_mags = []
    reward_components = []

    obs = initial_state.copy()
    for _ in range(num_steps):
        action, _ = rl_model.predict(obs, deterministic=True)
        with torch.no_grad():
            branch_input = torch.FloatTensor(obs).unsqueeze(0)
            trunk_input = torch.FloatTensor([len(rl_states) * solver.dt]).unsqueeze(0)
            deeponet_action = deeponet(branch_input, trunk_input).numpy().squeeze()
        action = deeponet_action + action
        obs, _, done, info = env.step(action)
        obs = obs[0]
        done = done[0]
        info = info[0]
        rl_states.append(obs.copy())
        rl_energy.append(np.mean(obs**2))
        rl_action_mags.append(info["action_magnitude"])
        reward_components.append({
            "state_term": -info["state_energy"] / 0.5,
            "action_term": -0.005 * info["action_magnitude"],
            "grad_term": -0.002 * info["grad_penalty"],
            "u_dot_term": -0.001 * info["u_dot_penalty"],
            "total_reward": info["reward"],
        })
        if done:
            break

    # Log reward components for the last step
    last_reward = reward_components[-1]
    print("Reward components at last step:")
    for key, value in last_reward.items():
        print(f"{key}: {value:.4f}")

    state = initial_state.copy()
    for _ in range(num_steps):
        control = traditional_control.compute_control(state)
        state = solver.step(state, control=control)
        trad_states.append(state.copy())
        trad_energy.append(np.mean(state**2))

    print(f"Average RL action magnitude: {np.mean(rl_action_mags):.4f}")
    return np.array(rl_states), np.array(trad_states), np.array(rl_energy), np.array(trad_energy)

# Plotting Results
def plot_comparison(rl_states, trad_states, rl_energy, trad_energy, solver):
    try:
        t = np.arange(rl_states.shape[0]) * solver.dt
        x = solver.x
        T, X = np.meshgrid(t, x)

        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        plt.contourf(X, T, rl_states.T, cmap='jet')
        plt.colorbar(label='u(x,t)')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('DeepONet-Guided RL Control')

        plt.subplot(2, 2, 2)
        plt.contourf(X, T, trad_states.T, cmap='jet')
        plt.colorbar(label='u(x,t)')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Traditional Feedback Control')

        plt.subplot(2, 1, 2)
        plt.plot(t, rl_energy, label='DeepONet-Guided RL', color='blue')
        plt.plot(t, trad_energy, label='Traditional Control', color='red')
        plt.xlabel('Time')
        plt.ylabel('Energy (Mean Squared State)')
        plt.title('Energy Over Time')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('ks_control_ddpg_improved.png')
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

# Main Execution
def main():
    solver = KSSolver()
    deeponet = DeepONet()
    traditional_control = TraditionalControl(solver, gain=1.5, grad_gain=0.1)

    branch_inputs, trunk_inputs, outputs = generate_deeponet_data(solver)
    train_deeponet(deeponet, branch_inputs, trunk_inputs, outputs)

    global env
    env = KSEnv(solver, deeponet)
    env = make_vec_env(lambda: env, n_envs=1, seed=42)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))  # Reduced noise
    try:
        model = DDPG(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            verbose=1,
            buffer_size=20000,  # Increased buffer size
            seed=42,
            learning_rate=0.00005,  # Reduced learning rate
            policy_kwargs=dict(net_arch=[512, 512]),  # Larger policy network
        )
        model.learn(total_timesteps=50000)  # Increased training steps
    except Exception as e:
        print(f"RL training failed: {e}")
        return

    rl_states, trad_states, rl_energy, trad_energy = compare_control_strategies(solver, deeponet, model, traditional_control)
    plot_comparison(rl_states, trad_states, rl_energy, trad_energy, solver)

if __name__ == "__main__":
    main()
