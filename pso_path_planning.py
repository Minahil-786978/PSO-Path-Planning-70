import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PSO PATH PLANNING
# Roll Number: 70
# ============================================================

SEED = 70
np.random.seed(SEED)

# -----------------------------
# Grid settings
# -----------------------------
GRID_SIZE = 30

START = np.array([1.0, 1.0])
GOAL = np.array([28.0, 28.0])

# Number of intermediate points in each particle's path
N_WAYPOINTS = 12

# -----------------------------
# Generate obstacles
# -----------------------------
def create_obstacles():
    obstacles = []

    # Horizontal obstacles
    obstacles.extend([
        (5, 4, 18, 2),
        (8, 10, 16, 2),
        (3, 16, 14, 2),
        (17, 21, 10, 2),
    ])

    # Vertical obstacles
    obstacles.extend([
        (4, 6, 2, 9),
        (20, 3, 2, 10),
        (15, 13, 2, 8),
        (24, 18, 2, 8),
    ])

    return obstacles


OBSTACLES = create_obstacles()


# -----------------------------
# Check whether a point is
# inside an obstacle
# -----------------------------
def point_in_obstacle(point):
    x, y = point

    for ox, oy, width, height in OBSTACLES:
        if ox <= x <= ox + width and oy <= y <= oy + height:
            return True

    return False


# -----------------------------
# Check whether a line segment
# intersects an obstacle
# -----------------------------
def segment_collision(p1, p2):
    distance = np.linalg.norm(p2 - p1)
    steps = max(int(distance * 5), 2)

    for t in np.linspace(0, 1, steps):
        point = p1 + t * (p2 - p1)

        if point_in_obstacle(point):
            return True

    return False


# -----------------------------
# Check complete path
# -----------------------------
def path_collision(path):
    for i in range(len(path) - 1):
        if segment_collision(path[i], path[i + 1]):
            return True

    return False


# -----------------------------
# Calculate path length
# -----------------------------
def path_length(path):
    return np.sum(
        np.linalg.norm(np.diff(path, axis=0), axis=1)
    )


# -----------------------------
# Fitness function
# -----------------------------
def fitness(path):
    length = path_length(path)

    # Strong penalty for collision
    if path_collision(path):
        return length + 1000

    return length


# ============================================================
# PARTICLE SWARM OPTIMIZATION
# ============================================================

NUM_PARTICLES = 50
MAX_ITERATIONS = 200

W = 0.7
C1 = 1.5
C2 = 1.5


def run_pso():

    # Each particle contains N_WAYPOINTS x 2 coordinates
    particles = np.random.uniform(
        1,
        GRID_SIZE - 1,
        size=(NUM_PARTICLES, N_WAYPOINTS, 2)
    )

    velocities = np.random.uniform(
        -1,
        1,
        size=particles.shape
    )

    personal_best = particles.copy()

    personal_best_scores = np.array([
        fitness(np.vstack([START, p, GOAL]))
        for p in particles
    ])

    best_index = np.argmin(personal_best_scores)

    global_best = personal_best[best_index].copy()
    global_best_score = personal_best_scores[best_index]

    history = []

    for iteration in range(MAX_ITERATIONS):

        r1 = np.random.random(particles.shape)
        r2 = np.random.random(particles.shape)

        velocities = (
            W * velocities
            + C1 * r1 * (personal_best - particles)
            + C2 * r2 * (global_best - particles)
        )

        particles += velocities

        # Keep particles inside the grid
        particles = np.clip(
            particles,
            0.5,
            GRID_SIZE - 0.5
        )

        for i in range(NUM_PARTICLES):

            current_path = np.vstack([
                START,
                particles[i],
                GOAL
            ])

            current_score = fitness(current_path)

            # Update personal best
            if current_score < personal_best_scores[i]:

                personal_best[i] = particles[i].copy()
                personal_best_scores[i] = current_score

                # Update global best
                if current_score < global_best_score:

                    global_best = particles[i].copy()
                    global_best_score = current_score

        history.append(global_best_score)

    final_path = np.vstack([
        START,
        global_best,
        GOAL
    ])

    return final_path, history


# ============================================================
# RUN PSO
# ============================================================

best_path, history = run_pso()

print("=" * 50)
print("PSO PATH PLANNING")
print("=" * 50)
print(f"Seed / Roll Number : {SEED}")
print(f"Particles           : {NUM_PARTICLES}")
print(f"Iterations          : {MAX_ITERATIONS}")
print(f"Waypoints           : {N_WAYPOINTS}")
print(f"Path Length         : {path_length(best_path):.2f}")
print(f"Collision Free      : {not path_collision(best_path)}")
print("=" * 50)


# ============================================================
# VISUALIZATION
# ============================================================

fig, ax = plt.subplots(figsize=(9, 9))

# Draw obstacles
for ox, oy, width, height in OBSTACLES:

    rectangle = plt.Rectangle(
        (ox, oy),
        width,
        height
    )

    ax.add_patch(rectangle)


# Draw path
ax.plot(
    best_path[:, 0],
    best_path[:, 1],
    marker="o",
    linewidth=2,
    label="PSO Path"
)


# Start point
ax.scatter(
    START[0],
    START[1],
    s=150,
    marker="s",
    label="Start"
)


# Goal point
ax.scatter(
    GOAL[0],
    GOAL[1],
    s=150,
    marker="*",
    label="Goal"
)


ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)

ax.set_xlabel("X")
ax.set_ylabel("Y")

ax.set_title("Particle Swarm Optimization Path Planning")

ax.grid(True)
ax.legend()

plt.show()


# ============================================================
# CONVERGENCE GRAPH
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(history)

plt.xlabel("Iteration")
plt.ylabel("Best Fitness")

plt.title("PSO Convergence")

plt.grid(True)

plt.show()