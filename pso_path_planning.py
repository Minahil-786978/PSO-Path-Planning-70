import random
import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PSO PATH PLANNING
# ============================================================

ROLL_NUMBER = 70

random.seed(ROLL_NUMBER)
np.random.seed(ROLL_NUMBER)


# ============================================================
# 1. GENERATE UNIQUE PROBLEM
# ============================================================

def generate_problem():
    """
    Generate a unique grid, obstacle layout, start point,
    and goal point using the roll number as the seed.
    """

    # Grid size is generated from the seed
    grid_size = random.randint(20, 30)

    all_cells = [
        (x, y)
        for x in range(grid_size)
        for y in range(grid_size)
    ]

    # Generate 12% to 18% obstacles
    min_obstacles = int(grid_size * grid_size * 0.12)
    max_obstacles = int(grid_size * grid_size * 0.18)

    obstacle_count = random.randint(
        min_obstacles,
        max_obstacles
    )

    obstacles = set(
        random.sample(
            all_cells,
            obstacle_count
        )
    )

    # Select start and goal from free cells
    free_cells = [
        cell
        for cell in all_cells
        if cell not in obstacles
    ]

    start = random.choice(free_cells)

    remaining_free = [
        cell
        for cell in free_cells
        if cell != start
    ]

    goal = random.choice(remaining_free)

    return (
        grid_size,
        obstacles,
        np.array(start, dtype=float),
        np.array(goal, dtype=float)
    )


GRID_SIZE, OBSTACLES, START, GOAL = generate_problem()


# ============================================================
# 2. PATH REPRESENTATION
# ============================================================

NUM_WAYPOINTS = 5


def build_path(waypoints):
    """Build a complete path from start through waypoints to goal."""

    return np.vstack([
        START,
        waypoints,
        GOAL
    ])


# ============================================================
# 3. FAST COLLISION CHECKING
# ============================================================

def point_in_obstacle(point):
    """Check whether a point belongs to an obstacle cell."""

    x = int(round(point[0]))
    y = int(round(point[1]))

    if x < 0 or x >= GRID_SIZE:
        return True

    if y < 0 or y >= GRID_SIZE:
        return True

    return (x, y) in OBSTACLES


def segment_collision(point1, point2):
    """
    Check whether a line segment crosses an obstacle.

    The segment is sampled at grid-level resolution, making
    collision checking much faster than checking every
    obstacle distance separately.
    """

    distance = np.linalg.norm(point2 - point1)

    steps = max(
        int(distance * 2),
        2
    )

    for t in np.linspace(0, 1, steps + 1):

        point = (
            point1
            + t * (point2 - point1)
        )

        if point_in_obstacle(point):
            return True

    return False


def path_collision(path):
    """Return True if any segment of the path hits an obstacle."""

    for i in range(len(path) - 1):

        if segment_collision(
            path[i],
            path[i + 1]
        ):
            return True

    return False


# ============================================================
# 4. PATH LENGTH
# ============================================================

def path_length(path):
    """Calculate the total Euclidean path length."""

    differences = np.diff(
        path,
        axis=0
    )

    distances = np.linalg.norm(
        differences,
        axis=1
    )

    return np.sum(distances)


# ============================================================
# 5. FITNESS FUNCTION
# ============================================================

def fitness(path):
    """
    Lower fitness is better.

    Collision-free paths are strongly preferred.
    """

    length = path_length(path)

    if path_collision(path):
        return length + 1000

    return length


# ============================================================
# 6. INITIALIZE PARTICLES
# ============================================================

def initialize_particles():

    particles = np.zeros(
        (
            NUM_PARTICLES,
            NUM_WAYPOINTS,
            2
        )
    )

    # Generate particles around the direct path
    base_path = np.linspace(
        START,
        GOAL,
        NUM_WAYPOINTS + 2
    )[1:-1]

    for i in range(NUM_PARTICLES):

        noise_scale = (
            GRID_SIZE * 0.15
        )

        noise = np.random.normal(
            0,
            noise_scale,
            size=(
                NUM_WAYPOINTS,
                2
            )
        )

        particles[i] = (
            base_path + noise
        )

    # Keep all particles inside the grid
    particles = np.clip(
        particles,
        0,
        GRID_SIZE - 1
    )

    return particles


# ============================================================
# 7. PSO PARAMETERS
# ============================================================

NUM_PARTICLES = 40
MAX_ITERATIONS = 150

COGNITIVE = 1.6
SOCIAL = 1.6


# ============================================================
# 8. PARTICLE SWARM OPTIMIZATION
# ============================================================

def run_pso():

    particles = initialize_particles()

    velocities = np.random.uniform(
        -1,
        1,
        size=particles.shape
    )

    personal_best = particles.copy()

    personal_best_scores = np.array([
        fitness(
            build_path(particle)
        )
        for particle in particles
    ])

    best_index = np.argmin(
        personal_best_scores
    )

    global_best = (
        personal_best[
            best_index
        ].copy()
    )

    global_best_score = (
        personal_best_scores[
            best_index
        ]
    )

    # Track best collision-free solution
    best_collision_free_path = None
    best_collision_free_length = float("inf")

    fitness_history = []

    stagnant_iterations = 0

    for iteration in range(
        MAX_ITERATIONS
    ):

        # Decrease inertia during optimization
        inertia = (
            0.9
            - 0.5
            * iteration
            / MAX_ITERATIONS
        )

        r1 = np.random.random(
            particles.shape
        )

        r2 = np.random.random(
            particles.shape
        )

        # PSO velocity update
        velocities = (
            inertia * velocities
            + COGNITIVE
            * r1
            * (
                personal_best
                - particles
            )
            + SOCIAL
            * r2
            * (
                global_best
                - particles
            )
        )

        # Limit particle movement
        maximum_velocity = (
            GRID_SIZE * 0.20
        )

        velocities = np.clip(
            velocities,
            -maximum_velocity,
            maximum_velocity
        )

        # Update particle positions
        particles += velocities

        particles = np.clip(
            particles,
            0,
            GRID_SIZE - 1
        )

        improved = False

        # Evaluate particles
        for i in range(NUM_PARTICLES):

            current_path = build_path(
                particles[i]
            )

            current_score = fitness(
                current_path
            )

            # Save collision-free solution
            if not path_collision(
                current_path
            ):

                current_length = (
                    path_length(
                        current_path
                    )
                )

                if (
                    current_length
                    < best_collision_free_length
                ):

                    best_collision_free_length = (
                        current_length
                    )

                    best_collision_free_path = (
                        current_path.copy()
                    )

            # Update personal best
            if (
                current_score
                < personal_best_scores[i]
            ):

                personal_best[i] = (
                    particles[i].copy()
                )

                personal_best_scores[i] = (
                    current_score
                )

                improved = True

                # Update global best
                if (
                    current_score
                    < global_best_score
                ):

                    global_best = (
                        particles[i].copy()
                    )

                    global_best_score = (
                        current_score
                    )

        if improved:
            stagnant_iterations = 0
        else:
            stagnant_iterations += 1

        # Restart a few particles if swarm stagnates
        if stagnant_iterations >= 15:

            reset_count = max(
                1,
                NUM_PARTICLES // 5
            )

            reset_indices = np.random.choice(
                NUM_PARTICLES,
                reset_count,
                replace=False
            )

            for index in reset_indices:

                particles[index] = np.random.uniform(
                    0,
                    GRID_SIZE - 1,
                    size=(
                        NUM_WAYPOINTS,
                        2
                    )
                )

                velocities[index] = np.random.uniform(
                    -1,
                    1,
                    size=(
                        NUM_WAYPOINTS,
                        2
                    )
                )

            stagnant_iterations = 0

        fitness_history.append(
            global_best_score
        )

    # Use the best collision-free solution found
    if best_collision_free_path is not None:

        return (
            best_collision_free_path,
            fitness_history
        )

    # Fallback
    return (
        build_path(global_best),
        fitness_history
    )


# ============================================================
# 9. SIMPLIFY FINAL PATH
# ============================================================

def simplify_path(path):
    """
    Remove unnecessary waypoints whenever two points can
    be connected directly without hitting an obstacle.
    """

    simplified = [
        path[0]
    ]

    current = 0

    while current < len(path) - 1:

        furthest = current + 1

        for candidate in range(
            current + 1,
            len(path)
        ):

            if not segment_collision(
                path[current],
                path[candidate]
            ):

                furthest = candidate

            else:

                break

        simplified.append(
            path[furthest]
        )

        current = furthest

    return np.array(
        simplified
    )


# ============================================================
# 10. RUN PSO
# ============================================================

print("Running PSO path planning...")
print("Please wait...")

best_path, fitness_history = run_pso()

best_path = simplify_path(
    best_path
)

final_length = path_length(
    best_path
)

collision_free = not path_collision(
    best_path
)


# ============================================================
# 11. CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# 12. PRINT RESULTS
# ============================================================

print()
print("=" * 65)
print("PSO PATH PLANNING RESULTS")
print("=" * 65)

print(
    f"Roll Number / Seed : {ROLL_NUMBER}"
)

print(
    f"Generated Grid     : "
    f"{GRID_SIZE} x {GRID_SIZE}"
)

print(
    f"Obstacle Count     : "
    f"{len(OBSTACLES)}"
)

print(
    f"Start Point        : "
    f"{tuple(START.astype(int))}"
)

print(
    f"Goal Point         : "
    f"{tuple(GOAL.astype(int))}"
)

print(
    f"Particles          : "
    f"{NUM_PARTICLES}"
)

print(
    f"Iterations         : "
    f"{MAX_ITERATIONS}"
)

print(
    f"PSO Waypoints      : "
    f"{NUM_WAYPOINTS}"
)

print(
    f"Final Path Length  : "
    f"{final_length:.2f}"
)

print(
    f"Collision Free     : "
    f"{collision_free}"
)

print("=" * 65)

print()
print("Best Path:")

for point in best_path:

    print(
        f"({point[0]:.2f}, "
        f"{point[1]:.2f})"
    )


# ============================================================
# 13. FINAL PATH VISUALIZATION
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 9)
)


# Obstacles
for x, y in OBSTACLES:

    obstacle = plt.Rectangle(
        (
            x - 0.5,
            y - 0.5
        ),
        1,
        1,
        facecolor="gray",
        edgecolor="black"
    )

    ax.add_patch(
        obstacle
    )


# Path
ax.plot(
    best_path[:, 0],
    best_path[:, 1],
    marker="o",
    linewidth=2.5,
    label="PSO Path",
    zorder=4
)


# Start
ax.scatter(
    START[0],
    START[1],
    s=180,
    marker="s",
    label="Start",
    zorder=5
)


# Goal
ax.scatter(
    GOAL[0],
    GOAL[1],
    s=220,
    marker="*",
    label="Goal",
    zorder=5
)


# Grid limits
ax.set_xlim(
    -0.5,
    GRID_SIZE - 0.5
)

ax.set_ylim(
    -0.5,
    GRID_SIZE - 0.5
)


# Grid ticks
ax.set_xticks(
    range(GRID_SIZE)
)

ax.set_yticks(
    range(GRID_SIZE)
)

ax.grid(
    True,
    alpha=0.4
)

ax.set_aspect(
    "equal"
)

ax.set_xlabel(
    "X Coordinate"
)

ax.set_ylabel(
    "Y Coordinate"
)

ax.set_title(
    f"PSO Path Planning | "
    f"Seed: {ROLL_NUMBER} | "
    f"Path Length: {final_length:.2f} | "
    f"Collision Free: {collision_free}"
)


# Legend outside the grid
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1)
)

plt.tight_layout()

plt.savefig(
    "results/path.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 14. CONVERGENCE GRAPH
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    fitness_history,
    linewidth=2
)

plt.xlabel(
    "Iteration"
)

plt.ylabel(
    "Best Fitness"
)

plt.title(
    f"PSO Convergence | Seed: {ROLL_NUMBER}"
)

plt.grid(
    True,
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    "results/convergence.png",
    dpi=150
)

plt.show()