import pygame
import heapq
import math
import collections
import os
import random

# --- Constants ---
# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BROWN = (139, 69, 19)
ORANGE = (255, 165, 0)

COLOR_ASTAR_PATH = (255, 195, 80)
COLOR_DIJKSTRA_PATH = (80, 180, 255)
COLOR_BFS_PATH = (180, 80, 220)
COLOR_GREEDY_PATH = (100, 220, 100)

# Grid settings
GRID_WIDTH = 800
GRID_HEIGHT = 600
CELL_SIZE = 32
COLS = GRID_WIDTH // CELL_SIZE
ROWS = GRID_HEIGHT // CELL_SIZE
# Costs
COST_NORMAL_CELL = 1
COST_TRAP_CELL = 10

# --- Global Variables for Sprites ---
SPRITES = {}
BACKGROUND_IMAGE = None


# --- Sprite Loading Function ---
def load_sprites_and_background():
    global SPRITES, BACKGROUND_IMAGE
    print("--- Starting to load sprites and background ---")

    base_path = os.path.dirname(__file__)
    img_dir = os.path.join(base_path, "images_game")
    if not os.path.exists(img_dir):
        try:
            os.makedirs(img_dir)
            print(f"Created directory: {img_dir}.")
        except OSError as e:
            print(f"Error creating dir {img_dir}: {e}")
            img_dir = base_path

    ground_path = os.path.join(img_dir, "ground.png")
    try:
        BACKGROUND_IMAGE = pygame.image.load(ground_path).convert()
        BACKGROUND_IMAGE = pygame.transform.scale(
            BACKGROUND_IMAGE, (GRID_WIDTH, GRID_HEIGHT)
        )
        print(f"Successfully loaded background: {ground_path}")
    except pygame.error as e:
        print(f"ERROR loading background {ground_path}: {e}")
        BACKGROUND_IMAGE = None

    map_element_files = {
        "wall": "wall.png",
        "trap": "trap.png",
        "start_flag": "start_flag.png",
        "end_flag": "end_flag.png",
    }
    for key_name, filename in map_element_files.items():
        path = os.path.join(img_dir, filename)
        try:
            image = pygame.image.load(path).convert_alpha()
            SPRITES[key_name] = pygame.transform.scale(image, (CELL_SIZE, CELL_SIZE))
            print(f"Loaded MAP sprite: '{key_name}' from {path}")
        except pygame.error as e:
            print(f"ERROR loading MAP sprite '{key_name}' from {path}: {e}")
            SPRITES[key_name] = None

    car_sprite_files = {
        "car_astar": "car_astar.png",
        "car_dijkstra": "car_dijkstra.png",
        "car_bfs": "car_bfs.png",
        "car_greedy": "car_greedy.png",
        "default_car": "default_car.png",
    }
    for key_name, filename in car_sprite_files.items():
        path = os.path.join(img_dir, filename)
        try:
            image = pygame.image.load(path).convert_alpha()
            SPRITES[key_name] = pygame.transform.scale(
                image, (int(CELL_SIZE * 0.9), int(CELL_SIZE * 0.9))
            )
            print(f"Loaded CAR sprite: '{key_name}' from {path}")
        except pygame.error as e:
            print(f"ERROR loading CAR sprite '{key_name}' from {path}: {e}")
            SPRITES[key_name] = None

    print("--- Finished loading sprites and background ---")
    if SPRITES.get("wall") is None:
        print("Warning: wall.png failed. Walls will be red.")
    if SPRITES.get("trap") is None:
        print("Warning: trap.png failed. Traps will be brown.")


# --- Graph Representation ---
class Graph:
    def __init__(self):
        self.edges = {}
        self.nodes = set()

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, from_node, to_node, weight):
        self.add_node(from_node)
        self.add_node(to_node)
        if from_node not in self.edges:
            self.edges[from_node] = {}
        self.edges[from_node][to_node] = weight

    def get_neighbors(self, node):
        return self.edges.get(node, {}).items()

    def get_edge_weight(self, from_node, to_node):
        return self.edges.get(from_node, {}).get(to_node, float("inf"))


# --- Heuristics ---
def heuristic_manhattan(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def heuristic_zero(node, goal):
    return 0


# --- Pathfinding Algorithms ---
def a_star_search(graph, start, goal, h_func=heuristic_manhattan):
    open_set = []
    heapq.heappush(open_set, (0 + h_func(start, goal), 0, start, [start]))
    closed_set_costs = {}
    expanded_nodes_count = 0
    while open_set:
        f_cost, g_cost, current_node, path = heapq.heappop(open_set)
        expanded_nodes_count += 1
        if current_node == goal:
            return path, g_cost, expanded_nodes_count
        if (
            current_node in closed_set_costs
            and closed_set_costs[current_node] <= g_cost
        ):
            continue
        closed_set_costs[current_node] = g_cost
        for neighbor, weight in graph.get_neighbors(current_node):
            new_g_cost = g_cost + weight
            if (
                neighbor not in closed_set_costs
                or new_g_cost < closed_set_costs[neighbor]
            ):
                new_f_cost = new_g_cost + h_func(neighbor, goal)
                heapq.heappush(
                    open_set, (new_f_cost, new_g_cost, neighbor, path + [neighbor])
                )
    return None, float("inf"), expanded_nodes_count


def dijkstra_search(graph, start, goal):
    return a_star_search(graph, start, goal, h_func=heuristic_zero)


def bfs_search(graph, start, goal):
    queue = collections.deque([(start, [start])])
    visited = {start}
    expanded_nodes_count = 0
    while queue:
        current_node, path = queue.popleft()
        expanded_nodes_count += 1
        if current_node == goal:
            actual_cost = 0
            for i in range(len(path) - 1):
                from_n, to_n = path[i], path[i + 1]
                cost_edge = graph.get_edge_weight(from_n, to_n)
                if cost_edge == float("inf"):
                    return None, float("inf"), expanded_nodes_count
                actual_cost += cost_edge
            return path, actual_cost, expanded_nodes_count
        for neighbor_node, _ in graph.get_neighbors(current_node):
            if neighbor_node not in visited:
                visited.add(neighbor_node)
                queue.append((neighbor_node, path + [neighbor_node]))
    return None, float("inf"), expanded_nodes_count


def greedy_bfs_search(graph, start, goal, h_func=heuristic_manhattan):
    open_set_with_cost = []
    heapq.heappush(open_set_with_cost, (h_func(start, goal), 0, start, [start]))
    closed_set = set()
    expanded_nodes_count = 0
    while open_set_with_cost:
        h_val, g_val_current, current_node, path = heapq.heappop(open_set_with_cost)
        expanded_nodes_count += 1
        if current_node == goal:
            return path, g_val_current, expanded_nodes_count
        if current_node in closed_set:
            continue
        closed_set.add(current_node)
        for neighbor, weight in graph.get_neighbors(current_node):
            if neighbor not in closed_set:
                new_g_cost = g_val_current + weight
                heapq.heappush(
                    open_set_with_cost,
                    (h_func(neighbor, goal), new_g_cost, neighbor, path + [neighbor]),
                )
    return None, float("inf"), expanded_nodes_count


# --- Pygame Grid and Drawing ---
class GridNode:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x_pixel = col * CELL_SIZE
        self.y_pixel = row * CELL_SIZE
        self.type = "normal"
        self.cost = COST_NORMAL_CELL
        self.is_player_path_node = False
        self.animation_timer = 0.0
        self.pulsate_alpha = 255
        self.pulsate_direction = -1  # -1 for dimming, 1 for brightening
        self.PULSATE_SPEED = 250  # Alpha units per second
        self.MIN_ALPHA = 100
        self.MAX_ALPHA = 255

    def get_map_element_sprite(self):
        if self.type == "obstacle":
            return SPRITES.get("wall")
        if self.type == "trap":
            return SPRITES.get("trap")
        if self.type == "start":
            return SPRITES.get("start_flag")
        if self.type == "end":
            return SPRITES.get("end_flag")
        return None

    def update_animation(self, dt):
        if self.type in ["start", "end", "trap"]:
            self.pulsate_alpha += self.pulsate_direction * self.PULSATE_SPEED * dt
            if self.pulsate_alpha < self.MIN_ALPHA:
                self.pulsate_alpha = self.MIN_ALPHA
                self.pulsate_direction = 1
            elif self.pulsate_alpha > self.MAX_ALPHA:
                self.pulsate_alpha = self.MAX_ALPHA
                self.pulsate_direction = -1
        else:  # Reset for other types if they were previously animated
            self.pulsate_alpha = self.MAX_ALPHA

    def draw(self, screen):
        element_sprite = self.get_map_element_sprite()
        if element_sprite:
            temp_sprite = (
                element_sprite.copy()
            )  # Important to copy for alpha modification
            if self.type in ["start", "end", "trap"]:
                temp_sprite.set_alpha(int(self.pulsate_alpha))
            screen.blit(temp_sprite, (self.x_pixel, self.y_pixel))
        elif (
            self.type != "normal"
        ):  # Fallback color if sprite missing AND not a normal cell
            color_map = {"obstacle": RED, "trap": BROWN, "start": GREEN, "end": BLUE}
            fallback_color = color_map.get(self.type, WHITE)

            rect_to_draw = pygame.Rect(self.x_pixel, self.y_pixel, CELL_SIZE, CELL_SIZE)
            if self.type in ["start", "end", "trap"]:
                s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                r, g, b = fallback_color
                s.fill((r, g, b, int(self.pulsate_alpha)))
                screen.blit(s, rect_to_draw.topleft)
            else:
                pygame.draw.rect(screen, fallback_color, rect_to_draw)

        if self.is_player_path_node:
            s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            s.fill((ORANGE[0], ORANGE[1], ORANGE[2], 100))
            screen.blit(s, (self.x_pixel, self.y_pixel))

    def make_obstacle(self):
        self.type = "obstacle"
        self.cost = float("inf")
        self.pulsate_alpha = self.MAX_ALPHA

    def make_start(self):
        self.type = "start"
        self.cost = COST_NORMAL_CELL
        self.pulsate_alpha = self.MAX_ALPHA

    def make_end(self):
        self.type = "end"
        self.cost = COST_NORMAL_CELL
        self.pulsate_alpha = self.MAX_ALPHA

    def make_trap(self):
        self.type = "trap"
        self.cost = COST_TRAP_CELL
        self.pulsate_alpha = self.MAX_ALPHA

    def reset(self):
        self.type = "normal"
        self.cost = COST_NORMAL_CELL
        self.is_player_path_node = False
        self.pulsate_alpha = self.MAX_ALPHA

    def is_obstacle_type(self):
        return self.type == "obstacle"

    def is_start_type(self):
        return self.type == "start"

    def is_end_type(self):
        return self.type == "end"


def make_grid(rows, cols):
    return [[GridNode(i, j) for j in range(cols)] for i in range(rows)]


def draw_grid_lines(screen):
    for i in range(ROWS + 1):
        pygame.draw.line(
            screen, GREY, (0, i * CELL_SIZE), (GRID_WIDTH, i * CELL_SIZE), 1
        )
    for j in range(COLS + 1):
        pygame.draw.line(
            screen, GREY, (j * CELL_SIZE, 0), (j * CELL_SIZE, GRID_HEIGHT), 1
        )


def draw_paths_and_agents(screen, agents_data, player_path_nodes=None):
    if player_path_nodes:
        for i in range(len(player_path_nodes) - 1):
            r1, c1 = player_path_nodes[i]
            r2, c2 = player_path_nodes[i + 1]
            x1, y1 = (c1 + 0.5) * CELL_SIZE, (r1 + 0.5) * CELL_SIZE
            x2, y2 = (c2 + 0.5) * CELL_SIZE, (r2 + 0.5) * CELL_SIZE
            pygame.draw.line(screen, ORANGE, (x1, y1), (x2, y2), 5)
    for data in agents_data:
        if data["path_nodes"]:
            path_nodes = data["path_nodes"]
            color = data["path_color"]
            for i in range(len(path_nodes) - 1):
                r1, c1 = path_nodes[i]
                r2, c2 = path_nodes[i + 1]
                x1, y1 = (c1 + 0.5) * CELL_SIZE, (r1 + 0.5) * CELL_SIZE
                x2, y2 = (c2 + 0.5) * CELL_SIZE, (r2 + 0.5) * CELL_SIZE
                pygame.draw.line(screen, color, (x1, y1), (x2, y2), 3)
        if data["agent_obj"]:
            data["agent_obj"].draw(screen)


def draw_main(
    screen,
    grid,
    agents_data_list,
    winner_text="",
    current_maze_text="",
    player_path_nodes=None,
    player_path_cost_text="",
    drawing_mode_text="",
):
    if BACKGROUND_IMAGE:
        screen.blit(BACKGROUND_IMAGE, (0, 0))
    else:
        screen.fill(WHITE)
    for r_list in grid:
        for node_item in r_list:
            node_item.draw(screen)
    draw_paths_and_agents(screen, agents_data_list, player_path_nodes)
    # draw_grid_lines(screen) # Uncomment if you want grid lines on top of background

    font_info = pygame.font.SysFont("arial", 20, bold=True)
    y_offset = 15
    texts_to_draw = [drawing_mode_text, winner_text, player_path_cost_text]
    for text_content in texts_to_draw:
        if text_content:
            text_surf = font_info.render(
                text_content, True, BLACK, (220, 220, 220, 200)
            )
            text_rect = text_surf.get_rect(centerx=GRID_WIDTH // 2, top=y_offset)
            screen.blit(text_surf, text_rect)
            y_offset += text_surf.get_height() + 5
    if current_maze_text:
        maze_text_surf = font_info.render(
            current_maze_text, True, BLACK, (220, 220, 220, 200)
        )
        maze_text_rect = maze_text_surf.get_rect(
            centerx=GRID_WIDTH // 2, bottom=GRID_HEIGHT - 15
        )
        screen.blit(maze_text_surf, maze_text_rect)
    pygame.display.flip()


def get_clicked_pos(pos):
    x, y = pos
    r = y // CELL_SIZE
    c = x // CELL_SIZE
    return r, c


def create_graph_from_grid(grid_nodes):
    graph = Graph()
    rows, cols = len(grid_nodes), len(grid_nodes[0])
    for r_idx in range(rows):
        for c_idx in range(cols):
            current_node_obj = grid_nodes[r_idx][c_idx]
            if not current_node_obj.is_obstacle_type():
                node = (r_idx, c_idx)
                graph.add_node(node)
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r_idx + dr, c_idx + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbor_node_obj = grid_nodes[nr][nc]
                        if not neighbor_node_obj.is_obstacle_type():
                            neighbor_node = (nr, nc)
                            cost_to_neighbor = neighbor_node_obj.cost
                            graph.add_edge(node, neighbor_node, cost_to_neighbor)
    return graph


# --- Agent Class ---
class Agent:
    def __init__(self, start_node_pos, sprite_key_in_dict, name="Agent"):
        self.row, self.col = start_node_pos
        self.x_center = self.col * CELL_SIZE + CELL_SIZE // 2
        self.y_center = self.row * CELL_SIZE + CELL_SIZE // 2
        self.name = name
        self.path = []
        self.current_path_index = 0
        self.speed = 2.5
        self.finished = False
        self.angle = 0

        actual_sprite = SPRITES.get(sprite_key_in_dict, SPRITES.get("default_car"))
        if actual_sprite:
            self.original_image = actual_sprite
            self.image_to_draw = self.original_image
        else:
            print(
                f"Agent '{name}': Sprite key '{sprite_key_in_dict}' or 'default_car' missing. Using fallback."
            )
            self.original_image = None
            self.image_to_draw = None
            self.fallback_color = (
                random.randint(0, 100),
                random.randint(0, 100),
                random.randint(0, 100),
            )

        self.dust_particles = []
        self.dust_emit_timer = 0.0
        self.dust_emit_interval = 0.05  # Seconds
        self.max_dust_particles = 30

    def set_path(self, path_nodes):
        self.path = path_nodes if path_nodes else []
        self.current_path_index = 0
        self.finished = False
        self.dust_particles.clear()  # Clear old dust
        if self.path:
            self.row, self.col = self.path[0]
            self.x_center = self.col * CELL_SIZE + CELL_SIZE // 2
            self.y_center = self.row * CELL_SIZE + CELL_SIZE // 2
            self.angle = 0
            if len(self.path) > 1:
                next_r, next_c = self.path[1]
                dx = (next_c * CELL_SIZE + CELL_SIZE // 2) - self.x_center
                dy = (next_r * CELL_SIZE + CELL_SIZE // 2) - self.y_center
                if dx != 0 or dy != 0:
                    self.angle = math.degrees(math.atan2(-dy, dx))
        else:
            self.finished = True

    def _emit_dust_particle(self):
        if len(self.dust_particles) < self.max_dust_particles:
            rad_angle = math.radians(
                self.angle + 180 + random.uniform(-15, 15)
            )  # Behind with some spread
            offset_dist = CELL_SIZE * 0.3
            offset_x = math.cos(rad_angle) * offset_dist
            offset_y = -math.sin(rad_angle) * offset_dist
            particle_x = self.x_center + offset_x
            particle_y = self.y_center + offset_y
            size = random.randint(2, 5)
            lifetime = random.uniform(0.2, 0.6)
            color_val = random.randint(180, 220)
            alpha_start = random.randint(100, 180)
            color_rgb = (color_val, color_val, color_val - 30)  # Greyish/Brownish
            vel_x = random.uniform(-20, 20)
            vel_y = random.uniform(-20, 20)  # Pixels/sec
            self.dust_particles.append(
                {
                    "x": particle_x,
                    "y": particle_y,
                    "size": size,
                    "lifetime": lifetime,
                    "initial_lifetime": lifetime,
                    "alpha_start": alpha_start,
                    "color_rgb": color_rgb,
                    "vel_x": vel_x,
                    "vel_y": vel_y,
                }
            )

    def _update_dust_particles(self, dt):
        new_particles = []
        for p in self.dust_particles:
            p["lifetime"] -= dt
            if p["lifetime"] > 0:
                p["x"] += p["vel_x"] * dt
                p["y"] += p["vel_y"] * dt
                p["current_alpha"] = int(
                    (p["lifetime"] / p["initial_lifetime"]) * p["alpha_start"]
                )
                p["current_size"] = max(
                    1, int(p["size"] * (p["lifetime"] / p["initial_lifetime"]))
                )
                new_particles.append(p)
        self.dust_particles = new_particles

    def update(self, dt):
        is_moving = False
        if (
            not self.finished
            and self.path
            and self.current_path_index < len(self.path) - 1
        ):
            target_r, target_c = self.path[self.current_path_index + 1]
            target_x_center = target_c * CELL_SIZE + CELL_SIZE // 2
            target_y_center = target_r * CELL_SIZE + CELL_SIZE // 2
            dx = target_x_center - self.x_center
            dy = target_y_center - self.y_center
            distance = math.sqrt(dx**2 + dy**2)

            if dx != 0 or dy != 0:
                self.angle = math.degrees(math.atan2(-dy, dx))

            effective_speed = self.speed * (
                dt * 60
            )  # Assuming speed is units per 1/60th sec
            if distance < effective_speed:
                self.x_center, self.y_center = target_x_center, target_y_center
                self.row, self.col = target_r, target_c
                self.current_path_index += 1
                if self.current_path_index >= len(self.path) - 1:
                    self.finished = True
                is_moving = True
            else:
                self.x_center += (dx / distance) * effective_speed
                self.y_center += (dy / distance) * effective_speed
                is_moving = True

            if self.original_image:
                self.image_to_draw = pygame.transform.rotate(
                    self.original_image, self.angle
                )
        elif (
            self.path
            and self.current_path_index >= len(self.path) - 1
            and not self.finished
        ):
            self.finished = True  # Ensure finished is set

        self.dust_emit_timer += dt
        if is_moving and self.dust_emit_timer >= self.dust_emit_interval:
            self.dust_emit_timer = 0
            self._emit_dust_particle()
        self._update_dust_particles(dt)

    def draw(self, screen):
        for p in self.dust_particles:  # Draw dust behind car
            if p["current_alpha"] > 0 and p["current_size"] > 0:
                particle_surf = pygame.Surface(
                    (p["current_size"] * 2, p["current_size"] * 2), pygame.SRCALPHA
                )
                pygame.draw.circle(
                    particle_surf,
                    (*p["color_rgb"], p["current_alpha"]),
                    (p["current_size"], p["current_size"]),
                    p["current_size"],
                )
                screen.blit(
                    particle_surf,
                    (int(p["x"] - p["current_size"]), int(p["y"] - p["current_size"])),
                )

        if self.image_to_draw:
            rect = self.image_to_draw.get_rect(
                center=(int(self.x_center), int(self.y_center))
            )
            screen.blit(self.image_to_draw, rect.topleft)
        elif self.original_image is None:
            pygame.draw.circle(
                screen,
                self.fallback_color,
                (int(self.x_center), int(self.y_center)),
                CELL_SIZE // 3,
            )

        font = pygame.font.SysFont("arial", 11, bold=True)
        text_surf = font.render(self.name, True, BLACK, (255, 255, 255, 180))
        text_rect = text_surf.get_rect(
            center=(int(self.x_center), int(self.y_center - CELL_SIZE * 0.65))
        )  # Raise name tag a bit
        screen.blit(text_surf, text_rect)


# --- Maze Patterns ---
MAZE_PATTERNS = {
    "Maze 1: Simple Wall": {
        "start": (1, 1),
        "end": (ROWS - 2, COLS - 2),
        "obstacles": [(r, COLS // 2) for r in range(1, ROWS - 1) if r != ROWS // 2],
        "traps": [],
    },
    "Maze 2: Trap Bridge": {
        "start": (1, 1),
        "end": (ROWS - 2, COLS - 2),
        "obstacles": [(r, COLS // 3) for r in range(0, ROWS // 2 - 1)]
        + [(r, COLS // 3) for r in range(ROWS // 2 + 2, ROWS)]
        + [(r, 2 * COLS // 3) for r in range(0, ROWS // 2 - 1)]
        + [(r, 2 * COLS // 3) for r in range(ROWS // 2 + 2, ROWS)],
        "traps": [(ROWS // 2, c) for c in range(COLS // 3, 2 * COLS // 3 + 1)],
    },
    "Maze 3: Spiral Trap": {
        "start": (1, 1),
        "end": (ROWS // 2, COLS // 2),
        "obstacles": [(r, 2) for r in range(1, ROWS - 2)]
        + [(ROWS - 3, c) for c in range(2, COLS - 2)]
        + [(r, COLS - 3) for r in range(2, ROWS - 2)]
        + [(2, c) for c in range(4, COLS - 3)],
        "traps": [(r, 4) for r in range(4, ROWS - 4)]
        + [(ROWS - 5, c) for c in range(4, COLS - 4)]
        + [(r, COLS - 5) for r in range(4, ROWS - 6)]
        + [(4, c) for c in range(6, COLS - 5)]
        + [(ROWS // 2, c) for c in range(6, COLS // 2)],
    },
    "Maze 4: Long Detour": {
        "start": (1, 1),
        "end": (1, COLS - 2),
        "obstacles": [(r, COLS // 2) for r in range(0, ROWS - 3)],
        "traps": [(r, COLS // 2 - 1) for r in range(ROWS // 2, ROWS - 1)]
        + [(r, COLS // 2 + 1) for r in range(ROWS // 2, ROWS - 1)],
    },
}
MAZE_NAMES = list(MAZE_PATTERNS.keys())


def load_maze(grid, pattern_name, agent_configs_ref, agents_info_list_ref):
    global start_node_obj, end_node_obj
    for r_list in grid:
        for node_item in r_list:
            node_item.reset()
    start_node_obj = None
    end_node_obj = None
    agents_info_list_ref.clear()
    if pattern_name not in MAZE_PATTERNS:
        print(f"Maze '{pattern_name}' not found.")
        return
    pattern = MAZE_PATTERNS[pattern_name]
    for r, c in pattern.get("obstacles", []):
        if 0 <= r < ROWS and 0 <= c < COLS:
            grid[r][c].make_obstacle()
    for r, c in pattern.get("traps", []):
        if 0 <= r < ROWS and 0 <= c < COLS:
            grid[r][c].make_trap()
    sr, sc = pattern["start"]
    if 0 <= sr < ROWS and 0 <= sc < COLS and grid[sr][sc].type == "normal":
        grid[sr][sc].make_start()
        start_node_obj = grid[sr][sc]
        for config in agent_configs_ref:
            agent = Agent(
                (sr, sc), sprite_key_in_dict=config["sprite_key"], name=config["name"]
            )
            agents_info_list_ref.append(
                {
                    **config,
                    "agent_obj": agent,
                    "path_nodes": None,
                    "cost": 0,
                    "expanded": 0,
                    "finished_race": False,
                }
            )
    else:
        print(f"Warning: Start pos { (sr,sc) } for '{pattern_name}' invalid.")
    er, ec = pattern["end"]
    if 0 <= er < ROWS and 0 <= ec < COLS and grid[er][ec].type == "normal":
        grid[er][ec].make_end()
        end_node_obj = grid[er][ec]
    else:
        print(f"Warning: End pos { (er,ec) } for '{pattern_name}' invalid.")
    print(f"Loaded {pattern_name}")


def calculate_path_cost(grid_ref, path_nodes_list):
    cost = 0
    if not path_nodes_list:
        return float("inf")
    # Cost is based on the cells entered. The start cell's cost is for "being there".
    # Subsequent cells add their cost as you "enter" them.
    for i, (r, c) in enumerate(path_nodes_list):
        if 0 <= r < ROWS and 0 <= c < COLS:
            cost += grid_ref[r][c].cost
        else:
            return float("inf")
    # If we only count cost of *transitions*, and not the start cell itself unless it's the only cell:
    # if len(path_nodes_list) > 1:
    #     cost -= grid_ref[path_nodes_list[0][0]][path_nodes_list[0][1]].cost # Subtract start cell cost if path has moves
    return cost


# --- Main Loop ---
def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT))
    pygame.display.set_caption("Pathfinding Arena FX - Player vs AI")
    load_sprites_and_background()  # Load after display mode is set
    clock = pygame.time.Clock()

    global start_node_obj, end_node_obj
    start_node_obj = None
    end_node_obj = None

    grid = make_grid(ROWS, COLS)
    agents_info_list = []
    winner_info_text = ""
    current_maze_name = "Custom"
    player_path_cost_text = ""
    current_drawing_mode_text = ""
    player_path = []
    player_drawing_mode = False

    agent_configs = [
        {
            "name": "A*",
            "algo_func": a_star_search,
            "path_color": COLOR_ASTAR_PATH,
            "sprite_key": "car_astar",
            "heuristic": heuristic_manhattan,
        },
        {
            "name": "Dijkstra",
            "algo_func": dijkstra_search,
            "path_color": COLOR_DIJKSTRA_PATH,
            "sprite_key": "car_dijkstra",
            "heuristic": None,
        },
        {
            "name": "BFS",
            "algo_func": bfs_search,
            "path_color": COLOR_BFS_PATH,
            "sprite_key": "car_bfs",
            "heuristic": None,
        },
        {
            "name": "Greedy",
            "algo_func": greedy_bfs_search,
            "path_color": COLOR_GREEDY_PATH,
            "sprite_key": "car_greedy",
            "heuristic": heuristic_manhattan,
        },
    ]
    setting_mode = "wall"

    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # Delta time in seconds

        mouse_pos = pygame.mouse.get_pos()
        clicked_row, clicked_col = get_clicked_pos(mouse_pos)
        current_drawing_mode_text = f"Build: {setting_mode.upper()} [W/T] | Player Draw: {'ON [P]' if player_drawing_mode else 'OFF [P]'} | Mazes [1-4] | Clear [C] | Run [SPACE]"

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            grid_modified_by_user_action = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # LEFT CLICK
                    if 0 <= clicked_row < ROWS and 0 <= clicked_col < COLS:
                        node = grid[clicked_row][clicked_col]
                        if player_drawing_mode:
                            if not player_path or player_path[-1] != (
                                clicked_row,
                                clicked_col,
                            ):
                                if node.type != "obstacle":
                                    player_path.append((clicked_row, clicked_col))
                                    node.is_player_path_node = True
                        else:
                            if not start_node_obj and node.type == "normal":
                                start_node_obj = node
                                node.make_start()
                                current_maze_name = "Custom"
                                agents_info_list.clear()
                                winner_info_text = ""
                                player_path_cost_text = ""
                                player_path.clear()
                                for r_list in grid:
                                    for n_item in r_list:
                                        n_item.is_player_path_node = False
                                for config in agent_configs:
                                    agent = Agent(
                                        (clicked_row, clicked_col),
                                        sprite_key_in_dict=config["sprite_key"],
                                        name=config["name"],
                                    )
                                    agents_info_list.append(
                                        {
                                            **config,
                                            "agent_obj": agent,
                                            "path_nodes": None,
                                            "cost": 0,
                                            "expanded": 0,
                                            "finished_race": False,
                                        }
                                    )
                                grid_modified_by_user_action = True
                            elif (
                                not end_node_obj
                                and node.type == "normal"
                                and node != start_node_obj
                            ):
                                end_node_obj = node
                                node.make_end()
                                current_maze_name = "Custom"
                                grid_modified_by_user_action = True
                            elif (
                                node.type == "normal"
                                and node != start_node_obj
                                and node != end_node_obj
                            ):
                                if setting_mode == "wall":
                                    node.make_obstacle()
                                elif setting_mode == "trap":
                                    node.make_trap()
                                current_maze_name = "Custom"
                                grid_modified_by_user_action = True
                elif event.button == 3:  # RIGHT CLICK
                    if 0 <= clicked_row < ROWS and 0 <= clicked_col < COLS:
                        node = grid[clicked_row][clicked_col]
                        if player_drawing_mode:
                            if player_path:
                                last_r, last_c = player_path.pop()
                                grid[last_r][last_c].is_player_path_node = False
                        else:
                            if node.is_start_type():
                                start_node_obj = None
                            elif node.is_end_type():
                                end_node_obj = None
                            node.reset()
                            current_maze_name = "Custom"
                            grid_modified_by_user_action = True
            if grid_modified_by_user_action:
                winner_info_text = ""
                player_path_cost_text = ""
                if not player_drawing_mode:
                    player_path.clear()
                    [
                        n.reset() for r in grid for n in r if n.is_player_path_node
                    ]  # Clear highlights only
                for agent_data in agents_info_list:
                    if agent_data["agent_obj"]:
                        agent_data["agent_obj"].set_path(None)
                    agent_data["path_nodes"] = None
                    agent_data["cost"] = 0
                    agent_data["expanded"] = 0
                    agent_data["finished_race"] = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start_node_obj and end_node_obj:
                    if player_drawing_mode:
                        player_drawing_mode = False
                        print("Player path drawing finished by SPACE.")
                        if (
                            player_path
                            and len(player_path) > 1
                            and player_path[0]
                            == (start_node_obj.row, start_node_obj.col)
                            and player_path[-1] == (end_node_obj.row, end_node_obj.col)
                        ):
                            cost_player = calculate_path_cost(grid, player_path)
                            player_path_cost_text = f"Player Cost: {cost_player if cost_player!=float('inf') else 'Invalid'}"
                        else:
                            player_path_cost_text = "Player path invalid."
                    current_graph = create_graph_from_grid(grid)
                    start_tuple = (start_node_obj.row, start_node_obj.col)
                    end_tuple = (end_node_obj.row, end_node_obj.col)
                    winner_info_text = ""
                    print(f"\n--- AI Pathfinding ({current_maze_name}) ---")
                    for agent_data in agents_info_list:
                        agent_data["finished_race_this_run"] = False
                        algo_func = agent_data["algo_func"]
                        path_nodes, cost, expanded_nodes = (None, float("inf"), 0)
                        try:
                            if agent_data["name"] in ["Dijkstra", "BFS"]:
                                path_nodes, cost, expanded_nodes = algo_func(
                                    current_graph, start_tuple, end_tuple
                                )
                            elif agent_data["heuristic"]:
                                path_nodes, cost, expanded_nodes = algo_func(
                                    current_graph,
                                    start_tuple,
                                    end_tuple,
                                    agent_data["heuristic"],
                                )
                        except Exception as e:
                            print(f"Error {agent_data['name']}:{e}")
                        agent_data["path_nodes"] = path_nodes
                        agent_data["cost"] = cost if cost != float("inf") else "N/A"
                        agent_data["expanded"] = expanded_nodes
                        print(
                            f"  {agent_data['name']}:Cost={agent_data['cost']},Exp={expanded_nodes},Path={'Yes' if path_nodes else 'No'}"
                        )
                        if agent_data["agent_obj"]:
                            agent_data["agent_obj"].set_path(path_nodes)
                elif event.key == pygame.K_p:
                    if start_node_obj and end_node_obj:
                        player_drawing_mode = not player_drawing_mode
                        if player_drawing_mode:
                            print("Player drawing ON.")
                            player_path_cost_text = ""
                            player_path = [(start_node_obj.row, start_node_obj.col)]
                            grid[start_node_obj.row][
                                start_node_obj.col
                            ].is_player_path_node = True
                            winner_info_text = ""
                            [
                                ad["agent_obj"].set_path(None)
                                for ad in agents_info_list
                                if ad["agent_obj"]
                            ]
                            [
                                ad.update(
                                    {"path_nodes": None, "finished_race_this_run": True}
                                )
                                for ad in agents_info_list
                            ]
                        else:
                            print("Player drawing OFF.")
                            if (
                                player_path
                                and len(player_path) > 1
                                and player_path[0]
                                == (start_node_obj.row, start_node_obj.col)
                                and player_path[-1]
                                == (end_node_obj.row, end_node_obj.col)
                            ):
                                cost_player = calculate_path_cost(grid, player_path)
                                player_path_cost_text = f"Player Cost: {cost_player if cost_player!=float('inf') else 'Invalid'}"
                            else:
                                player_path_cost_text = "Player path not finalized."
                    else:
                        print("Set Start & End first.")
                elif event.key == pygame.K_c:
                    start_node_obj = None
                    end_node_obj = None
                    winner_info_text = ""
                    current_maze_name = "Custom"
                    player_path.clear()
                    player_drawing_mode = False
                    player_path_cost_text = ""
                    grid = make_grid(ROWS, COLS)
                    agents_info_list.clear()
                elif event.key == pygame.K_w:
                    setting_mode = "wall"
                    print("Mode: Walls")
                elif event.key == pygame.K_t:
                    setting_mode = "trap"
                    print("Mode: Traps")
                maze_key_map = {
                    pygame.K_1: 0,
                    pygame.K_2: 1,
                    pygame.K_3: 2,
                    pygame.K_4: 3,
                }
                if event.key in maze_key_map:
                    maze_idx = maze_key_map[event.key]
                    if maze_idx < len(MAZE_NAMES):
                        player_path.clear()
                        player_drawing_mode = False
                        player_path_cost_text = ""
                        load_maze(
                            grid, MAZE_NAMES[maze_idx], agent_configs, agents_info_list
                        )
                        current_maze_name = MAZE_NAMES[maze_idx]
                        winner_info_text = ""

        for r_list in grid:  # Update all node animations
            for node_item in r_list:
                node_item.update_animation(dt)

        all_AIs_done_this_run = True
        current_valid_finishers_this_run = []
        if start_node_obj and end_node_obj and not player_drawing_mode:
            for agent_data in agents_info_list:
                if agent_data["agent_obj"]:
                    if not agent_data["agent_obj"].finished:
                        agent_data["agent_obj"].update(dt)
                        all_AIs_done_this_run = False
                    if (
                        agent_data["agent_obj"].finished
                        and agent_data["path_nodes"]
                        and not agent_data.get("finished_race_this_run", False)
                    ):
                        agent_data["finished_race_this_run"] = True
                        if agent_data["cost"] != "N/A":
                            current_valid_finishers_this_run.append(agent_data)
                elif not agent_data["path_nodes"] and not agent_data.get(
                    "finished_race_this_run", False
                ):
                    agent_data["finished_race_this_run"] = True
                elif not agent_data.get("finished_race_this_run", False):
                    all_AIs_done_this_run = False
            if all_AIs_done_this_run and not winner_info_text:
                if current_valid_finishers_this_run:
                    current_valid_finishers_this_run.sort(
                        key=lambda x: (
                            x["cost"]
                            if isinstance(x["cost"], (int, float))
                            else float("inf")
                        )
                    )
                    winner = current_valid_finishers_this_run[0]
                    winner_info_text = (
                        f"AI Winner: {winner['name']} (Cost: {winner['cost']})"
                    )
                    print(f"--- AI RACE FINISHED ({current_maze_name}) ---")
                    print(winner_info_text)
                    for i, f in enumerate(current_valid_finishers_this_run):
                        print(
                            f"  {i+1}. {f['name']}:Cost={f['cost']},Exp={f['expanded']}"
                        )
                elif any(ad["path_nodes"] for ad in agents_info_list):
                    winner_info_text = "No AI reached end."
                else:
                    winner_info_text = "No paths for AI."
            elif (
                all_AIs_done_this_run
                and not current_valid_finishers_this_run
                and not winner_info_text
            ):
                winner_info_text = "No AI successfully reached end."

        draw_main(
            screen,
            grid,
            agents_info_list,
            winner_info_text,
            current_maze_name,
            player_path,
            player_path_cost_text,
            current_drawing_mode_text,
        )
        # clock.tick(60) is handled by dt calculation at the start of the loop

    pygame.quit()


if __name__ == "__main__":
    main()
