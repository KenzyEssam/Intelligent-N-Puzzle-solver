import heapq
import random
import time
import numpy as np
import pygame
from pygame.locals import QUIT


class PuzzleState:
    def __init__(self, puzzle, parent=None, move=None, cost=0):
        # PuzzleState represents a state in the puzzle solving process
        self.puzzle = puzzle
        self.parent = parent
        self.move = move
        self.cost = cost
        self.size = puzzle.shape[0]
        self.heuristic = self.manhattan_distance() + self.misplaced_tiles() + self.linear_conflict() + self.max_swap()


    def __lt__(self, other):
        # Comparison method for priority queue in best_first_search
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

    def __eq__(self, other):
        # Equality check for states
        return np.array_equal(self.puzzle, other.puzzle)

    def get_blank_position(self):
        # Get the position of the blank (0) in the puzzle
        return np.argwhere(self.puzzle == 0)[0]

    def manhattan_distance(self):
        # Calculate the Manhattan distance heuristic
        goal = np.arange(self.size ** 2).reshape((self.size, self.size))
        distance = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.puzzle[i, j] != 0:
                    goal_position = np.argwhere(goal == self.puzzle[i, j])[0]
                    distance += abs(i - goal_position[0]) + abs(j - goal_position[1])
        return distance

    def misplaced_tiles(self):
        # Calculate the Misplaced Tiles heuristic
        goal = np.arange(self.size ** 2).reshape((self.size, self.size))
        return np.sum(self.puzzle != goal)

    def linear_conflict(self):
        # Calculate the Linear Conflict heuristic
        conflicts = 0
        for i in range(self.size):
            row = self.puzzle[i, :]
            col = self.puzzle[:, i]
            conflicts += self.count_conflicts(row) + self.count_conflicts(col)
        return conflicts

    def count_conflicts(self, line):
        # Count conflicts in a line for Linear Conflict heuristic
        conflicts = 0
        max_val = -1
        for i in range(len(line)):
            if line[i] != 0 and line[i] > max_val:
                max_val = line[i]
            elif line[i] != 0 and line[i] < max_val:
                conflicts += 2
        return conflicts

    def max_swap(self):
        # Calculate the Max Swap heuristic
        goal = np.arange(self.size ** 2).reshape((self.size, self.size))
        max_swaps = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.puzzle[i, j] != 0:
                    goal_position = np.argwhere(goal == self.puzzle[i, j])[0]
                    # Check if the tile is not in its correct row
                    if i != goal_position[0]:
                        max_swaps += 1
                    # Check if the tile is not in its correct column
                    if j != goal_position[1]:
                        max_swaps += 1
        return max_swaps

def generate_random_puzzle(size):
    puzzle = list(range(size ** 2))
    random.shuffle(puzzle)
    return np.array(puzzle).reshape((size, size))

def is_solvable(puzzle):
    inversions = 0
    puzzle_flat = puzzle.flatten()
    for i in range(puzzle.size - 1):
        for j in range(i + 1, puzzle.size):
            if puzzle_flat[i] > puzzle_flat[j] and puzzle_flat[i] != 0 and puzzle_flat[j] != 0:
                inversions += 1
    return inversions % 2 == 0

def get_neighbors(state):
    # Get neighboring states for a given state
    neighbors = []
    blank_position = state.get_blank_position()
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, right, left
    for move in moves:
        new_position = blank_position + move
        if 0 <= new_position[0] < state.size and 0 <= new_position[1] < state.size:
            new_puzzle = state.puzzle.copy()
            new_puzzle[blank_position[0], blank_position[1]] = state.puzzle[new_position[0], new_position[1]]
            new_puzzle[new_position[0], new_position[1]] = 0
            neighbors.append(PuzzleState(new_puzzle, parent=state, move=move, cost=state.cost + 1))
    return neighbors

def best_first_search(initial_state):
    heap = [initial_state]
    visited = set()
    start_time = time.time()

    while heap:
        current_state = heapq.heappop(heap)
        if np.array_equal(current_state.puzzle, np.arange(current_state.size ** 2).reshape((current_state.size, current_state.size))):
            end_time = time.time()
            return current_state, end_time - start_time

        visited.add(tuple(current_state.puzzle.flatten()))
        neighbors = get_neighbors(current_state)
        for neighbor in neighbors:
            if tuple(neighbor.puzzle.flatten()) not in visited:
                heapq.heappush(heap, neighbor)

    return None, None

class PygameGUI:
    def __init__(self, size, puzzle_states, heuristic_name, execution_time, best_heuristic=None):
        # Initialize the Pygame GUI
        pygame.init()
        self.size = size
        self.puzzle_states = puzzle_states
        self.heuristic_name = heuristic_name
        self.execution_time = execution_time
        self.best_heuristic = best_heuristic

        # Set up the text screen
        self.text_screen = pygame.Surface((500, 100))
        self.text_screen.fill((255, 255, 255))
        self.text_font = pygame.font.Font(None, 24)

        # Set up the puzzle screen
        self.puzzle_screen = pygame.Surface((500, 500))

        # Set up the main screen
        self.screen = pygame.display.set_mode((500, 600))
        pygame.display.set_caption('Puzzle Solver')
        self.clock = pygame.time.Clock()

        self.current_state_index = 0

    def draw_text_screen(self, move_number):
        # Draw the text screen with information
        self.text_screen.fill((120, 120, 120))

        # Display execution time above the puzzle
        time_text = self.text_font.render(f"Execution Time: {self.execution_time:.2f} seconds", True, (0, 0, 0))
        self.text_screen.blit(time_text, (10, 10))

        # Display best heuristic information at the bottom
        best_heuristic_text = self.text_font.render(f"Best Heuristic: {self.best_heuristic} ", True, (0, 0, 0))
        self.text_screen.blit(best_heuristic_text, (10, 70))

        # Display move number above the puzzle
        move_text = self.text_font.render(f"Move: {move_number}", True, (0, 0, 0))
        self.text_screen.blit(move_text, (10, 30))

        # Display heuristic function name above the puzzle
        heuristic_text = self.text_font.render(f"Heuristic: {self.heuristic_name}", True, (0, 0, 0))
        self.text_screen.blit(heuristic_text, (10, 50))

        self.screen.blit(self.text_screen, (0, 0))

    def draw_puzzle_screen(self, state):
        # Draw the puzzle screen with the current state
        self.puzzle_screen.fill((44, 44, 44))
        cell_size = 500 // self.size

        for i in range(self.size):
            for j in range(self.size):
                pygame.draw.rect(self.puzzle_screen, (250, 250, 250), (j * cell_size, i * cell_size, cell_size, cell_size), 2)
                
                # Check if the cell is empty (value is 0)
                if state[i, j] != 0:
                    font_size = 36  # Change this to your desired font size
                    text_font = pygame.font.Font(None, font_size)
                    text = text_font.render(str(state[i, j]), True, (239, 20, 50))
                    text_rect = text.get_rect(center=(j * cell_size + cell_size // 2, i * cell_size + cell_size // 2))
                    self.puzzle_screen.blit(text, text_rect)
                else:
                    # Fill the empty cell with white color
                    pygame.draw.rect(self.puzzle_screen, (255, 255, 255), (j * cell_size, i * cell_size, cell_size, cell_size))

                self.screen.blit(self.puzzle_screen, (0, 100))
                
        self.screen.blit(self.puzzle_screen, (0, 100))

    def run(self):
        # Run the Pygame GUI
        move_number = 1  # Initialize move number
        for state in self.puzzle_states:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
            self.draw_text_screen(move_number)
            self.draw_puzzle_screen(state)
            pygame.display.flip()
            self.clock.tick(2)
            move_number += 1  # Increment move number

        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return

def get_size_input():
    # Get puzzle size input from the user using Pygame
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)
    input_box = pygame.Rect(100, 100, 140, 32)
    color_inactive = pygame.Color('lightskyblue3')
    color_active = pygame.Color('red')
    color = color_inactive
    text = ''
    active = False
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                if input_box.collidepoint(event.pos):
                    active = not active
                else:
                    active = False
                color = color_active if active else color_inactive
            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        done = True
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        text += event.unicode

        screen.fill((30, 30, 30))
        prompt_text = font.render("Enter the size of the puzzle:", True, (255, 255, 255))
        screen.blit(prompt_text, (50, 50))
        txt_surface = font.render(text, True, color)
        width = max(200, txt_surface.get_width() + 10)
        input_box.w = width
        screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
        pygame.draw.rect(screen, color, input_box, 2)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    return int(text) if text.isdigit() else None

def main():
    # Main function to run the puzzle solver
    puzzle_size = get_size_input()

    if puzzle_size is None or puzzle_size <= 2:
        print("Invalid puzzle size. Exiting...")
        return
    initial_puzzle = generate_random_puzzle(puzzle_size)
    while not is_solvable(initial_puzzle):
        initial_puzzle = generate_random_puzzle(puzzle_size)

    initial_state = PuzzleState(initial_puzzle)

    heuristics = ["Manhattan Distance", "Misplaced Tiles", "Linear Conflict", "Max Swap"]
    best_heuristic = None
    best_execution_time = np.double('inf')

    for heuristic in heuristics:
        initial_state.calculate_heuristic = getattr(initial_state, heuristic.lower().replace(" ", "_"))
        solution, execution_time = best_first_search(initial_state)

        print(f"\nHeuristic: {heuristic}")
        print("Initial Puzzle:")
        for row in initial_state.puzzle:
            print(' '.join(map(str, row)))
        print("\nSolution:")
        if solution:
            path = []
            while solution:
                path.append(solution.puzzle)
                solution = solution.parent
            path.reverse()

            print("Execution Time:", execution_time, "seconds")

            # Update best heuristic if the current one is better
            if execution_time < best_execution_time:
                best_execution_time = execution_time
                best_heuristic = heuristic

            pygame_gui = PygameGUI(puzzle_size, path, heuristic, execution_time, best_heuristic)
            pygame_gui.run()

        else:
            print("No solution found.")

    # Display summary with the best heuristic found
    print(f"\nBest Heuristic: {best_heuristic} (Execution Time: {best_execution_time:.2f} seconds)")

if __name__ == "__main__":
    main()
