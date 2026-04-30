import math

import numpy as np
import pygame

from nlri.envs.maze_reservoir_env import CELL_SIZE, GRID_SIZE, MAZE, SCREEN_SIZE


class PygameViewer:
    def __init__(self, env, show_energy_rays: bool = True, fps: int = 24):
        self.env = env
        self.show_energy_rays = show_energy_rays
        self.fps = fps
        self.closed = False
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        pygame.display.set_caption("self-sim NLRI")
        self.clock = pygame.time.Clock()

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True

        self.screen.fill((12, 12, 12))
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if MAZE[y, x] == 1:
                    pygame.draw.rect(self.screen, (32, 32, 32), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for (x, y), _energy in self.env.food_positions.items():
            pygame.draw.circle(
                self.screen,
                (128, 128, 128),
                (int(x * CELL_SIZE + CELL_SIZE / 2), int(y * CELL_SIZE + CELL_SIZE / 2)),
                3,
            )

        for agent in self.env.agents:
            if self.show_energy_rays:
                self._draw_energy_gradient(agent)
            start_pos = (
                int(agent.pos[0] * CELL_SIZE + CELL_SIZE / 2),
                int(agent.pos[1] * CELL_SIZE + CELL_SIZE / 2),
            )
            end_pos = (
                int(start_pos[0] + 10 * math.cos(math.radians(agent.angle))),
                int(start_pos[1] + 10 * math.sin(math.radians(agent.angle))),
            )
            self._draw_arrow(pygame.Color(agent.color), start_pos, end_pos)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if not self.closed:
            pygame.quit()
            self.closed = True

    def _draw_arrow(self, color, start, end):
        pygame.draw.line(self.screen, color, start, end, 2)
        rotation = math.degrees(math.atan2(start[1] - end[1], end[0] - start[0])) + 90
        pygame.draw.polygon(
            self.screen,
            color,
            (
                (end[0] + 5 * math.sin(math.radians(rotation)), end[1] + 5 * math.cos(math.radians(rotation))),
                (end[0] + 5 * math.sin(math.radians(rotation - 120)), end[1] + 5 * math.cos(math.radians(rotation - 120))),
                (end[0] + 5 * math.sin(math.radians(rotation + 120)), end[1] + 5 * math.cos(math.radians(rotation + 120))),
            ),
        )

    def _draw_energy_gradient(self, agent):
        center = (int(agent.pos[0] * CELL_SIZE + CELL_SIZE / 2), int(agent.pos[1] * CELL_SIZE + CELL_SIZE / 2))
        gradient = np.asarray(agent.raw_energy_gradient, dtype=np.float32)
        max_energy = float(np.max(gradient))
        if max_energy <= 0:
            return
        normalized = gradient / max_energy
        base_color = pygame.Color(agent.color)
        for angle in range(0, 360, 10):
            energy = normalized[angle]
            end_x = center[0] + int(50 * energy * math.cos(math.radians(angle)))
            end_y = center[1] + int(50 * energy * math.sin(math.radians(angle)))
            color = (
                max(0, min(255, int(base_color.r * energy))),
                max(0, min(255, int(base_color.g * energy))),
                max(0, min(255, int(base_color.b * energy))),
            )
            pygame.draw.line(self.screen, color, center, (end_x, end_y), 1)
