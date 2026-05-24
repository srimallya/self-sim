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
        self.font = pygame.font.SysFont("couriernew", 13)

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

        self._draw_overlay()
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

    def _draw_overlay(self):
        overlay_stats = getattr(self.env, "overlay_stats", None)
        if not overlay_stats:
            return

        box_height = 18 + 18 * len(overlay_stats)
        surface = pygame.Surface((430, box_height), pygame.SRCALPHA)
        surface.fill((0, 0, 0, 150))
        self.screen.blit(surface, (6, 6))

        step = getattr(self.env, "step_count", 0)
        mode = overlay_stats[0].get("mode") if overlay_stats else None
        title_text = f"{mode} step {step}" if mode else f"step {step}"
        title = self.font.render(title_text, True, (220, 220, 220))
        self.screen.blit(title, (12, 10))

        for idx, stats in enumerate(overlay_stats):
            if stats.get("mode") == "DUAL SYSTEM":
                self._draw_dual_system_overlay_row(idx, stats)
                continue
            if stats.get("mode") == "ZERO SUM":
                self._draw_zero_sum_overlay_row(idx, stats)
                continue
            leakage = stats.get("leakage", 0.0)
            compute_budget = stats.get("compute_budget", 0.0)
            action = stats.get("selected_action", -1)
            fallback_used = "Y" if stats.get("fallback_used") else "N"
            loss = stats.get("loss")
            loss_text = "--" if loss is None else f"{loss:.2f}"
            energy = stats.get("energy", 0.0)
            evo_text = ""
            if "lineage_id" in stats:
                last_winner = stats.get("last_winner")
                last_loser = stats.get("last_loser")
                last_text = "--" if last_winner is None else f"{last_winner}>{last_loser}"
                evo_text = (
                    f" G:{int(stats.get('generation', 0))} "
                    f"Lin:{int(stats.get('lineage_id', idx))} "
                    f"S:{float(stats.get('window_score', 0.0)):.1f} W/L:{last_text}"
                )
            line = (
                f"A{idx} E:{energy:6.1f} L:{leakage:0.03f} "
                f"B:{compute_budget:0.02f} Act:{action:02d} FB:{fallback_used} Loss:{loss_text}{evo_text}"
            )
            text = self.font.render(line, True, pygame.Color(stats.get("color", "white")))
            self.screen.blit(text, (12, 28 + 18 * idx))

    def _draw_dual_system_overlay_row(self, idx, stats):
        goal_probs = np.asarray(stats.get("slow_goal_probs", []), dtype=np.float32).reshape(-1)
        if goal_probs.size:
            top_goal = int(np.argmax(goal_probs[:4]))
            goal_text = f"G{top_goal}:{float(goal_probs[top_goal]):.2f}"
        else:
            goal_text = "--"
        line = (
            f"A{idx} E{stats.get('energy', 0.0):.0f} F{int(stats.get('food_eaten', 0))} "
            f"C{stats.get('collisions_per_100_steps', 0.0):.0f} FB{stats.get('fallback_rate', 0.0):.2f} "
            f"Err{stats.get('world_error', 0.0):.2f} SE{stats.get('slow_energy_scale', 1.0):.2f} "
            f"T{int(stats.get('slow_countdown', 0)):02d} {goal_text} A{int(stats.get('selected_action', -1)):02d}"
        )
        text = self.font.render(line, True, pygame.Color(stats.get("color", "white")))
        self.screen.blit(text, (12, 28 + 18 * idx))

    def _draw_zero_sum_overlay_row(self, idx, stats):
        line = (
            f"A{idx} E{stats.get('energy', 0.0):.0f} F{int(stats.get('food_eaten', 0))} "
            f"Gap{stats.get('energy_gap', 0.0):+.0f} Cap{int(stats.get('captures', 0))} "
            f"St{int(stats.get('steals', 0))} FB{stats.get('fallback_rate', 0.0):.2f} "
            f"Err{stats.get('world_error', 0.0):.2f} SE{stats.get('slow_energy_scale', 1.0):.2f} "
            f"T{int(stats.get('slow_countdown', 0)):02d} LC{int(stats.get('lead_changes', 0))} "
            f"Rsum{stats.get('reward_sum', 0.0):+.2f}"
        )
        text = self.font.render(line, True, pygame.Color(stats.get("color", "white")))
        self.screen.blit(text, (12, 28 + 18 * idx))
