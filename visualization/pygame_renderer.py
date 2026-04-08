import pygame
import requests
import json
import time
import sys
import os

# Initialize Pygame
pygame.init()

# --- Configurations ---
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NexFlow AI - PyGame Grid Visualizer")

# Colors
BG_COLOR = (15, 23, 42)        # Slate 900
TEXT_COLOR = (241, 245, 249)   # Slate 100
ACCENT = (52, 211, 153)        # Emerald 400
WARNING = (248, 113, 113)      # Red 400
NODE_BG = (30, 41, 59)         # Slate 800

font_title = pygame.font.SysFont("consolas", 32, bold=True)
font_main = pygame.font.SysFont("consolas", 20)
font_small = pygame.font.SysFont("consolas", 16)

# The mock state matching your environment
mock_state = {
    "city": "bengaluru",
    "intersections": [
        {"junction_id": "silk_board", "queue_lengths": [50, 10, 5, 2], "emergency_present": True, "current_phase": 0},
        {"junction_id": "hsr_layout", "queue_lengths": [1, 0, 2, 0], "emergency_present": False, "current_phase": 2}
    ]
}

API_URL = "http://127.0.0.1:8000/predict"
ai_decisions = []
is_fetching = False
status_message = "Press SPACE to trigger GNN Optimization"

def fetch_ai_decisions():
    global ai_decisions, status_message, is_fetching
    is_fetching = True
    status_message = "📡 Pinging PyTorch API..."
    draw_screen() # Force an update to show the loading message
    
    try:
        response = requests.post(API_URL, json=mock_state, timeout=5)
        if response.status_code == 200:
            ai_decisions = response.json().get("actions", [])
            status_message = "✅ GNN Inference Complete. Lights updated."
        else:
            status_message = f"❌ API Error: {response.status_code}"
    except Exception as e:
        status_message = "❌ Connection Failed. Is uvicorn running on Port 8000?"
    
    is_fetching = False

def draw_screen():
    screen.fill(BG_COLOR)
    
    # Header
    title = font_title.render("🚦 NexFlow AI System Matrix", True, ACCENT)
    screen.blit(title, (30, 30))
    
    status = font_main.render(status_message, True, TEXT_COLOR if not is_fetching else ACCENT)
    screen.blit(status, (30, 80))

    # Draw Intersections
    start_x = 50
    start_y = 150
    box_w = 400
    box_h = 250
    padding = 50

    for i, intersection in enumerate(mock_state["intersections"]):
        x = start_x + (i * (box_w + padding))
        y = start_y
        
        # Base Card
        pygame.draw.rect(screen, NODE_BG, (x, y, box_w, box_h), border_radius=15)
        pygame.draw.rect(screen, (71, 85, 105), (x, y, box_w, box_h), width=2, border_radius=15)

        # Junction Name
        name_text = font_title.render(intersection["junction_id"].replace('_', ' ').upper(), True, TEXT_COLOR)
        screen.blit(name_text, (x + 20, y + 20))

        # Emergency Status
        if intersection["emergency_present"]:
            em_text = font_main.render("🚨 EMERGENCY DETECTED", True, WARNING)
            screen.blit(em_text, (x + 20, y + 60))
        else:
            em_text = font_main.render("✅ Standard Flow", True, ACCENT)
            screen.blit(em_text, (x + 20, y + 60))

        # Queues
        q_text = font_small.render(f"Queues: {intersection['queue_lengths']}", True, (148, 163, 184))
        screen.blit(q_text, (x + 20, y + 100))

        # AI Decision Overlay
        decision = next((d for d in ai_decisions if d["junction_id"] == intersection["junction_id"]), None)
        if decision:
            pygame.draw.rect(screen, (15, 23, 42), (x + 20, y + 140, box_w - 40, 90), border_radius=10)
            
            p_text = font_main.render(f"Target Phase : P-{decision['next_phase']}", True, ACCENT)
            d_text = font_main.render(f"Hold Duration: {decision['duration']} seconds", True, WARNING if intersection["emergency_present"] else (96, 165, 250))
            
            screen.blit(p_text, (x + 40, y + 160))
            screen.blit(d_text, (x + 40, y + 190))
        else:
            wait_text = font_main.render("Awaiting Matrix Instructions...", True, (100, 116, 139))
            screen.blit(wait_text, (x + 20, y + 180))

    # Controls footer
    controls = font_small.render("[SPACE] Run Inference  |  [ESC] Exit Simulator", True, (100, 116, 139))
    screen.blit(controls, (WIDTH // 2 - controls.get_width() // 2, HEIGHT - 40))

    pygame.display.flip()

# --- Main Loop ---
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE and not is_fetching:
                fetch_ai_decisions()

    draw_screen()
    clock.tick(30)

pygame.quit()
sys.exit()