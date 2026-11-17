import os
import time
import random

import numpy as np
from playwright.sync_api import sync_playwright

GAME_URL = "https://wayou.github.io/t-rex-runner/"

EPISODES = 300
ALPHA = 0.08
GAMMA = 0.92
EPSILON = 0.1
STEP_DELAY = 0.05
MAX_STEPS = 11600
MODEL_FILE = "q_table_v2.npy"
TRAIN_MODE = True
LOAD_MODEL = True
SAVE_MODEL = True

# Действия: 0 = Бежать (ничего не жать), 1 = Прыгать, 2 = Приседать (если бежит)
ACTIONS = [0, 1, 2]

# Словарь вида {state_tuple: [q_run, q_jump, q_duck]}
q_table = {}


def ensure_state(state):
    if state != -1 and state not in q_table:
        q_table[state] = [0.0, 0.0, 0.0]


def load_q_table():
    global q_table
    if LOAD_MODEL and os.path.exists(MODEL_FILE):
        loaded = np.load(MODEL_FILE, allow_pickle=True).item()
        if isinstance(loaded, dict):
            q_table = loaded
            print(f"Загружена сохранённая Q-таблица ({len(q_table)} состояний).")
            return
    q_table = {}
    print("Q-таблица создана с нуля.")


def save_q_table():
    if SAVE_MODEL:
        np.save(MODEL_FILE, q_table)
        print(f"Q-таблица сохранена в файл {MODEL_FILE} ({len(q_table)} состояний)")


def get_state(page):
    try:
        result = page.evaluate(
            """
            () => {
                const runner = window.Runner?.instance_;
                if (!runner) {
                    return { status: "NO_RUNNER" };
                }

                if (runner.crashed) {
                    return { status: "DEAD" };
                }

                if (!runner.playing) {
                    return { status: "NOT_PLAYING" };
                }

                const dino = runner.tRex;
                const obstacles = runner.horizon?.obstacles || [];

                const upcoming = obstacles.filter(obs => {
                    const width = obs.width || (obs.typeConfig?.width || 25) * (obs.size || 1);
                    return (obs.xPos + width) > dino.xPos;
                });

                const obs1 = upcoming[0] || null;
                const obs2 = upcoming[1] || null;

                let dist1 = 1000;
                let y1 = 100;
                let gap2 = 1000;
                let y2 = 100;

                if (obs1) {
                    const width1 = obs1.width || (obs1.typeConfig?.width || 25) * (obs1.size || 1);
                    dist1 = obs1.xPos - dino.xPos;
                    y1 = obs1.yPos;

                    if (obs2) {
                        gap2 = obs2.xPos - (obs1.xPos + width1);
                        y2 = obs2.yPos;
                    }
                }

                return {
                    status: "OK",
                    dist1: dist1,
                    y1: y1,
                    gap2: gap2,
                    y2: y2,
                    speed: runner.currentSpeed || 6,
                    isJumping: Boolean(dino.jumping),
                    isDucking: Boolean(dino.ducking)
                };
            }
            """
        )

        status = result["status"]
        if status != "OK":
            return -1, status

        speed = max(result["speed"], 6.0)
        speed_factor = speed / 6.0

        # Эффективная дистанция с учётом скорости игры
        eff_dist1 = result["dist1"] / speed_factor
        eff_gap2 = result["gap2"] / speed_factor

        # 1. Дистанция до ближайшего препятствия
        if eff_dist1 > 220:
            d1_bin = 0  # Далеко
        elif eff_dist1 > 125:
            d1_bin = 1  # Средне
        elif eff_dist1 > 60:
            d1_bin = 2  # Близко
        else:
            d1_bin = 3  # Вплотную

        # 2. Высота ближайшего препятствия (кактус vs птеродактиль)
        # yPos >= 85: земля / низкий полёт (надо прыгать)
        # 65 <= yPos < 85: средний птеродактиль (можно пригнуться)
        # yPos < 65: высокий птеродактиль (можно пробежать или пригнуться)
        if result["dist1"] >= 900:
            h1_bin = 0
        elif result["y1"] >= 85:
            h1_bin = 0
        elif result["y1"] >= 65:
            h1_bin = 1
        else:
            h1_bin = 2

        # 3. Расстояние от 1-го препятствия до 2-го (следующего)
        if eff_gap2 > 220:
            g2_bin = 0  # Следующее далеко или отсутствует
        elif eff_gap2 > 110:
            g2_bin = 1  # Средний промежуток
        else:
            g2_bin = 2  # Вплотную за первым

        # 4. Тип высоты 2-го препятствия (0 = земля/нет, 1 = воздух)
        if result["gap2"] >= 900 or result["y2"] >= 85:
            h2_bin = 0
        else:
            h2_bin = 1

        is_jumping = 1 if result["isJumping"] else 0

        state = (d1_bin, h1_bin, g2_bin, h2_bin, is_jumping)
        ensure_state(state)

        info = (
            f"d1={result['dist1']:.0f}, y1={result['y1']:.0f}, "
            f"gap2={result['gap2']:.0f}, spd={speed:.1f}, jump={is_jumping}"
        )
        return state, info

    except Exception as error:
        return -1, f"ERROR: {error}"


def choose_action(state):
    ensure_state(state)
    if TRAIN_MODE and random.random() < EPSILON:
        return random.choice(ACTIONS)

    return int(np.argmax(q_table[state]))


def update_q_table(state, action, reward, next_state):
    ensure_state(state)
    current_q = q_table[state][action]

    if next_state == -1:
        max_next_q = 0.0
    else:
        ensure_state(next_state)
        max_next_q = max(q_table[next_state])

    new_q = current_q + ALPHA * (
        reward + GAMMA * max_next_q - current_q
    )
    q_table[state][action] = new_q


def release_duck(page, is_duck_held):
    if is_duck_held:
        page.keyboard.up("ArrowDown")
    return False


def apply_action(page, action, state, is_duck_held):
    is_jumping = (state != -1 and state[4] == 1)

    if action == 2 and not is_jumping:
        # Приседаем, только если динозавр бежит по земле
        if not is_duck_held:
            page.keyboard.down("ArrowDown")
            is_duck_held = True
    else:
        # Если бежим или прыгаем — сначала отпускаем присед
        is_duck_held = release_duck(page, is_duck_held)
        if action == 1 and not is_jumping:
            page.keyboard.press("Space")

    return is_duck_held


def start_game(page):
    page.keyboard.up("ArrowDown")
    canvas = page.locator("canvas")
    canvas.click()
    time.sleep(0.3)
    page.keyboard.press("Space")
    time.sleep(1.0)
    state, info = get_state(page)
    return state, info


def print_q_table():
    print(f"\nИтоговая Q-таблица (всего состояний: {len(q_table)}):")
    print("Формат состояния: (dist1, height1, gap2, height2, is_jumping)")
    print("Действия: [0=Бежать, 1=Прыгать, 2=Приседать]")

    action_names = {
        0: "БЕЖАТЬ",
        1: "ПРЫГАТЬ",
        2: "ПРИСЕДАТЬ",
    }

    for state in sorted(q_table.keys()):
        values = q_table[state]
        best_action = int(np.argmax(values))
        print(
            f"Состояние {state}: "
            f"[{values[0]:.2f}, {values[1]:.2f}, {values[2]:.2f}] "
            f"-> {action_names[best_action]}"
        )


def main():
    print("Запускаем программу...")
    load_q_table()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chromium",
            headless=False,
            args=[
                "--disable-gpu",
                "--disable-extensions",
            ]
        )

        page = browser.new_page(
            viewport={
                "width": 800,
                "height": 600
            }
        )

        print("Открываем игру...")
        page.goto(GAME_URL, wait_until="networkidle")

        print("Ждём загрузку игры...")
        time.sleep(2.5)

        for episode in range(1, EPISODES + 1):
            print(f"\n--- Игра №{episode} ---")

            state, info = start_game(page)

            if state == -1:
                print(f"Игра не запустилась. Причина: {info}")
                print("Перезагружаем страницу...")
                page.reload(wait_until="networkidle")
                time.sleep(2.5)
                continue

            total_reward = 0.0
            is_duck_held = False

            try:
                for step in range(1, MAX_STEPS + 1):
                    action = choose_action(state)
                    is_duck_held = apply_action(page, action, state, is_duck_held)

                    time.sleep(STEP_DELAY)

                    next_state, info = get_state(page)

                    if next_state == -1:
                        reward = -100.0
                    elif action == 0:
                        reward = 1.0
                    else:
                        # Небольшой штраф за лишний прыжок/присед, чтобы не спамил зря
                        reward = 0.95

                    if TRAIN_MODE:
                        update_q_table(
                            state=state,
                            action=action,
                            reward=reward,
                            next_state=next_state
                        )

                    total_reward += reward
                    state = next_state

                    if next_state == -1:
                        print(
                            f"Игра окончена. "
                            f"Шагов: {step}. "
                            f"Награда: {total_reward:.1f}. "
                            f"Причина: {info}"
                        )
                        break
                else:
                    print(
                        f"Достигнут лимит шагов: {MAX_STEPS}. "
                        f"Награда: {total_reward:.1f}"
                    )
            finally:
                is_duck_held = release_duck(page, is_duck_held)

            if TRAIN_MODE and episode % 10 == 0:
                save_q_table()

            time.sleep(1.0)

        print("\nОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print_q_table()

        if TRAIN_MODE:
            save_q_table()

        input("\nНажми Enter, чтобы закрыть браузер...")
        browser.close()


if __name__ == "__main__":
    main()