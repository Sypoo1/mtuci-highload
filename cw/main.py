import math
import numpy as np
import pandas as pd
import heapq
import matplotlib.pyplot as plt
from collections import deque

np.random.seed(42)


# ─────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────

def linear_time(t_min, t_max):
    """Линейный (равномерный) закон распределения."""
    return np.random.uniform(t_min, t_max)


def exponential_time(rate):
    """Экспоненциальный закон распределения."""
    return np.random.exponential(1.0 / rate)


# ─────────────────────────────────────────────
# Имитационная модель СМО с буфером
# ─────────────────────────────────────────────

def simulate_system(
    arrival_type="linear",
    service_type="linear",
    T=3600,
    n_servers=3,
    buffer_size=3,
    tz_min=1/2, tz_max=5/6,
    ts_min=1.0, ts_max=5.0,
    lmbda=1.5,
    mu=0.5
):
    """
    Имитационная модель СМО с n_servers серверами и буфером buffer_size.

    Состояния системы (0..n_servers+buffer_size):
      0           — система пуста
      1..n_servers — заняты 1..n_servers серверов, буфер пуст
      n_servers+1..n_servers+buffer_size — все серверы заняты, в буфере 1..buffer_size программ

    Возвращает словарь со всеми характеристиками ВС.
    """
    max_in_system = n_servers + buffer_size   # максимум программ в системе
    n_states = max_in_system + 1              # состояния 0..6

    current_time = 0.0
    last_event_time = 0.0
    busy_servers = 0
    buffer_count = 0

    # Очередь событий: (время, тип, данные)
    events = []

    total_arrivals = 0
    total_served = 0
    total_rejected = 0

    # Время пребывания в каждом состоянии
    state_time = np.zeros(n_states)

    # Для расчёта K и Nбуф
    busy_servers_sum = 0.0
    buffer_sum = 0.0

    # Для расчёта Tбуф: суммарное время ожидания в буфере
    total_wait_in_buffer = 0.0
    total_passed_through_buffer = 0

    # Для расчёта Tпрог: суммарное время пребывания в системе
    total_time_in_system = 0.0

    # Буфер хранит время прихода каждой программы в буфер
    buffer_queue = deque()

    def gen_arrival():
        if arrival_type == "linear":
            return linear_time(tz_min, tz_max)
        else:
            return exponential_time(lmbda)

    def gen_service():
        if service_type == "linear":
            return linear_time(ts_min, ts_max)
        else:
            return exponential_time(mu)

    # Первый приход
    heapq.heappush(events, (gen_arrival(), "arrival", None))

    while events:
        current_time, event_type, event_data = heapq.heappop(events)

        if current_time > T:
            break

        # Обновляем статистику за прошедший интервал [last_event_time, current_time]
        dt = current_time - last_event_time
        state = busy_servers + buffer_count
        state_time[state] += dt
        busy_servers_sum += busy_servers * dt
        buffer_sum += buffer_count * dt
        last_event_time = current_time

        if event_type == "arrival":
            total_arrivals += 1
            # Планируем следующий приход
            heapq.heappush(events, (current_time + gen_arrival(), "arrival", None))

            if busy_servers < n_servers:
                # Есть свободный сервер — сразу обслуживаем
                busy_servers += 1
                svc = gen_service()
                # event_data = (время_прихода_в_систему, время_начала_обслуживания)
                heapq.heappush(events, (current_time + svc, "departure", (current_time, current_time)))
            elif buffer_count < buffer_size:
                # Серверы заняты, есть место в буфере
                buffer_count += 1
                buffer_queue.append(current_time)  # время прихода в буфер
            else:
                # Буфер заполнен — отказ
                total_rejected += 1

        elif event_type == "departure":
            arrival_in_system, service_start = event_data
            total_served += 1

            # Время пребывания в системе = текущее время - время прихода в систему
            total_time_in_system += current_time - arrival_in_system

            if buffer_count > 0:
                # Берём программу из буфера на освободившийся сервер
                buffer_count -= 1
                buf_arrival = buffer_queue.popleft()

                # Время ожидания в буфере = текущее время - время прихода в буфер
                wait = current_time - buf_arrival
                total_wait_in_buffer += wait
                total_passed_through_buffer += 1

                svc = gen_service()
                # Программа пришла в систему в момент buf_arrival, начала обслуживаться сейчас
                heapq.heappush(events, (current_time + svc, "departure", (buf_arrival, current_time)))
            else:
                busy_servers -= 1

    # ── Вычисление характеристик ──────────────────────────────────────────────

    total_sim_time = np.sum(state_time)
    if total_sim_time == 0:
        total_sim_time = T

    # Вероятности состояний
    P = state_time / total_sim_time

    # Относительная пропускная способность
    Q = total_served / total_arrivals if total_arrivals > 0 else 0.0

    # Абсолютная пропускная способность
    A = total_served / T

    # Вероятность отказа
    P_rej = total_rejected / total_arrivals if total_arrivals > 0 else 0.0

    # Среднее число занятых серверов
    K = busy_servers_sum / total_sim_time

    # Среднее число программ в буфере
    N_buf = buffer_sum / total_sim_time

    # Среднее число программ в системе (через вероятности состояний)
    N_prog = sum(i * P[i] for i in range(n_states))

    # Среднее время нахождения программы в системе (формула Литтла: N_prog = A * T_prog)
    T_prog = N_prog / A if A > 0 else 0.0

    # Среднее время ожидания в буфере
    T_buf = total_wait_in_buffer / total_passed_through_buffer if total_passed_through_buffer > 0 else 0.0

    return {
        "P": P,
        "P0": P[0], "P1": P[1], "P2": P[2], "P3": P[3],
        "P4": P[4], "P5": P[5], "P6": P[6],
        "Q": Q, "A": A, "P_rej": P_rej, "K": K,
        "N_prog": N_prog, "T_prog": T_prog,
        "N_buf": N_buf, "T_buf": T_buf,
        "total_arrivals": total_arrivals,
        "total_served": total_served,
        "total_rejected": total_rejected,
    }


def run_experiments(n_runs=10, **kwargs):
    """Запускает n_runs симуляций и возвращает усреднённые результаты."""
    results = [simulate_system(**kwargs) for _ in range(n_runs)]
    avg = {}
    for key in results[0]:
        if key == "P":
            avg[key] = np.mean([r[key] for r in results], axis=0)
        else:
            avg[key] = np.mean([r[key] for r in results])
    return avg, results


# ─────────────────────────────────────────────
# Теоретическая модель M/M/c/K
# ─────────────────────────────────────────────

def theoretical_MMcK(lmbda, mu, c=3, K=6):
    """
    Теоретическая модель M/M/c/K.
    c — число серверов, K — максимальное число программ в системе.
    """
    rho = lmbda / mu  # суммарная нагрузка

    # Ненормированные вероятности
    unnorm = []
    for n in range(K + 1):
        if n <= c:
            unnorm.append(rho**n / math.factorial(n))
        else:
            unnorm.append(rho**n / (math.factorial(c) * c**(n - c)))

    denom = sum(unnorm)
    P = [u / denom for u in unnorm]

    P_rej = P[K]
    Q = 1 - P_rej
    A = lmbda * Q
    K_avg = sum(min(n, c) * P[n] for n in range(K + 1))
    N_buf = sum((n - c) * P[n] for n in range(c + 1, K + 1))
    N_prog = sum(n * P[n] for n in range(K + 1))
    T_prog = N_prog / A if A > 0 else 0.0
    T_buf = N_buf / A if A > 0 else 0.0

    return {
        "P": P,
        "P0": P[0], "P1": P[1], "P2": P[2], "P3": P[3],
        "P4": P[4], "P5": P[5], "P6": P[6],
        "Q": Q, "A": A, "P_rej": P_rej, "K": K_avg,
        "N_prog": N_prog, "T_prog": T_prog,
        "N_buf": N_buf, "T_buf": T_buf,
    }


# ─────────────────────────────────────────────
# Запуск симуляций
# ─────────────────────────────────────────────

np.random.seed(42)

print("=== Запуск: линейный закон ===")
linear_res, _ = run_experiments(
    n_runs=10,
    arrival_type="linear", service_type="linear",
    T=3600, n_servers=3, buffer_size=3,
    tz_min=1/2, tz_max=5/6,
    ts_min=1.0, ts_max=5.0
)

print("=== Запуск: экспоненциальный закон ===")
exp_res, _ = run_experiments(
    n_runs=10,
    arrival_type="exponential", service_type="exponential",
    T=3600, n_servers=3, buffer_size=3,
    lmbda=1.5, mu=0.5
)

print("=== Теоретическая модель M/M/3/6 ===")
theory = theoretical_MMcK(lmbda=1.5, mu=0.5, c=3, K=6)

# ─────────────────────────────────────────────
# Таблицы результатов
# ─────────────────────────────────────────────

chars = [
    ("P0",     "P0 — ВС не загружена"),
    ("P1",     "P1 — загружен 1 сервер"),
    ("P2",     "P2 — загружены 2 сервера"),
    ("P3",     "P3 — загружены 3 сервера"),
    ("P4",     "P4 — в буфере 1 программа"),
    ("P5",     "P5 — в буфере 2 программы"),
    ("P6",     "P6 — в буфере 3 программы"),
    ("Q",      "Q — относит. пропускная способность"),
    ("A",      "A — абсолют. пропускная способность (1/сек)"),
    ("P_rej",  "Pотк — вероятность отказа"),
    ("K",      "K — среднее число занятых серверов"),
    ("N_prog", "Nпрог — среднее число программ в ВС"),
    ("T_prog", "Tпрог — среднее время в ВС (сек)"),
    ("N_buf",  "Nбуф — среднее число программ в буфере"),
    ("T_buf",  "Tбуф — среднее время ожидания в буфере (сек)"),
]

pd.set_option("display.float_format", "{:.6f}".format)
pd.set_option("display.max_colwidth", 50)

df1 = pd.DataFrame({
    "Характеристика": [c[1] for c in chars],
    "Линейный закон": [linear_res[c[0]] for c in chars],
})

df2 = pd.DataFrame({
    "Характеристика": [c[1] for c in chars],
    "Имитация (эксп)": [exp_res[c[0]] for c in chars],
    "Теория M/M/3/6":  [theory[c[0]]   for c in chars],
})

df3 = pd.DataFrame({
    "Характеристика": [c[1] for c in chars],
    "Линейный закон":          [linear_res[c[0]] for c in chars],
    "Экспоненциальный закон":  [exp_res[c[0]]    for c in chars],
})

print("\n── Таблица 1: Линейный закон ──────────────────────────────────────────")
print(df1.to_string(index=False))

print("\n── Таблица 2: Экспоненциальный — имитация vs теория ───────────────────")
print(df2.to_string(index=False))

print("\n── Таблица 3: Линейный vs Экспоненциальный ────────────────────────────")
print(df3.to_string(index=False))

# ─────────────────────────────────────────────
# Графики
# ─────────────────────────────────────────────

states = np.arange(7)
labels = [f"P{i}" for i in range(7)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(states, exp_res["P"],  marker='o', label="Имитация (эксп)")
ax.plot(states, theory["P"],   marker='s', linestyle='--', label="Теория M/M/3/6")
ax.set_xticks(states)
ax.set_xticklabels(labels)
ax.set_xlabel("Состояние системы")
ax.set_ylabel("Вероятность")
ax.set_title("Имитация (эксп) vs Теория M/M/3/6")
ax.legend()
ax.grid(True)

ax = axes[1]
ax.plot(states, linear_res["P"], marker='o', label="Линейный закон")
ax.plot(states, exp_res["P"],    marker='s', linestyle='--', label="Экспоненциальный закон")
ax.set_xticks(states)
ax.set_xticklabels(labels)
ax.set_xlabel("Состояние системы")
ax.set_ylabel("Вероятность")
ax.set_title("Линейный vs Экспоненциальный")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig("cw/results.png", dpi=150)
plt.show()
print("\nГрафики сохранены в cw/results.png")
