import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import messagebox

# --- Genres and Base Transition Matrix ---
genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi']
n = len(genres)

P_base = np.array([
    [0.5, 0.1, 0.1, 0.05, 0.05, 0.2],
    [0.1, 0.5, 0.2, 0.05, 0.1, 0.05],
    [0.1, 0.2, 0.5, 0.05, 0.1, 0.05],
    [0.2, 0.05, 0.1, 0.5, 0.1, 0.05],
    [0.05, 0.1, 0.3, 0.05, 0.4, 0.1],
    [0.3, 0.05, 0.1, 0.05, 0.05, 0.45],
])

# --- Helper Functions ---
def normalize_preference_vector(vec):
    total = np.sum(vec)
    if total == 0:
        raise ValueError("Preference vector has all zeros.")
    return vec / total

def apply_duplicity_factor(P, watch_history, fatigue_decay=0.6, min_prob=0.05):
    counts = np.zeros(len(genres))
    for g in watch_history:
        counts[genres.index(g)] += 1
    freq = counts / counts.sum() if counts.sum() > 0 else counts
    penalties = np.exp(-fatigue_decay * freq)
    penalties = np.maximum(penalties, min_prob)
    P_mod = P * penalties.reshape(1, -1)
    P_mod = P_mod / P_mod.sum(axis=1, keepdims=True)
    return P_mod

def personalize_matrix(P_base, user_vec, watch_history, alpha, exploration_prob):
    P_dup = apply_duplicity_factor(P_base, watch_history)
    P_user = alpha * P_dup + (1 - alpha) * user_vec.reshape(1, -1)
    uniform = np.ones((n, n)) / n
    P_user = (1 - exploration_prob) * P_user + exploration_prob * uniform
    P_user = P_user / P_user.sum(axis=1, keepdims=True)
    return P_user

def compute_stationary(P, tol=1e-10, max_iter=10000):
    pi = np.ones(P.shape[0]) / P.shape[0]
    for _ in range(max_iter):
        new_pi = pi @ P
        if np.linalg.norm(new_pi - pi, 1) < tol:
            return new_pi
        pi = new_pi
    return pi

def check_reversibility(P, pi, tol=1e-6):
    for i in range(len(P)):
        for j in range(len(P)):
            if abs(pi[i]*P[i,j] - pi[j]*P[j,i]) > tol:
                return False
    return True

def check_absorbing_states(P):
    return [genres[i] for i in range(len(P)) if np.all(P[i] == np.eye(len(P))[i])] or ["None"]

def predict_next_genre(P, current_index):
    return np.random.choice(n, p=P[current_index])

def simulate_predictions(P, start_index, steps=10):
    result = []
    state = start_index
    for _ in range(steps):
        state = predict_next_genre(P, state)
        result.append(genres[state])
    return result

def hitting_time_all(P, start_state, max_steps=1000, num_simulations=50):
    times = {}
    for target in range(n):
        if start_state == target:
            times[genres[target]] = 0
            continue
        hits = []
        for _ in range(num_simulations):
            state = start_state
            steps = 0
            while state != target and steps < max_steps:
                state = np.random.choice(n, p=P[state])
                steps += 1
            hits.append(steps)
        times[genres[target]] = np.mean(hits)
    return times

# --- GUI App ---
def run_app():
    root = tk.Tk()
    root.title("Markov Chain Movie Genre Recommender")

    tk.Label(root, text="Enter your genre interests (e.g. 1 to 10):").pack()
    entries = []
    for g in genres:
        frame = tk.Frame(root)
        frame.pack()
        tk.Label(frame, text=g).pack(side=tk.LEFT)
        entry = tk.Entry(frame, width=5)
        entry.pack(side=tk.RIGHT)
        entries.append(entry)

    # Alpha
    alpha_frame = tk.Frame(root)
    alpha_frame.pack()
    tk.Label(alpha_frame, text="Teleportation factor (alpha):").pack(side=tk.LEFT)
    alpha_entry = tk.Entry(alpha_frame, width=5)
    alpha_entry.insert(0, "0.7")
    alpha_entry.pack(side=tk.RIGHT)

    # Exploration
    explore_frame = tk.Frame(root)
    explore_frame.pack()
    tk.Label(explore_frame, text="Exploration probability:").pack(side=tk.LEFT)
    explore_entry = tk.Entry(explore_frame, width=5)
    explore_entry.insert(0, "0.05")
    explore_entry.pack(side=tk.RIGHT)

    # Genre selection
    tk.Label(root, text="\nSelect your current genre:").pack()
    genre_var = tk.StringVar()
    genre_var.set(genres[0])
    tk.OptionMenu(root, genre_var, *genres).pack()

    def submit():
        try:
            user_vec = np.array([float(e.get()) for e in entries])
            user_vec = normalize_preference_vector(user_vec)
            alpha = float(alpha_entry.get())
            exploration_prob = float(explore_entry.get())
        except:
            messagebox.showerror("Error", "Please enter valid numerical values.")
            return

        start_genre = genre_var.get()
        start_index = genres.index(start_genre)
        watch_history = [start_genre]

        P_user = personalize_matrix(P_base, user_vec, watch_history, alpha, exploration_prob)

        # Show Transition Matrix
        plt.figure(figsize=(7, 6))
        sns.heatmap(P_user, annot=True, fmt=".2f", xticklabels=genres, yticklabels=genres, cmap="YlGnBu")
        plt.title("Personalized Transition Matrix")
        plt.xlabel("To Genre")
        plt.ylabel("From Genre")
        plt.tight_layout()
        plt.show()

        # Predict next 5 genres
        predictions = simulate_predictions(P_user, start_index, steps=10)

        # Hitting time to all genres
        hit_times = hitting_time_all(P_user, start_index)
        hit_report = "\n".join([f"{k}: {v:.2f} steps" for k, v in hit_times.items()])

        # Absorbing and reversibility
        absorbing = check_absorbing_states(P_user)
        pi = compute_stationary(P_user)
        reversible = check_reversibility(P_user, pi)

        report = (
            f"Selected Genre: {start_genre}\n\n"
            f"Next 10 Predictions: {', '.join(predictions)}\n\n"
            f"Absorbing States: {', '.join(absorbing)}\n"
            f"Chain Reversibility: {'Yes' if reversible else 'No'}\n\n"
            f"Hitting Times from '{start_genre}':\n{hit_report}"
        )

        messagebox.showinfo("Markov Chain Results", report)

    tk.Button(root, text="Run Recommendation", command=submit).pack(pady=10)

    root.mainloop()

run_app()
