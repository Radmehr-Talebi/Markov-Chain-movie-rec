Markov Chain Movie Genre Recommender

This project implements a Markov chain–based movie genre recommendation system with a graphical user interface (GUI) built in Tkinter. It personalizes recommendations based on user preferences, watch history, and advanced probabilistic features like teleportation, exploration, reversibility checks, and hitting time analysis.

Features

Interactive GUI – Enter your genre preferences directly in the app.

Personalized transition matrix – Combines user preferences, watch history, and randomness for diversity.

Fatigue & duplicity modeling – Reduces the chance of recommending the same genre repeatedly.

Teleportation factor (α) – Controls the balance between preferences and Markov transitions.

Exploration probability – Adds randomness to avoid overfitting to a single genre.

Stationary distribution computation – Finds long-term viewing preferences.

Absorbing state detection – Identifies genres that trap the chain.

Reversibility check – Verifies if the Markov chain is reversible.

Hitting time analysis – Estimates how many steps it takes to reach each genre from the starting point.

Heatmap visualization – Displays the personalized transition matrix using Seaborn.

How It Works

User Input – Rate your interest in each genre (e.g., 1–10).

Set Parameters – Choose teleportation factor (α) and exploration probability.

Select Current Genre – Pick your current watching genre.

Run Recommendation –

Generates a personalized transition matrix.

Predicts next 10 genres.

Computes hitting times, reversibility, and absorbing states.

Displays results in a popup report.

Concepts Used

Markov Chains – Probabilistic modeling of transitions between genres.

Stationary Distribution – Long-term equilibrium viewing distribution.

Absorbing States – States that once entered, cannot be left.

Reversibility – Ensures detailed balance between transitions.

Hitting Times – Expected steps to reach a given state.

