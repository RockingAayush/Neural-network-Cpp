import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv("predictions.csv")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df["x"], df["sin_x"], label="sin(x)", linewidth=2)
plt.plot(df["x"], df["predicted"], label="Predicted", linestyle="--", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Neural Network Approximation of sin(x)")
plt.legend()
plt.grid(True)
plt.show()
