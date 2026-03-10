import numpy as np
from code.ara_core import ARA

G = np.array([
    [8.5, 7.0, 9.0, 6.5],
    [7.5, 8.5, 6.5, 6.2],
    [6.0, 8.1, 5.5, 8.0]
])

I = np.array([
    [8.0, 7.5, 8.5, 6.0],
    [7.0, 8.0, 6.0, 6.5],
    [6.5, 7.5, 5.0, 8.2]
])

model = ARA(G, I)

ranking, scores = model.rank(alpha=0.618)

print("Scores:", scores)
print("Ranking:", ranking)
