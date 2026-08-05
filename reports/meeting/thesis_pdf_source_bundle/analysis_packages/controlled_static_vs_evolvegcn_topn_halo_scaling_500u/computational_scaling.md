# Computational scaling

At k=8, Top100, Top200, and Top500 imply 800, 1,600, and 4,000 directed
neighbor selections per snapshot before reciprocal symmetrization. These
exclude self-loops and padded nodes; reciprocal relations may duplicate after
symmetrization, and model layers add self-loops. Dense adjacency capacity is
10,000, 40,000, and 250,000 entries per snapshot.

Wall time, GPU/CPU peak memory, and prediction time were not recorded. Top500
therefore cannot be recommended on a fully quantified efficiency basis.
Evolve gains are meaningful relative to observed seed variability; Static
gains are negligible relative to graph-size growth.
