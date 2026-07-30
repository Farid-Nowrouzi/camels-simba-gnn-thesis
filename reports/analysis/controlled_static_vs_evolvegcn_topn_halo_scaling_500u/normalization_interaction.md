# Normalization interaction

Canonical features use min-max normalization independently per universe,
snapshot, and feature after Top-N selection. Changing Top-N therefore changes
both the retained nodes and the local minima/maxima. This is neither target
leakage nor train/test leakage; it is part of the established Top-N
intervention. Results do not isolate node count while holding all numerical
feature values fixed.
