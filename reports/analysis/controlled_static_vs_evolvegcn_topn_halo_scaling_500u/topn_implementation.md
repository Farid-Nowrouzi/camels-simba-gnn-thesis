# Top-N implementation

Invalid/nonpositive raw Mvir rows are removed. Halos are sorted by raw Mvir
descending and the first N are retained independently per universe and
snapshot. log10(Mvir), node features, and sample-local normalization follow
selection. Raw XYZ is separately retained for periodic minimum-image kNN.
Fewer-than-N snapshots are zero-padded after normalization; masks exclude
padded nodes, and zero-real-node snapshots are rejected.

Top100/Top200/Top500 membership is expected to be nested for identical inputs,
but selected IDs/rank hashes were not stored and equal-mass ties lack an
explicit stable secondary key. Graph topology is rebuilt at each N; larger
graphs are not induced supergraphs.
