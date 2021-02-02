# Read Me
Code available for the methods described in Random Graphs with Prescribed K-Core Sequences:  A New Null Model for Network Analysis.

The generator will take a list of core values and generate a graph that has nodes that correspond to those values. Note that not all core value combinations are viable.

The rejection sampling model takes a graph and creates a random graph with the same core values using the markov chain method described in the accompanying paper. Any graph input needs to be undirected and a SparseMatrixCSC. There are a number of adjustable parameters and print outs.
