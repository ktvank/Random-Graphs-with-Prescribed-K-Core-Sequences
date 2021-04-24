using StatsBase
using LinearAlgebra
using SparseArrays
using Random
using Combinatorics
using MatrixNetworks
using MAT
using GraphRecipes
using Plots
using CSV, DataFrames
using DelimitedFiles
using Arpack
using LightGraphs
using Distributions


struct SmallNullException <: Exception end
const SMALL_NULL = 1
const GOOD_RETURN = 0


##SET-UP AND USEFUL FUNCTIONS

#When we are combining the row, column, and value vectors, we want to make sure the sparse matrix is composed of only 0's and 1's
function comb(a::Int64, b::Int64)
    if a+b > 0
        return 1
    end
    return 0
end

#A function to check if the graph given has the same core-values as the sequence given
#Uses corenums function for faster calculations
#We may want to update this to an incremental algorithm later
function checkKCore(G::SparseMatrixCSC{Int64,Int64}, seq::Vector{Int64}, get_cores=false)
    core_seq = core_number(SimpleGraph(G))
    if (core_seq != seq) && get_cores
        println(string("""New Cores: """, core_seq))
        println(string("""Core Sequence: """, seq))
    end
    return (core_seq == seq)
end

#This returns a subgraph of a graph G, where only the nodes in the vector are present (usually used to get a subgraph of all
#nodes of a certain core-value or greater)
function getCoresSubgraph(G::SparseMatrixCSC{Int64,Int64}, nodes::Vector{Int64})
    return G[nodes[1]:nodes[length(nodes)], nodes[1]:nodes[length(nodes)]]
end

#Checks if a sequence of core-values is realizable as a graph, there is one rule:
#there must be at least K nodes of core-value K, where K is the largest corevalue
function isRealizable(core_values::Vector{Int64})
    num_max_core_val = findlast(x -> x == core_values[1], core_values)
    if num_max_core_val < core_values[1]
        return false
    end
    return true
end

#Given a partially completed graph of core-values up to c in the form of row, column, value vectors
#a list of the desired core-values to be added
function addLowerNodes(core_values::Vector{Int64}, I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, k::Int64)
    if k == 0
        return I,J,V
    end
    println("""Core-Level: """, k)
    K = core_values[1]
    higher_nodes = collect(1:findlast(x -> x > k, core_values))
    if  !any(x->x==k, core_values)
        return addLowerNodes(core_values, I, J, V, k-1)
    end
    new_nodes = collect(findfirst(x -> x == k, core_values):findlast(x -> x == k, core_values))
    for node in new_nodes
        rands = collect(sample(higher_nodes, k, replace = false))
        I = vcat(I, [node for i in 1:k], rands)
        J = vcat(J, rands, [node for i in 1:k])
        V = vcat(V, [1 for i in 1:2*k])
    end
    return addLowerNodes(core_values, I, J, V, k-1)
end

#Given empty row, column, and value vectors, K - a core-value, and N - a number of nodes, create a graph of N nodes of 
#core-value K, by the method described in the write-up. This will result in a nearly K-regular graph.
function generateKGraph(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, K::Int64, N::Int64)
    if ! (K % 2 == N % 2)
        rows = collect(1:(N-1))
        cols = collect(2:N)
        I = vcat(I, rows, [rows[1]], cols, N)
        J = vcat(J, cols, N, rows, [rows[1]])
        V = vcat(V, [1 for i in 1:(2*length(rows)+2)])
        z = ceil((N-K+1)/2)
        for i in 1:N
            start = (i+z)%(N) + 1
            stop = (i-z-1)%(N)
            if start <= 0
                start = start + N
            end
            if stop <= 0
                stop = stop + N
            end
            if stop >= start
                edges = collect(Set([r for r in start:stop]))
            else
                edges = collect(Set([r for l in [collect(start:N), collect(1:stop)] for r in l]))
            end
            I = vcat(I, edges, [i for j in 1:length(edges)])
            J = vcat(J, [i for j in 1:length(edges)], edges)
            V = vcat(V, [1 for j in 1:2*length(edges)])
        end
    else
        I, J, V = generateKGraph(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, K, N-1)
        edges = collect(sample(1:(N-1), K, replace=false))
        I = vcat(I, edges, [N for j in 1:length(edges)])
        J = vcat(J, [N for j in 1:length(edges)], edges)
        V = vcat(V, [1 for j in 1:2*length(edges)])
    end
    
    return I, J, V
end

#Given a list of core-values, construct a graph of those core values by the method in the paper
function generateGraph(core_values::Vector{Int64})
    sort!(core_values, rev=true)
    println("""Initializing set:""", core_values)
    if !isRealizable(core_values)
        return false
    end
    K = core_values[1]
    nums = countmap(core_values)
    println("""CountMap: """, nums)
    I::Vector{Int64} = []
    J::Vector{Int64} = []
    V::Vector{Int64} = []
    if K == 0
        return sparse(I,J,V, length(core_values), length(core_values))
    end
    if K == 1
        rows = [i for i in 1:(nums[1]-1)]
        cols = [i for i in 2:nums[1]]
        I = vcat(I, rows, cols)
        J = vcat(J, cols, rows)
        V = vcat(V, [1 for i in 1:2*length(rows)])
        G =  sparse(I, J, V, length(core_values), length(core_values), comb)
        G = convert(SparseMatrixCSC{Int64,Int64}, G)
        return G
    end
    
    if K == 2
        rows = [i for i in 1:(nums[2]-1)]
        cols = [i for i in 2:nums[2]]
        I = vcat(I, rows, [rows[1]], cols, [nums[2]])
        J = vcat(J, cols, [nums[2]], rows, [rows[1]])
        V = vcat(V, [1 for i in 1:(2*length(rows)+2)])
        I, J, V =  addLowerNodes(core_values, I, J, V, 1)
        G =  sparse(I, J, V, length(core_values), length(core_values), comb)
        G = convert(SparseMatrixCSC{Int64,Int64}, G)
        return G
    end
    I, J, V = generateKGraph(I, J, V, K, nums[K])
    I, J, V = addLowerNodes(core_values, I, J, V, K-1)
    
    G =  sparse(I, J, V, length(core_values), length(core_values), comb)
    G = convert(SparseMatrixCSC{Int64,Int64}, G)
    return G
end

#Given a graph and a node n, return a list of all neighbors of n
function getNeighbors(G::SparseMatrixCSC{Int64,Int64}, n::Int64)
    nzval = G.nzval
    rowval = G.rowval
    nbrs = Int64[]
    for nzi in G.colptr[n]:(G.colptr[n + 1] - 1)
        if nzval[nzi] == 1
            push!(nbrs, rowval[nzi])
        end
    end
    return nbrs
end

##COUNT AND GENERATE MOVES

#Overestimate the number of edges where each edge has a "movable" endpoint - one endpoint is in a higher core than the other
#Return a number greater than the number of possible endpoint moves
function get_move_endpoints(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64})
    num = 0
    for i in 1:length(I)
        if cores[I[i]] > cores[J[i]]
            num += findfirst(x -> x > cores[J[i]], cores)
        end
    end
    return num
end

#given a random number, generate that endpoint move
function get_move_endpoints_move(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, i::Int64)
    num = 0
    
    j = 0
    while(num < i)
        j = j+1
        if cores[I[j]] > cores[J[j]]
            num += findlast(x -> x > cores[J[j]], cores)
        end
    end
    pos = [n for n in 1:findlast(x -> x > cores[J[j]], cores)]
    new_end = rand(pos)
    while (G[new_end, J[j]] == 1 || new_end == J[j])
        new_end = rand(pos)
    end
    return [(I[j], J[j]), (new_end, J[j])]
end  

#Given a graph, an old edge and a new edge, move the endpoint of the old edge to the new edge
function move_endpoint(G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, edges::Array{Tuple{Int64,Int64},1})
    G[edges[1][1], edges[1][2]] = 0
    G[edges[1][2], edges[1][1]] = 0
    
    G[edges[2][1], edges[2][2]] = 1
    G[edges[2][2], edges[2][1]] = 1
    return G, 0
end

#Overestimate the number of collapsable edges between cores - see paper
#Return a number greater than the number of possible collapse moves
function get_collapse_edges(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64})
    num = 0
    for i in 1:length(cores)
        k = cores[i]
        if k <= cores[end]
            continue
        end
        m = findfirst(x -> x < k, cores)
        edges = [cores[j] for j in m:length(cores) if G[i, j] == 1]
        if length(edges) <= 1
            continue
        end
        nums = countmap(cores)
        lens = sum(l*(get(nums,c,0)-1) for (c,l) in countmap(edges))
        num += lens  
    end
    return num   
end

#Given a random number, generate that collapse move
function get_collapse_edges_move(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, i::Int64)
    num = 0
    n = 1
    for n in 1:length(cores)
        k = cores[n]
        if k <= cores[end]
            continue
        end
        m = findfirst(x -> x < k, cores)
        edges = [j for j in m:length(cores) if G[n, j] == 1]
        edges_c = [cores[e] for e in edges]
        if length(edges) < 1
            continue
        end
        node_nums = countmap(cores)
        num += sum(l*(get(node_nums,c, 0)-1) for (c,l) in countmap(edges_c))
        if num >= i
            while true
                e_1 = edges[rand(1:length(edges))]
                k1 = cores[e_1]
                e_2 = rand(findfirst(x -> x == k1, cores):(findlast(x -> x == k1, cores)))
                new_edge = (e_1, e_2)
                if cores[new_edge[1]] == cores[new_edge[2]] && e_1 != e_2
                    return [(n, new_edge[1]), (n, new_edge[2])]
                end
            end
        end
    end
end

#Collapse edges between cores - see paper
function collapse_edges(G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, pair::Array{Tuple{Int64,Int64},1})
    n1 = pair[1][1]
    n2 = pair[1][2]
    n3 = pair[2][1]
    n4 = pair[2][2]
    
    if (cores[n2] != cores[n4])
        error("Cannot collapse edges if the core values are not the same")
    end
        
    if (n1 > n2)
        error("The edge is backwards")
    end
        
    if (n3 > n4)
        error("The edge is backwards")
    end
    
    if G[n2, n4] == 1
        return G, 1
    end
                    
    G_prime = deepcopy(G)
    
    G[n1, n2] = 0
    G[n3, n4] = 0
    G[n2, n1] = 0
    G[n4, n3] = 0
    
    
    G[n2, n4] = 1
    G[n4, n2] = 1
    
    if checkKCore(G, cores)
        return G, 0
    end
    return G_prime, 1
end

#Overestimate the number of expandable edges between cores - see paper
#Return a number greater than the number of possible expand moves
function get_expand_edges(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64})
    num = 0
    for i in 1:length(I)
        k = cores[I[i]]
        if k >= cores[1]
            continue
        end
        if (cores[I[i]] == cores[J[i]]) && k < cores[1]
            num += length([j for j in 1:findlast(x -> x > k, cores)])
        end
    end
    return num
end

#Given a random number, generate that expand move
function get_expand_edges_move(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, i::Int64)
    num = 0
    for n in 1:length(I)
        k = cores[I[n]]
        if k >= cores[1]
            continue
        end
        if (cores[I[n]] == cores[J[n]]) && k < cores[1]
            num += length([j for j in 1:findlast(x -> x > k, cores)])
            if num >= i
                node = rand([j for j in 1:findlast(x -> x > k, cores)])
                return [(I[n], J[n]), (I[n], node), (J[n], node)]
            end
        end
    end
end

#Expand edges between cores - see paper
function expand_edge(G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, edges::Array{Tuple{Int64,Int64},1})
    I, J, V = findnz(G)
    G_prime = deepcopy(G)
        
    n1 = edges[1][1]
    n2 = edges[1][2]
    n3 = edges[2][2]
    
    if G[n1, n3] == 1 || G[n2, n3] == 1
        return G, 1
    end 
    
    if  cores[n1] != cores[n2]
        print(cores[n1])
        print(cores[n2])
        print(n1)
        print(n2)
        error("Cannot expand edges if the core values are not the same")
    end
    
    if ! (cores[n1] < cores[n3])
        error("Cannot expand edges if the new core is not bigger")
    end
    
    c = cores[n1]
    neighbors_1 = [J[j] for j in 1:length(I) if I[j] == n1 && cores[J[j]] > c]
    neighbors_2 = [J[j] for j in 1:length(I) if I[j] == n2 && cores[J[j]] > c]

    edges::Array{Tuple{Int64,Int64}} = []
        
    if length(neighbors_1) < c
        push!(edges, (n1, n3))
    end
    if length(neighbors_2) < c
        push!(edges, (n2, n3))
    end
        
    if length(edges) == 0
        return G, 1
    end
    
    try
        
        G[n1, n2] = 0
        G[n2, n1] = 0
        
        for e in edges
            G[e[1], e[2]] = 1
            G[e[2], e[1]] = 1
        end
    catch
        print(edges)
        error("Failed to expand edges")
    end
    
    if checkKCore(G, cores)
        return G, 0
    end
    return G_prime, 1
end

#Overestimate the number of cross moves - see paper
#Return a number greater than the number of possible cross moves
function get_cross(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, k::Int64=cores[1])
    possible_edges = [(I[i], J[i]) for i in 1:length(I) if cores[I[i]]==k && cores[J[i]]==k]
    return binomial(length(possible_edges), 2)
end

#Generate a random cross move
function get_cross_move(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, i::Int64, k::Int64=cores[1])
    possible_edges = [(I[i], J[i]) for i in 1:length(I) if cores[I[i]]==k && cores[J[i]]==k]
    while true
        e = rand(possible_edges, 2, replace=false)
        if cores[e[1][1]] == cores[e[2][2]]
            return e
        end
    end
end

#Cross edges
function cross(G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, pair::Array{Tuple{Int64,Int64},1})
    n1 = pair[1][1]
    n2 = pair[1][2]
    n3 = pair[2][1]
    n4 = pair[2][2]
    
    if  length(unique([n1, n2, n3, n4]))!=4
        return G, 0
    end
    
    if !(G[n1, n4] == 0 && G[n2, n3] == 0)
        return G, 0
    end
    
    G[n1, n2] = 0
    G[n2, n1] = 0
    G[n3, n4] = 0
    G[n4, n3] = 0
    
    G[n1, n4] = 1
    G[n4, n1] = 1
    G[n2, n3] = 1
    G[n3, n2] = 1
    
    return G, 1
end


#Overestimate the number of addable edges - see paper
#Return a number greater than the number of possible add moves
function get_add_edge(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, k::Int64=cores[1])
    cores = collect(findfirst(x -> x == k, cores):findlast(x -> x == k, cores))
    G_sub = getindex(G::SparseMatrixCSC{Int64,Int64}, cores, cores)
    I, J, V = findnz(G_sub)
    return (length(I)*length(I) - sum(V))/2
end

#Generate a random add move based on a given random number
function get_add_edge_move(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, i::Int64, k::Int64=cores[1])
    cores = collect(findfirst(x -> x == k, cores):findlast(x -> x == k, cores))
    flag = false
    while ! flag
        i = rand(1:length(cores))
        j = rand(1:length(cores))
        if G[i,j] == 0
            return (i, j)
        end
    end
end

#add an edge
function add_edge(G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, edge::Tuple{Int64,Int64})
    if edge[1] == edge[2]
        return G, 1
    end
    G_prime = deepcopy(G)
    G_prime[edge[1], edge[2]] = 1
    G_prime[edge[2], edge[1]] = 1
    flag = checkKCore(G_prime, cores)
    
    I, J, V = findnz(G)
    I, J, V2 = findnz(G_prime)
    if flag 
        return G_prime, 0
    else
        return G, 1
    end
end

#Overestimate the number of removable edges - see paper
#Return a number greater than the number of possible remove moves
function get_remove_edge(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, k::Int64=cores[1])
    cores = collect(findfirst(x -> x == k, cores):findlast(x -> x == k, cores))
    G_sub = getindex(G::SparseMatrixCSC{Int64,Int64}, cores, cores)
    I, J, V = findnz(G_sub)
    return length(V)
end

#Given a random number, generate the appropriate remove move
function get_remove_edge_move(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, i::Int64, k::Int64=cores[1])
    return (I[i], J[i])
end

#Remove edge
function remove_edge(G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, edge::Tuple{Int64,Int64})
    G_prime = deepcopy(G)
    G_prime[edge[1], edge[2]] = 0
    G_prime[edge[2], edge[1]] = 0
    dropzeros!(G_prime)
    flag = checkKCore(G_prime, cores)
    if flag
        return G_prime, 0
    end
    return G, 1
end

#Overestimate the number of collapsable edges in top core - see paper
#Return a number greater than the number of possible collapse moves
function get_collapse_degrees(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, k::Int64=cores[1])
    nodes = collect(findfirst(x -> x == k, cores):findlast(x -> x == k, cores))
    G_sub = getindex(G, nodes, nodes)
    degrees = sum(G_sub, dims=2)
    
    nodes = [n for n in nodes if degrees[n] > k+1]
    nodes2 = [n for n in nodes if degrees[n] == k]
    
    return binomial(length(nodes2), 2)*length(nodes)
    
end

#Given a random number, generate the relevant collapse move
function get_collapse_degrees_move(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, i::Int64, k::Int64=cores[1])
    nodes = collect(findfirst(x -> x == k, cores):findlast(x -> x == k, cores))
    G_sub = getindex(G, nodes, nodes)
    degrees = sum(G_sub, dims=2)
    
    nodes = [n for n in nodes if degrees[n] > k+1]
    nodes2 = [n for n in nodes if degrees[n] == k]
    
    n1 = rand(nodes)
    n2, n3 = rand(nodes, 2, replace=false)
    
    return [(n1, n2), (n1, n3)]    
end
                
#collapse edges within the top core
function collapse_degrees(G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, pair::Array{Tuple{Int64,Int64}})
    n1 = pair[1][1]
    n2 = pair[1][2]
    n3 = pair[2][1]
    n4 = pair[2][2]

        
    if n1 != n3
        error("Cannot collapse degrees if nodes are different")
    end
    
    if G[n1, n2] == 0 || G[n1,n4] == 0 || G[n2,n4] == 1
        return G, 1
    end
    
    G[n1, n2] = 0
    G[n2, n1] = 0
    G[n3, n4] = 0
    G[n4, n3] = 0
    
    G[n2, n4] = 1
    G[n4, n2] = 1
    return G, 0
end

#Overestimate the number of expandable edges in the top core - see paper
#Return a number greater than the number of possible collapse moves
function get_expand_degrees(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, k::Int64=cores[1])
    nodes= collect(findfirst(x -> x == k, cores):findlast(x -> x == k, cores))
    G_sub = getindex(G, nodes, nodes)
    
    I_sub, J_sub, V_sub = findnz(G_sub)
    return (length(nodes)-2)*length(V_sub)
end

#Given a random number, generate the relevant expand move
function get_expand_degrees_move(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Int64}, G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, i::Int64, k::Int64=cores[1])
    nodes= collect(findfirst(x -> x == k, cores):findlast(x -> x == k, cores))
    G_sub = getindex(G, nodes, nodes)
    
    I_sub, J_sub, V_sub = findnz(G_sub)
    e = rand(1:length(V_sub))
    edge = (I_sub[e], J_sub[e])
    flag = false
    while !flag
        node = rand(nodes)
        if (node != edge[1] && node != edge[2])
            return (node, edge[1], edge[2])
        end
    end
end

#expand edges within the top core
function expand_degrees(G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, edges::Tuple{Int64,Int64,Int64})
    n1 = edges[1]
    n2 = edges[2]
    n3 = edges[3]
    
    if G[n1, n2] == 1 || G[n1, n3] == 1
        return G, 1
    end
    k = cores[1]
    nodes= collect(findfirst(x -> x == k, cores):findlast(x -> x == k, cores))
    
    G_sub = getindex(G, nodes, nodes)
    degrees = vec(sum(G_sub, dims=2))
    
    nbrs = getNeighbors(G_sub, n1)
    num_greater = length([n for n in nbrs if degrees[n] > k])
    if num_greater < k - 1
        G[n1, n2] = 1
        G[n2, n1] = 1

        G[n1, n3] = 1
        G[n3, n1] = 1

        G[n2, n3] = 0
        G[n3, n2] = 0
        return G, 0
    elseif num_greater < k && (degrees[n2] == k || degrees[n3] == k)
        G[n1, n2] = 1
        G[n2, n1] = 1

        G[n1, n3] = 1
        G[n3, n1] = 1

        G[n2, n3] = 0
        G[n3, n2] = 0
        return G, 0
    else
        G_prime = deepcopy(G)
        G_prime[n1, n2] = 1
        G_prime[n2, n1] = 1

        G_prime[n1, n3] = 1
        G_prime[n3, n1] = 1

        G_prime[n2, n3] = 0
        G_prime[n3, n2] = 0
        
        dropzeros!(G_prime)
        
        if checkKCore(G_prime, cores)
            return G_prime, 0
        else
            return G, 1
        end
    end
end

#Choose and implement a single transition
function transition(G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, num_nodes::Int64, test::Bool=true)
    collapsable_pairs = get_collapse_edges
    expandable_edges = get_expand_edges
    movable_edges = get_move_endpoints
    cross_pairs = get_cross
    addable_edges = get_add_edge
    removable_edges = get_remove_edge
    expandable_degrees = get_expand_degrees
    collapsable_degrees = get_collapse_degrees
    
    
    all_counts_gets = [get_collapse_edges, get_expand_edges, get_move_endpoints, get_add_edge, get_remove_edge, get_expand_degrees, get_collapse_degrees]
    
    all_move_gets = [get_collapse_edges_move, get_expand_edges_move, get_move_endpoints_move, get_add_edge_move, get_remove_edge_move, get_expand_degrees_move, get_collapse_degrees_move]
                
    transition_actions = [collapse_edges, expand_edge, move_endpoint, add_edge, remove_edge, expand_degrees, collapse_degrees]
    
    I, J, V = findnz(G)
    
    all_transitions = []
    for a in all_counts_gets
        #@show a
        #@time 
        push!(all_transitions, a(I, J, V, G, cores))
    end
    
    
    num_transitions = sum(all_transitions)
    
    if num_transitions >= num_nodes
        return G, 0, 0, -1, SMALL_NULL
    end
    
    if num_transitions == 0
        print(cores)
        error("No possible transitions!")
    end
    
    d = Geometric(num_transitions/num_nodes)
    x = rand(d, 1)
    num_steps = x[1]+1
    move = rand(1:num_transitions)
    
    i = 1
    while(move - all_transitions[i]) > 0
        move = move - all_transitions[i]
        i += 1
    end
    
    move = convert(Int64, move)
    
    chosen_move = all_move_gets[i](I, J, V, G, cores, move)
    G_p = deepcopy(G)
    G, num_transitions = transition_actions[i](G, cores, chosen_move)
    dropzeros!(G)
    
    #optionally test if graph still has correct core sequence
    if test
        is_sequence = checkKCore(G, cores, true)
        if ! is_sequence
            println(i)
            println(all_move_gets[i])
            println(transition_actions[i])
            println(chosen_move)
            println(num_transitions)
            error("Graph is no longer the correct core sequence")
        end
    end
    return G, num_steps, 1-num_transitions, i, GOOD_RETURN
end

#setup and run the model with given parameters
function run_model(folder::String, f::String, G::SparseMatrixCSC{Int64,Int64}, cores::Vector{Int64}, num_nodes::Int64, test::Bool=false, n::Int64=1, starter_node::Int64=0)
    println("""Started""")
    cores = sort!(cores, rev=true)
    println(cores)
    #G = generateGraph(cores)
    println("""Initiated""")
    step = 0
    move = 0
    prev_step = 0
    num_steps = 0
    actions_0 = []
    actions_1 = []    
    while(move < n)
        if step - prev_step >= 10000
            #@show step
            println("""Step: """, step)
            println("""Transitions: """, move)
                            
            #count of the  number of various types of transitions
            #first one is successful transitions
            #second is unsuccessful transitions
            println(countmap(actions_0))
            println(countmap(actions_1))
                            
            actions_0=[]
            actions_1=[]
            prev_step = step  
            
            #OPTIONAL OUTPUT EVERY 10000 STEPS
            #println(G)
            #CSV.write("../Results/$(folder)/$(f)_$(step)_$move.csv",  DataFrame(G), writeheader=false)
        end
        G, num_steps, num_transition, action, flag = transition(G, cores, num_nodes, test)
        
        if flag != GOOD_RETURN
            if flag == SMALL_NULL
                println("""Updating Null: """, num_nodes*2)
                step = 0
                move = 0
                prev_step = 0
                num_steps = 0
                num_nodes = num_nodes * 2
            else
                println("PROBLEM FOUND")
                return G
            end
        end
        
        
        step += num_steps 
        move += num_transition
        if num_transition == 0
            push!(actions_0, action)
        else
            push!(actions_1, action) 
        end
    end
    #FINAL OUTPUT
    CSV.write("../Results/$(folder)_final/$(f)_$(step)_$move.csv",  DataFrame(G), writeheader=false)
    return G
end

#Main function: set parameters to run the model
function trial(G, folder, name) 
    cores = core_number(SimpleGraph(G))
    println(cores)
    nodes = [i for i in 1:length(cores)]
    d = Dict(zip(nodes, cores))
    tups = sort(collect(d), by=x->x[2], rev=true)
    cores = [t[2] for t in tups]
    nodes = [t[1] for t in tups]
    dict = Dict(nodes[i] => i for i in 1:length(nodes))
    I, J, V = findnz(G)
    
    I2 = [dict[i] for i in I]
    J2 = [dict[i] for i in J]

    V2 = [1 for i in 1:length(J)]
                    
    G2 = sparse(I2, J2, V2)
    dropzeros!(G2)
    S = core_number(SimpleGraph(G2))
    println(S)
    println(G2)            
    num_edges = length(I2)
                            
    println(checkKCore(G2, cores, true))
     
    #Timed for convenience
    @time ret = run_model(folder, name, G2, cores, 50, true, Int(num_edges*100))
end

