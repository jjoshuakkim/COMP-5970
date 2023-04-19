from Bio import AlignIO, Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from graphviz import Graph
import numpy as np

with open("msa.pir") as pir_file:           # Strategy that I mocked somewhere online to convert pir file into a fasta file bc I could not get pir to work
    fasta_records = []
    for line in pir_file:
        if line.startswith(">"):
            fasta_records.append(f">{line[1:].strip()}\n")
        else:
            fasta_records[-1] += line.replace(".", "-")

    with open("msa.fasta", "w") as fasta_file:
        fasta_file.writelines(fasta_records)

align = AlignIO.read("msa.fasta", "fasta")  # Parse multiple sequence alignment

calculator = DistanceCalculator('identity') # Speedy way to calc dist matrix
dm = calculator.get_distance(align)
print("Original distance matrix:")
print(dm)

constructor = DistanceTreeConstructor()
tree = constructor.nj(dm)                   # Makes the phylogenetic tree using nj alg

names = []
for clade in tree.get_terminals() + tree.get_nonterminals():
    names.append(clade.name)                # Adds all terminals and non-terminals in the tree into a list

matrix = np.zeros((len(names), len(names)))
for row, name1 in enumerate(names):         # For loop calcs dist between each pair of nodes
    for col, name2 in enumerate(names):
        if name1 == name2:
            matrix[row, col] = 0            # Dist from a node to itself is 0
        else:
            mrca = tree.common_ancestor(name1, name2)   # Find the most recent common ancestor of 2 nodes
            matrix[col, row] = mrca.distance(name1) + mrca.distance(name2)      # This contains the pairwise dist between each node and the mrca
            
print("\nDistance Matrix with Internal Nodes:")
print("     " + "   ".join(names))                      # Joins node names
for i, name1 in enumerate(names):                       
    row = [name1]                                       # Names followed by dist
    for j in range(len(names)):
        row.append("{:.6f}".format(matrix[i,j]))        # Rounds distances 6 spots

    for x in row:
        print(x, end=" ")
    print()

graph = Phylo.to_networkx(tree)         # Converts tree to networkx graph
edges = graph.edges()                   # Get edges
dot = Graph()

for node in graph.nodes():
    dot.node(str(node), str(node))      # Add nodes to graph

for edge in edges:
    start, end = edge
    dot.edge(str(start), str(end))      # Add the edge of the start and end node to graph
dot.render("dot_graph", view=True)      # Displays graph as pdf