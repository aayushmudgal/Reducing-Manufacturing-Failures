
# coding: utf-8

# ### Visualizing the first 1000 Defective Ids and first 1000 Non-Defective Ids using [IBM System G](http://systemg.research.ibm.com/) and their movements across the production lines to develop useful insights about the data.
# 

# ## Import Node.csv and Edge.csv to system G visualizer for the respective graphs for visualization:
# 
# ### Graph -- Negative:
# 
# #### Visualizing the nodes that first 1000 Defective Ids visit to. There is an edge between every adjacent stations.
# 
# - Commands for visualizing the graph
# ---
# - [gremlin g=CreateGraph.openGraph("sgtrans","negative")]
# - [get_num_vertices --graph negative]
# - [print_all --graph negative]
# 
# >>number of nodes: 119
# >>number of edges: 407
# 
# <img src="../plots/negative.png">

# 
# - Visualizing near the station that has the highest degree
# ---
#  <img src="../plots/egonet-negative.png"/>

# 
# 
# - Command to get the stations that have more Ids coming into them
# ---
# - analytic_get_top_vertices_by_prop --graph negative --prop analytic_degree_in --topnum 10 <img src="../plots/negative_in_10.png"/>

# 
# - The stations that have the maximum number of Ids coming out of them
# ---
# 
# - analytic_get_top_vertices_by_prop --graph negative --prop analytic_degree_out --topnum 10
# 
# <img src="../plots/negative_out_10.png">
# 

# ### Graph -- Positive
# 
# #### Visualizing the nodes that first 1000 Non-Defective Ids visit to. There is an edge between every adjacent stations.
# 
# - visualizing the graph
# ---
# 
# - [gremlin g=CreateGraph.openGraph("sgtrans","positive")]
# - [get_num_vertices --graph positive]
# - [print_all --graph positive]
# 
# >>number of nodes: 119
# >>number of edges: 407
# 
# <img src="../plots/positive.png">

# - The stations that have more Ids coming into them
# ---
# - analytic_get_top_vertices_by_prop --graph negative --prop analytic_degree_in --topnum 10 
# <img src="../plots/positive_in_10.png"/>

# 
# - The stations that have the maximum number of Ids coming out of them
# ---
# 
# - analytic_get_top_vertices_by_prop --graph negative --prop analytic_degree_out --topnum 10
# 
# <img src="../plots/positive_out_10.png">
# 

# ## Observations
# Different stations/lines are more frequently visited by Faulty Items than the Non-Faulty Ones. Thus strengthening the belief that they are some sort of stations that act more crucial for deciding faults in the product, these are where more recordings and stricter tests are undergone to test for manufacturing quality. 
