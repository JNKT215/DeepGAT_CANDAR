 
import argparse
import networkx as nx
from igraph import Graph
from torch_geometric.datasets import Planetoid,Coauthor,Flickr
from torch_geometric.utils import to_networkx,is_undirected,add_self_loops,remove_self_loops

def add_self_loop_func(data):
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
    return edge_index

def calc_hub_node_num(nxg,hub_degree):
    hub_nodes = [node for node, degree in dict(nxg.degree).items() if degree >= hub_degree]
    return hub_nodes

def calc_diameter_igprah(num_edges,num_nodes):
    g = Graph(directed=False) 
    g.add_vertices(num_nodes)  
    g.add_edges(num_edges.T.numpy())  
    diameter = g.diameter()
    return  diameter
    
def main():
    root = "." + '/data/' + args.name
    
    if args.name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root      = root,
                            name      = args.name)
    elif args.name in ['CS', 'Physics']:
        dataset = Coauthor(root      = root,
                        name      = args.name)
    elif args.name == "Flickr":
        dataset = Flickr(root=root)
    
    print(f"Datast: {args.name}")
    print("num_graph:", len(dataset)) 
    print("num_class:",dataset.num_classes)  
    
    data = dataset[0]
    if args.with_self_loop:
        data.edge_index = add_self_loop_func(data) 
    print(f"num_nodes:{data.num_nodes}")     

    nx_g = to_networkx(data=data)
    if args.normal_undirected:
        nx_g = nx.to_undirected(nx_g)
    
    min_degree = min(dict(nx_g.degree).values())
    avg_degree = sum(dict(nx_g.degree).values()) / len(nx_g)
    max_degree = max(dict(nx_g.degree).values())
    density =  nx.density(nx_g)
    average_clustering =  nx.average_clustering(nx_g)
    diameter = calc_diameter_igprah(data.edge_index,data.num_nodes)
    hub_nodes = calc_hub_node_num(nx_g,args.hub_degree)
    percentage_of_hub_nodes = len(hub_nodes)/data.num_nodes
    
    
    print(f"graph_diameter: {diameter}")
    print(f"min_degree: {min_degree}")
    print(f"avg_degree: {avg_degree}")
    print(f"max_degree: {max_degree}")
    print(f"percentage of hub node in graph:{percentage_of_hub_nodes * 100:.2f}%")
    print(f"density: {density * 100:.2f}%")
    print(f"average_clustering: {average_clustering* 100:.2f}%")
    print(f"num_hub_nodes ( have over {args.hub_degree} degree ): {len(hub_nodes)}")
    print()    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='Cora')
    parser.add_argument('--with_self_loop', type=bool, default=False)
    parser.add_argument('--hub_degree', type=int, default=30)
    parser.add_argument('--normal_undirected', type=bool, default=False)
    args = parser.parse_args()
    main()