import os
import pickle
import networkx as nx
from networkx.algorithms import isomorphism
import copy

# inputs
graph_dir = 'nell_graph.pickle'
rule_graphs_dir = 'rule_graphs.pickle'
rule_details_dir = 'rule_details.pickle'

# outputs
samples_dir = '../data/samples.pickle'

# load graph as nx.DiGraph
df = open(graph_dir, 'rb')
G = pickle.load(df)
df.close()

# load rule graphs
df = open(rule_graphs_dir, 'rb')
rule_graphs = pickle.load(df)
df.close()

# load rule details
df = open(rule_details_dir, 'rb')
rule_details = pickle.load(df)
df.close()

# get matchings using VF2 in networkx
rule_mappings = [[] for i in range(len(rule_graphs))]
for i, G1 in enumerate(rule_graphs):
    gm = isomorphism.DiGraphMatcher(G, G1, edge_match=isomorphism.categorical_multiedge_match('relation', None))
    gm.subgraph_is_isomorphic()
    for subgraph in gm.subgraph_isomorphisms_iter():
        rule_mappings[i].append(subgraph)

# build TG
changeable_G = copy.deepcopy(G)
TG = nx.DiGraph()

relations = nx.get_edge_attributes(G, 'relation')
# edge_relation_id: {(e1, e2): {relation1: ids, relation2: id2, ...}, ...}
edge_relation_id = dict()
for edge in relations.keys():
    edge_relation_id.setdefault((edge[0], edge[1]), dict())
    edge_relation_id[(edge[0], edge[1])][relations[edge]] = edge[2]
confs = nx.get_edge_attributes(G, 'confidence')
new_confs = dict()
for k in confs.keys():
    new_k = (k[0], k[1], relations[k])
    new_confs[new_k] = confs[k]
confs = new_confs

for k, rm_list in enumerate(rule_mappings):
    rule_graph = rule_graphs[k]  
    rule_detail = rule_details[k]
    pres = rule_detail[0]
    cons = rule_detail[1]
    # premise edges
    pre_pairs = list(pres.keys())
    # conclution edges
    con_pairs = list(cons.keys())
    for m, dic in enumerate(rm_list):
        # dic: a mapping
        # r_node: the m-th instance of the k-th rule
        r_node = (k, m)
        TG.add_node(r_node, state=None)
        # reverse mapping
        dic = dict(zip(dic.values(), dic.keys())) 
        # premise triples
        for pp in pre_pairs:
            e1 = dic[pp[0]]
            e2 = dic[pp[1]]
            triple = (e1, pp[2], e2)
            conf = confs[(e1, e2, pp[2])]
            TG.add_node(triple, label=None, confidence_score=conf, doc_id=None, sentence=None, origin=None, triple=True)
            TG.add_edge(triple, r_node, condition=pres[pp])   
            eid = edge_relation_id[(e1, e2)][pp[2]]
            if (e1, e2, eid) in changeable_G.edges:
                changeable_G.remove_edge(e1, e2, eid)
        # conclusion triples
        for cp in con_pairs:
            e1 = dic[cp[0]]
            e2 = dic[cp[1]]
            triple = (e1, cp[2], e2)
            conf = confs[(e1, e2, cp[2])]
            TG.add_node(triple, label=None, confidence_score=conf, doc_id=None, sentence=None, origin=None, triple=True)
            TG.add_edge(r_node, triple, conclusion=cons[cp])
            eid = edge_relation_id[(e1, e2)][cp[2]]
            if (e1, e2, eid) in changeable_G.edges:
                changeable_G.remove_edge(e1, e2, eid)

samples = list(nx.connected_components(TG.to_undirected()))

triple_list = []
atts = nx.get_edge_attributes(changeable_G, 'relation')
confs = nx.get_edge_attributes(changeable_G, 'confidence')
ids = nx.get_edge_attributes(changeable_G, 'id')
cnt = 0
for node_pair, r in atts.items():
    t = (node_pair[0], r, node_pair[1])
    c = confs[node_pair]
    tmp = nx.DiGraph()
    tmp.add_node(t, label=None, confidence_score=c, doc_id=None, sentence=None, origin=None, triple=True)
    #tmp.add_node(t, label=None, id=id, confidence=c)
    triple_list.append(tmp)
    cnt += 1

sample_list = []
for s in samples:
    edges = list(TG.subgraph(list(s)).edges)
    subgraph = TG.edge_subgraph(edges).copy()
    sample_list.append(subgraph)

samples = sample_list + triple_list

f = open(samples_dir, 'wb')
pickle.dump(samples, f)
f.close()

print('finish TG construction.')