import networkx as nx
import pickle

print('=== Begin generating knowledge graph from triples ===')

label_collection = dict()
fr = open('../raw_data/YAGO_Mturk', 'r')
line = fr.readline()
while line:
    id, label = line.strip('\n').split('\t')
    label_collection[int(id)] = bool(int(label))
    line = fr.readline()
fr.close()

# %%
triple_label = dict()

G = nx.MultiDiGraph()

f = open('../raw_data/beliefs', 'r', encoding='utf-8')
line = f.readline()
while line:
    id, o, r, *s = line.strip('\n').split('\t')
    s = ' '.join(s)

    if int(id) in label_collection:
        G.add_node(o, label=None)
        G.add_node(s, label=None)
    line = f.readline()
f.close()

# only used by pre-processing 
nx.write_gpickle(G, 'yago_graph.pickle')

fw = open('../data/label_dict.pickle', 'wb')
pickle.dump(triple_label, fw)
fw.close()

print('=== End ===')