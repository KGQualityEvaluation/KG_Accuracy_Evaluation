import networkx as nx
import pickle

print('=== Begin changing rules into graphs (will be used in the TG constructoin stage) ===')
rules_dir = '../raw_data/horn_clause'
f = open(rules_dir, 'r')
line = f.readline()
G_list = []

rule_details = []
while line:
    pre, con = line.strip('\n').split(' => ')
    pre_list = pre.split('\t')
    con_list = con.split('\t')
    pre_dict = dict()
    con_dict = dict()
    G = nx.MultiDiGraph()
    for p in pre_list:
        print(p.split(' '))
        s, r, o, label = p.split(' ')
        label = bool(int(label))
        r = r.strip('<').strip('>')
        G.add_edge(s, o, relation=r)
        pre_dict[(s, o, r)] = label
    for c in con_list:
        s, r, o, label = c.split(' ')
        label = bool(int(label))
        r = r.strip('<').strip('>')
        G.add_edge(s, o, relation=r)
        con_dict[(s, o, r)] = label
    G_list.append(G)
    rule_details.append((pre_dict, con_dict))
    line = f.readline()
f.close()

# used only in pre-processing
df = open('rule_graphs.pickle', 'wb')
pickle.dump(G_list, df)
df.close()

# used only in pre-processing
df = open('rule_details.pickle', 'wb')
pickle.dump(rule_details, df) 
df.close()

print('=== End ===')