import pickle
from types import new_class 
import networkx as nx 
import random
import numpy as np
from collections import Counter

samples_dir = '../data/SYN100k.pickle'
label_dict_dir = '../data/label_dict.pickle'

# generate rules
# rules: [{'condition': [bool, ...], 'conclusion': bool}, ...]

# 1 condition 
rules1 = []
condition1 = []
for cond in [True, False]:
    d = dict()
    d['condition'] = [cond]
    condition1.append([cond])
    for conc in [True, False]:
        d['conclusion'] = conc 
        rules1.append(d)

# 2 condition 
rules2 = []
condition2 = []
for cond1 in [True, False]:
    for cond2 in [True, False]:
        d = dict()
        d['condition'] = [cond1, cond2]
        condition2.append([cond1, cond2])
        for conc in [True, False]:
            d['conclusion'] = conc
            rules2.append(d)

# 3 condition 
rules3 = []
condition3 = []
for cond1 in [True, False]:
    for cond2 in [True, False]:
        for cond3 in [True, False]:
            d = dict()
            d['condition'] = [cond1, cond2, cond3]
            condition3.append([cond1, cond2, cond3])
            for conc in [True, False]:
                d['conclusion'] = conc 
                rules3.append(d)

rules = rules1 + rules2 + rules3 
print('#Rules:', len(rules))

pure_TGs = []
TRUE_LABEL_DICT = dict()

cnt = 0

for i, rule_detail in enumerate(rules):
    labels = rule_detail['condition'] + [rule_detail['conclusion']]
    acc = np.mean(labels)
    if acc >= 0.8:
        num = random.randint(1800, 2000)
    else:
        num = random.randint(500, 800)
    for j in range(num):
        sample = nx.DiGraph()
        # rule node
        rule = (i, j)
        sample.add_node(rule, state=None)
        for cond in rule_detail['condition']:
            sample.add_node(cnt, label=None, confidence_score=None, doc_id=None, sentence=None, origin=None, triple=True)
            TRUE_LABEL_DICT[cnt] = cond
            sample.add_edge(cnt, rule, condition=cond)
            cnt += 1
        conc = rule_detail['conclusion']
        sample.add_node(cnt, label=None, confidence_score=None, doc_id=None, sentence=None, origin=None, triple=True)
        TRUE_LABEL_DICT[cnt] = conc
        sample.add_edge(rule, cnt, conclusion=conc)
        cnt += 1
        pure_TGs.append(sample)


def combine(s1, s2) -> nx.DiGraph:
    n = random.random()
    
    if n <= 0.4:
        # condition 1: condition & conclusion
        # cond: triple -> rule
        cond_edge1s = list(nx.get_edge_attributes(s1, 'condition').keys())
        cond_node1s = []
        for cond_edge1 in cond_edge1s:
            cond_node1s.append(cond_edge1[0])
        conc_edge2 = list(nx.get_edge_attributes(s2, 'conclusion').keys())[0]
        conc_node2 = conc_edge2[1]
        for cond_node1 in cond_node1s:
            label1 = TRUE_LABEL_DICT[cond_node1]
            label2 = TRUE_LABEL_DICT[conc_node2]
            if label1 != label2:
                continue 
            else:
                mapping = {conc_node2: cond_node1}
                tmp = nx.relabel_nodes(s2, mapping)
                new_s = nx.compose(s1, tmp)
                del TRUE_LABEL_DICT[conc_node2]
                return new_s

        cond_edge2s = list(nx.get_edge_attributes(s2, 'condition').keys())
        cond_node2s = []
        for cond_edge2 in cond_edge2s:
            cond_node2s.append(cond_edge2[0])
        conc_edge1 = list(nx.get_edge_attributes(s1, 'conclusion').keys())[0]
        conc_node1 = conc_edge1[1]
        for cond_node2 in cond_node2s:
            label1 = TRUE_LABEL_DICT[conc_node1]
            label2 = TRUE_LABEL_DICT[cond_node2]
            if label1 != label2:
                continue 
            else:
                mapping = {cond_node2: conc_node1}
                tmp = nx.relabel_nodes(s2, mapping)
                # 合并s1和tmp
                new_s = nx.compose(s1, tmp)
                del TRUE_LABEL_DICT[cond_node2]
                return new_s

    elif n <= 0.7:
    # condition 2: conclusion & conclusion
        conc_edge1 = list(nx.get_edge_attributes(s1, 'conclusion').keys())[0]
        # conc: rule -> triple
        conc_node1 = conc_edge1[1]
        conc_edge2 = list(nx.get_edge_attributes(s2, 'conclusion').keys())[0]
        conc_node2 = conc_edge2[1]
        label1 = TRUE_LABEL_DICT[conc_node1]
        label2 = TRUE_LABEL_DICT[conc_node2]
        if label1 != label2:
            return False 
        else:
            mapping = {conc_node2: conc_node1}
            tmp = nx.relabel_nodes(s2, mapping)
            new_s = nx.compose(s1, tmp)
            del TRUE_LABEL_DICT[conc_node2]
            return new_s
    else:
    # condition 3: condition & condition 
        cond_edge1s = list(nx.get_edge_attributes(s1, 'condition').keys())
        cond_node1s = []
        for cond_edge1 in cond_edge1s:
            cond_node1s.append(cond_edge1[0])
        cond_edge2s = list(nx.get_edge_attributes(s2, 'condition').keys())
        cond_node2s = []
        for cond_edge2 in cond_edge2s:
            cond_node2s.append(cond_edge2[0])
        for cond_node1 in cond_node1s:
            for cond_node2 in cond_node2s:
                label1 = TRUE_LABEL_DICT[cond_node1]
                label2 = TRUE_LABEL_DICT[cond_node2]
                if label1 != label2:
                    continue 
                else:
                    mapping = {cond_node2: cond_node1}
                    tmp = nx.relabel_nodes(s2, mapping)
                    new_s = nx.compose(s1, tmp)
                    del TRUE_LABEL_DICT[cond_node2]
                    return new_s

    return False

pure_samples = pure_TGs
combined_samples = []
while len(pure_samples) > 1:
    s1 = random.choice(pure_samples)
    pure_samples.remove(s1)
    s2 = random.choice(pure_samples)
    pure_samples.remove(s2)
    result = combine(s1, s2)
    if result != False:
        combined_samples.append(result)
    else:
        combined_samples.append(s1)
        combined_samples.append(s2)

if len(pure_samples) == 1:
    combined_samples.append(pure_samples[0])

sizes = [len(nx.get_node_attributes(s, 'triple')) for s in combined_samples]

pure_triples = []
for i in range(100000 - sum(sizes)):
    rand = random.random()
    if rand <= 0.9:
        label = True
    else:
        label = False
    sample = nx.DiGraph()
    sample.add_node(cnt, label=None, confidence_score=None, doc_id=None, sentence=None, origin=None, triple=True)
    TRUE_LABEL_DICT[cnt] = label 
    cnt += 1
    pure_triples.append(sample) 

pure_samples = combined_samples + pure_triples

all_labels = list(TRUE_LABEL_DICT.values())
acc = np.mean(all_labels)
print('#Triples (in all samples):', len(TRUE_LABEL_DICT))
print('#Samples (include simple triples):', len(pure_samples))
print('Accuracy:', acc)

intervals = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

conf_triples = [[], [], [], [], []]

tg_sizes = []
doc_id_numbers = []

# generate doc id and confidence score randomly 
for sample in pure_samples:
    triples = list(nx.get_node_attributes(sample, 'triple').keys())
    
    tg_sizes.append(len(triples))
    
    id_num = random.randint(1, len(triples))
    doc_id_numbers.append(id_num)
    
    doc_ids = []
    for i in range(len(triples)):
        # doc id begins from 1
        doc_ids.append(random.randint(1, id_num))

    for i, triple in enumerate(triples):
        # confidence score 
        label = TRUE_LABEL_DICT[triple]
        if label == True:
            weights = [0.05, 0.05, 0.2, 0.3, 0.4]
            idx = random.choices(population=list(range(5)), weights=weights, k=1)[0]
            interval = intervals[idx]
            conf = random.uniform(interval[0], interval[1])
        else:
            weights = [0.4, 0.3, 0.2, 0.05, 0.05]
            idx = random.choices(population=list(range(5)), weights=weights, k=1)[0]
            interval = intervals[idx]
            conf = random.uniform(interval[0], interval[1])
        sample.nodes[triple]['confidence_score'] = conf
        conf_triples[idx].append(label)

        # doc id 
        sample.nodes[triple]['doc_id'] = doc_ids[i]

for i, labels in enumerate(conf_triples):
    interval = intervals[i]
    accs = np.mean(labels)
    print(interval, accs)

NEW_TRUE_LABEL_DICT = dict()
final_samples = []
for sample in pure_samples:
    triples = list(nx.get_node_attributes(sample, 'triple').keys())
    new_triples = []
    for triple in triples:
        label = TRUE_LABEL_DICT[triple]
        new_triple = (triple, '', '')
        new_triples.append(new_triple)
        NEW_TRUE_LABEL_DICT[new_triple] = label 
    mapping = dict(zip(triples, new_triples))
    new_sample = nx.relabel_nodes(sample, mapping)
    final_samples.append(new_sample)

with open(samples_dir, 'wb') as f:
    pickle.dump(final_samples, f)

with open(label_dict_dir, 'wb') as f:
    pickle.dump(NEW_TRUE_LABEL_DICT, f)    


