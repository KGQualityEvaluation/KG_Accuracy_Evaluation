import pickle 
import networkx as nx
import random 
import numpy as np 
from scipy import stats as st
from collections import Counter 

with open('../SYN-IG/new-SYN10w.pickle', 'rb') as f:
    samples = pickle.load(f)

with open('../SYN-IG/SYN10w-entity_mention.pickle', 'rb') as f:
    entity_mention = pickle.load(f)

with open('../SYN-IG/new-SYN10wLabelDict.pickle', 'rb') as f:
    TRUE_LABEL_DICT = pickle.load(f)

with open('../SYN-IG/new-SYN10wNoteDict.pickle', 'rb') as f:
    TRUE_NOTE_DICT = pickle.load(f)

mention_entity = dict()

for entity in entity_mention:
    mentions = entity_mention[entity]
    for mention in mentions:
        mention_entity[mention] = entity

triple_origin = dict()

for sample in samples:
    tmp = dict(nx.get_node_attributes(sample, 'origin'))
    triple_origin.update(tmp)

triples = list(triple_origin.keys())

head_triples = dict()

for triple in triples:
    head_entity = triple[0]
    if head_entity not in head_triples.keys():
        head_triples[head_entity] = []
    head_triples[head_entity].append(triple)

heads = list(head_triples.keys())

sizes = [len(head_triples[e]) for e in head_triples.keys()]
ws = [sizes[i] / sum(sizes) for i in range(len(sizes))]

def get_estimator(head_labels):
    eu = []
    for labels in head_labels:
        eu.append(sum(labels) / len(labels))
    return sum(eu) / len(eu)

def get_MoE(head_labels):
    eu = []
    for labels in head_labels:
        eu.append(sum(labels) / len(labels))
    sigma = np.std(eu, ddof = 1)
    sizes = [len(labels) for labels in head_labels]
    tmp = [sizes[i] ** 2 for i in range(len(eu))]
    return z * ((1 / sum(sizes) ** 2) * sum(tmp)) ** 0.5 * sigma

def get_time(triple_num, entity_num):
    return (entity_num * 45 + triple_num * 25) / 3600

m = 5
alpha = 0.05
z = st.norm.isf(alpha / 2)

head_labels = []
MoE = float('inf')
triple_num = 0
entity_num = 0
while MoE > 0.05:
    head = random.choices(population=heads, weights=ws, k=1)[0]
    entity_num += 1
    
    pool = head_triples[head]
    ss = random.sample(pool, min(m, len(pool)))
    triple_num += len(ss)
    origins = [triple_origin[t] for t in ss]
    # 找哪些note占比最大，就识别成那个实体
    origin_heads = [t[0] for t in origins]
    notes = [TRUE_NOTE_DICT[h] for h in origin_heads]
    num_dict = Counter(notes)
    best_note = max(num_dict, key=num_dict.get)
    pre_labels = [(notes[i] == best_note) for i in range(len(notes))]
    
    labels = []
    for i, origin in enumerate(origins):
        pre_label = pre_labels[i]
        if pre_label:
            label = TRUE_LABEL_DICT[origin]
            labels.append(label)
        else:
            labels.append(False)
    
    head_labels.append(labels)

    if triple_num >= 30:
        estimator = get_estimator(head_labels)
        MoE = get_MoE(head_labels)

time = get_time(triple_num, entity_num)
estimator = get_estimator(head_labels)
MoE = get_MoE(head_labels)

print('estimator:')
print(estimator)
print('MoE:')
print(MoE)
print('time:')
print(time)
print('entity number:')
print(entity_num)
print('triple number:')
print(triple_num)
