import pickle 
import networkx as nx 
import random 
import time 

samples_dir = '../data/SYN100k.pickle'
label_dict_dir = '../data/label_dict.pickle'
entity_mention_dir = '../data/SYN100k-entity_mention.pickle'
note_dict_dir = '../data/note_dict.pickle'

entity_num = 10000
entity_list = list(range(entity_num))

with open(label_dict_dir, 'rb') as f:
    TRUE_LABEL_DICT = pickle.load(f)

NEW_TRUE_LABEL_DICT = dict()

with open(samples_dir, 'rb') as f:
    samples = pickle.load(f)

entity_mention = dict()
new_samples = []
for sample in samples:
    triples = list(nx.get_node_attributes(sample, 'triple').keys())
    
    triple_num = len(triples)
    en = random.randint(int(triple_num // 2) if int(triple_num // 2) > 0 else 1, triple_num)

    entities = random.choices(entity_list, k=en)
    
    for triple in triples:
        e1 = random.choice(entities)
        e2 = random.choice(entities)
        
        if e1 not in entity_mention.keys():
            entity_mention[e1] = []
        if e2 not in entity_mention.keys():
            entity_mention[e2] = []
        m1 = (e1, len(entity_mention[e1]))
        m2 = (e2, len(entity_mention[e2]))
        entity_mention[e1].append(m1)
        entity_mention[e2].append(m2)
        
        r = int(round(time.time() * 1000))
        
        label = TRUE_LABEL_DICT[triple]
        new_triple = (e1, r, e2)
        origin = (m1, r, m2)

        mapping = {triple: new_triple}
        sample = nx.relabel_nodes(sample, mapping)

        sample.nodes[new_triple]['origin'] = origin
        NEW_TRUE_LABEL_DICT[origin] = label 
    
    new_samples.append(sample)

with open(entity_mention_dir, 'wb') as f:
    pickle.dump(entity_mention, f)

with open(samples_dir, 'wb') as f:
    pickle.dump(new_samples, f)

with open(label_dict_dir, 'wb') as f:
    pickle.dump(NEW_TRUE_LABEL_DICT, f)

# {mention: note, ...}
TRUE_NOTE_DICT = dict()

for entity in entity_mention:
    mention_number = len(entity_mention[entity])
    nn = random.randint(1, int(mention_number // 2) if int(mention_number // 2) > 0 else 1)
    note_begin = str(int(round(time.time() * 1000)))
    possible_notes = [note_begin + str(i) for i in range(nn)]
    for mention in entity_mention[entity]:
        TRUE_NOTE_DICT[mention] = random.choice(possible_notes)

with open(note_dict_dir, 'wb') as f:
    pickle.dump(TRUE_NOTE_DICT, f)


