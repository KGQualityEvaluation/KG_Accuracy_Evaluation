from networkx.classes.function import neighbors
from paras import TRUE_LABEL_DICT, TRUE_NOTE_DICT
import networkx as nx 
import random
from paras import THRE_TYPE
from MCTS import mcts
import copy


class autoAnnotator(object):
    def __init__(self, if_linking, *true_note_dict) -> None:
        pass
    
    # origin triple
    def annotate_triple(self, triple) -> bool:
        return TRUE_LABEL_DICT[triple]
    
    def annotate_entity(self, mention) -> bool:
        return TRUE_NOTE_DICT[mention]


class userAnnotator(object):
    def __init__(self) -> None:
        pass 

    def annotate_triple(self, triple, information) -> bool:
        origin, doc_id, sentence = information
        print('triple after linking:', triple) # 这其实和标注结果无关，标注还是要看original triple
        print('original triple:', '(', origin[0][0], ',', origin[1], ',', origin[2][0], ')')
        print('doc id:', doc_id)
        print('sentence:', sentence)
        label = bool(int(input('Is this triple true of false? (1 for true and 0 for false.) ')))
        return label 
    
    def annotate_entity(self, mention, information) -> str:
        doc_id, sentence, linking_graph = information 
        print('mention:', mention)
        print('doc_id:', doc_id)
        print('sentence:', sentence)
        print('Annotation situation of this entity:')
        notes = list(set(nx.get_node_attributes(linking_graph, 'note').keys()))
        if len(notes) == 0:
            print('0. ')
        for i in range(len(notes)):
            print(str(i) + '. ' + notes[i])
        if_belong = bool(int(input('Is current mention belongs to one of the listed entities? (1 for true and 0 for false.) ')))
        if if_belong:
            idx = int(input('Which one? (use index as input.) '))
            return notes[idx]
        else:
            print('You can take a note and create a new entity for this mention.')
            note = input('Take your note: ')
            return note 


class Helper(object):
    def __init__(self, sample, if_linking, select_type, *args) -> None:
        self.sample = copy.deepcopy(sample)
        self.if_linking = if_linking
        self.select_type = select_type
        if if_linking:
            self.initial_entity_mention()
            self.create_linking()
            self.create_tris()
            self.boost_sample()
            self.create_num = 0
            self.classify_num = 0
            if self.select_type == 'MCTS':
                self.p1 = args[0]
                self.create_prob = args[1]
        else:
            if self.select_type == 'MCTS':
                self.p1 = args[0]
        # self.actions里的action都是不附带更多信息的action
        self.actions = self.initial_actions()
        self.initial_rules()


    def initial_actions(self) -> list:
        # 如果能保证每次初始化helper时用的都是完全没标注过的sample，那么不用检查triple/mention是否被识别也可以
        # 有标签的triple不作为action
        tld = dict(nx.get_node_attributes(self.sample, 'label'))
        ts = list(nx.get_node_attributes(self.sample, 'triple').keys())
        # entity linking nodes不要误加入actions！
        #print('tld', tld)
        triples = [t for t in ts if tld[t] == None]
        if not self.if_linking:
            return triples 
        else:
            ms = []
            for entity in self.entity_mention.keys():
                # mentions: [(mention, doc_id, sentence), ...]
                mentions = self.entity_mention[entity]
                if len(mentions) > 1:
                    # m: (meniton, doc_id, sentence)
                    for m in mentions:
                        ms.append(m[0])
            # 去重
            ms = list(set(ms))
            self.mention_num = len(ms)
            return triples + ms
    
    
    def initial_entity_mention(self) -> None:
        triples = list(nx.get_node_attributes(self.sample, 'triple'))
        mention_triples = dict(nx.get_node_attributes(self.sample, 'origin'))
        doc_ids = dict(nx.get_node_attributes(self.sample, 'doc_id'))
        sentences = dict(nx.get_node_attributes(self.sample, 'sentence'))
        
        # generate self.entity_mention
        self.entity_mention = dict()
        
        for triple in triples:
            #print(triple)
            e1, _, e2 = triple
            mention = mention_triples[triple]
            doc_id = doc_ids[triple]
            sentence = sentences[triple]
            #print(mention)
            m1, _, m2 = mention
            if e1 not in self.entity_mention:
                self.entity_mention[e1] = []
            if e2 not in self.entity_mention:
                self.entity_mention[e2] = []
            self.entity_mention[e1].append((m1, doc_id, sentence))
            self.entity_mention[e2].append((m2, doc_id, sentence))
            # 去重
            self.entity_mention[e1] = list(set(self.entity_mention[e1]))
            self.entity_mention[e2] = list(set(self.entity_mention[e2]))
        
        # generate self.mention_entity
        self.mention_entity = dict()
        for entity in self.entity_mention:
            mts = self.entity_mention[entity]
            for mt in mts:
                mention = mt[0]
                self.mention_entity[mention] = entity
        
    
    def initial_rules(self):
        self.unused_rules = list(nx.get_node_attributes(self.sample, 'state').keys())
    

    def create_linking(self):
        # generate linking graphs 
        self.linking_graphs = {}
        for entity in self.entity_mention:
            g = nx.Graph()
            for t in self.entity_mention[entity]:
                mention, doc_id, sentence = t
                g.add_node(mention, true_note=None, note=None, doc_id=doc_id, sentence=sentence)
            # 任意两点之间加边
            nodes = list(g.nodes)
            for i in range(len(nodes)):
                n1 = nodes[i]
                for j in range(i+1, len(nodes)):
                    n2 = nodes[j]
                    g.add_edge(n1, n2, label=None)
            self.linking_graphs[entity] = g 
    
    def create_tris(self):
        self.tris = dict()
        for entity in self.linking_graphs.keys():
            graph = self.linking_graphs[entity]
            cc = list(nx.connected_components(graph))
            triangles = [c for c in cc if len(c) == 3]
            edge_triangles = dict()
            for tri in triangles:
                list_tri = list(tri)
                pairs = [(min(list_tri[i], list_tri[j]), max(list_tri[i], list_tri[j])) for i in range(3) for j in range(i+1, 3)]
                for pair in pairs:
                    if pair not in edge_triangles:
                        edge_triangles[pair] = []
                    edge_triangles[pair].append(pairs)
            #self.tris.append(edge_triangles)
            self.tris[entity] = edge_triangles
        

    def boost_sample(self) -> None:
        cnt = 0
        rules = list(nx.get_node_attributes(self.sample, 'state').keys())
        mention_triples = dict(nx.get_node_attributes(self.sample, 'origin'))
        # 规则的原标签
        condition_labels = dict(nx.get_edge_attributes(self.sample, 'condition'))
        conclusion_labels = dict(nx.get_edge_attributes(self.sample, 'conclusion'))

        for rule in rules:
            preds = list(self.sample.predecessors(rule))    # 前驱/condition
            succs = list(self.sample.successors(rule))  # 后继/conclusion
            triples = succs + preds 
            
            # 构造rule周围的局部entity_mentions
            entity_mentions = dict()
            for triple in triples:
                e1, _, e2 = triple
                mention = mention_triples[triple]
                m1, _, m2 = mention
                if e1 not in entity_mentions:
                    entity_mentions[e1] = []
                if e2 not in entity_mentions:
                    entity_mentions[e2] = []
                entity_mentions[e1].append(m1)
                entity_mentions[e2].append(m2)
                # 去重
                entity_mentions[e1] = list(set(entity_mentions[e1]))
                entity_mentions[e2] = list(set(entity_mentions[e2]))

            # 当前rule要考虑的所有linking nodes
            linking_nodes = []
            for entity in entity_mentions:
                mentions = entity_mentions[entity]
                if len(mentions) > 1:
                    mention_pairs = [(min(mentions[i], mentions[j]), max(mentions[i], mentions[j])) for i in range(len(mentions)) for j in range(i+1, len(mentions))]
                    linking_nodes += mention_pairs
                    for pair in mention_pairs:
                        self.sample.add_node(pair, label=None, linking=True)
                        self.sample.add_edge(pair, rule, condition=True)
            
            # 对每个linking node构建新规则，新增规则节点
            for node in linking_nodes:
                new_rule = 'linking-' + str(cnt)
                cnt += 1
                # 这个linking node为结论（False）
                self.sample.add_edge(new_rule, node, conclusion=False)
                # 其他linking nodes+triple nodes为前提
                res_nodes = [n for n in linking_nodes if n != node]
                for res_node in res_nodes:
                    self.sample.add_edge(res_node, new_rule, condition=True)
                for triple in preds:
                    label = condition_labels[(triple, rule)]
                    self.sample.add_edge(triple, new_rule, condition=label)
                for triple in succs:
                    # 与旧规则出现矛盾（矛盾作为新规则的前提）
                    label = not conclusion_labels[(rule, triple)]
                    self.sample.add_edge(triple, new_rule, condition=label)

    
    # 返回(triple/entity, info)
    def select_one(self) -> tuple:
        # 只有一个就直接返回！
        if len(self.actions) == 1:
            action = self.actions[0]
        
        if self.select_type == 'MCTS':
            action = self.select_one_MCTS()
        elif self.select_type == 'linking_first':
            action = self.select_one_entity()
        else:
            action = self.select_one_random()

        # triple
        if len(action) == 3:
            doc_id = self.sample.nodes[action]['doc_id']
            sent = self.sample.nodes[action]['sentence']
            return ('triple', action, (self.sample.nodes[action]['origin'], doc_id, sent))
        
        # mention
        if len(action) == 2:
            entity = self.mention_entity[action]
            linking_graph = self.linking_graphs[entity]
            doc_id = linking_graph.nodes[action]['doc_id']
            sent = linking_graph.nodes[action]['sentence']
            return ('mention', action, (doc_id, sent, linking_graph))
        
        #raise Exception('Invalid action (not mention or triple)')
    
    def select_one_random(self) -> tuple:
        # 选到的可能是mention也可能是triple
        action = random.choice(self.actions)
        return action 
        
    def select_one_entity(self) -> tuple:
        # 先随机把mention全选完再选triple
        mentions = []
        for a in self.actions:
            if len(a) == 2:
                mentions.append(a)
        # 如果mention都选完
        if len(mentions) == 0:
            action = random.choice(self.actions)
        else:
            action = random.choice(mentions)
        return action 
    
    def select_one_MCTS(self) -> tuple:
        if self.if_linking:
            # args: 传入entity_mention, mention_entity, create_prob
            #print('actions:', self.actions)
            m = mcts((self.sample, self.linking_graphs), self.actions, self.unused_rules, self.if_linking, THRE_TYPE, self.p1, self.entity_mention, self.mention_entity, self.create_prob)
        else:
            m = mcts(self.sample, self.actions, self.unused_rules, self.if_linking, THRE_TYPE, self.p1)
        action = m.run() 
        return action 
    
    def update_triple(self, triple, label) -> None:
        nx.set_node_attributes(self.sample, {triple: {'label': label}})
        # 更新self.actions
        self.actions.remove(triple)
    
    '''
    def update_entity(self, mention, note) -> None:
        entity = self.mention_entity[mention]
        graph = self.linking_graphs[entity]
        # 更新当前mention的note
        nx.set_node_attributes(graph, {mention: {'note': note}})
        # 更新self.actions
        self.actions.remove(mention)
        neighbors = list(graph.neighbors(mention))
        for n in neighbors:
            # 看是否能更新边
            if graph.nodes[n]['note'] != None:
                if graph.edges[(n, mention)]['label'] == None:
                    label = (graph.nodes[n]['note'] == note)
                    attr = {'label': label}
                    nx.set_edge_attributes(graph, {(n, mention): attr})
                    # 如果更新了，再更新self.sample
                    # self.sample中的linking点中mention是要考虑顺序的，都按(min, max)的顺序
                    #linking_node = (min(n, mention), max(n, mention))
                    #nx.set_node_attributes(self.sample, {linking_node: attr})
            # 看有label的边能不能更新没有note的别的mention，如果边的label是True则可以更新note
            if graph.edges[(n, mention)] == True:
                if graph.nodes[n]['note'] == None:
                    attr = {'note': note}
                    nx.set_node_attributes(graph, {n: attr})
                    # 更新self.actions
                    self.actions.remove(n)
        
        if self.if_new_note(mention, note):
            self.create_num += 1
        else:
            self.classify_num += 1
    '''    
        
    '''
    def update_linking(self, linking, label) -> None:
        mention1, mention2 = linking
        entity = self.mention_entity[mention1]
        linking_graph = self.linking_graphs[entity]
        # 更新self.sample上linking node的label
        self.sample.nodes[linking]['label'] = label
        # 更新linking graph上的边
        linking_graph.edges[linking]['label'] = label
        # 检查能否更新mention
        if label == True:
            note1 = linking_graph.nodes[mention1]['note']
            note2 = linking_graph.nodes[mention2]['note']
            if note1 != None and note2 == None:
                # 这里应该用update entity
                self.update_entity(mention2, note1)
                #linking_graph.nodes[mention2]['note'] = note1 
                #self.actions.remove(mention2)
            elif note2 != None and note1 == None:
                self.update_entity(mention1, note2)
                #linking_graph.nodes[mention1]['note'] = note2
                #self.actions.remove(mention1)
    '''

    def update_graph(self, graph):
        def update_one_round(graph):
            if_update = False 
            nodes = list(graph.nodes)
            node_notes = nx.get_node_attributes(graph, 'note')

            # nodes
            for node in nodes:
                note = node_notes[node]
                if note != None:
                    neighbors = list(graph.neighbors(node))
                    for n in neighbors:
                        if graph.edges[(node, n)] == True and node_notes[n] == None:
                            if_update = True 
                            graph.nodes[n]['note'] = note 
            
            # remove from self.actions (not update)
            for node in nodes:
                note = node_notes[node]
                if note != None and node in self.actions:
                    self.actions.remove(node)
                    continue 
                if node in self.actions:
                    neighbors = nx.neighbors(graph, node)
                    labels = [graph.edges[(node, n)]['label'] for n in neighbors]
                    if not None in labels:
                        self.actions.remove(node)
            
            # edges 
            #edge_labels = nx.get_edge_attributes(graph, 'label')
            for edge in graph.edges:
                label = graph.edges[edge]['label']
                if not label == None:
                    continue 
                node1, node2 = edge 
                note1 = node_notes[node1]
                note2 = node_notes[node2]
                if note1 == None or note2 == None:
                    continue 
                else:
                    if_update = True
                    graph.edges[edge]['label'] = bool(note1 == note2)
            
            # edge-inference 
            for edge in graph.edges:
                now_edge = (min(edge), max(edge))
                label = graph.edges[edge]['label']
                if not label == None:
                    continue
                m = edge[0]
                entity = self.mention_entity[m]
                entity_tris = self.tris[entity]
                #print(self.tris)
                #print(self.tris[entity])
                if now_edge not in entity_tris:
                    continue 
                tri_list = entity_tris[now_edge]
                for tri in tri_list:
                    ls = []
                    for e in tri:
                        if e == now_edge:
                            continue 
                        else:
                            #print('edge labels:', dict(edge_labels))
                            l = graph.edges[e]['label']
                            if l == None:
                                continue 
                            else:
                                ls.append(l)
                    if len(ls) == 2:
                        if True in ls:
                            if_update = True
                            ls.remove(True)
                            graph.edges[now_edge]['label'] = ls[0] 
 
            return if_update
        
        while update_one_round(graph):
            pass 

    def synch_graph2sample(self, graph):
        label_dict = dict(nx.get_edge_attributes(graph, 'label'))
        #print(label_dict)
        for edge in label_dict.keys():
            label = label_dict[edge]
            if label != None:
                pair = (min(edge), max(edge))
                #print(self.sample.nodes)
                # graph上有的边，sample上未必有
                if pair in self.sample.nodes:
                    self.sample.nodes[pair]['label'] = label

    def synch_sample2graphs(self, linking_nodes_dict):
        graphs = []
        
        for linking_node in linking_nodes_dict.keys():
            label = linking_nodes_dict[linking_node]
            m1, m2 = linking_node
            e = self.mention_entity[m1]
            graph = self.linking_graphs[e]
            graph.edges[linking_node]['label'] = label
            graphs.append(graph)
        
        graphs = list(set(graph))
        for graph in graphs:
            self.update_graph(graph)
            self.synch_graph2sample(graph)

    def infer(self) -> None:
        # 循环推理至没有新的可推理处
        if self.if_linking:
            result = self.infer_linking()
            while result != False:
                if len(result) != 0:
                    self.synch_sample2graphs(result)
                result = self.infer_linking()
        else:
            #print('begin inferring')
            #print(self.infer_simple())
            while self.infer_simple():
                pass
    
    
    # 简单推理，只在self.sample上推理且self.sample上只有triple node，没有linking node
    def infer_simple(self) -> bool:
        possible_rules = []
        triples = list(nx.get_node_attributes(self.sample, 'triple').keys())
        #print('triples', triples)
        #print('actions', self.actions)
        labeled_triples = list(set(triples).difference(set(self.actions)))
        #print('labeled triples', labeled_triples)
        for t in labeled_triples:
            rs = list(self.sample.successors(t))
            possible_rules += list(set(rs).intersection(set(self.unused_rules)))
        #print('possible rule', possible_rules)
        #print('lnegth of possible rules', len(possible_rules))
        if len(possible_rules) == 0:
            return False 
        else:
            if_infer = False
            for r in possible_rules:
                # 先检查下规则是否真的没被使用
                if r not in self.unused_rules:
                    continue 

                conds = list(self.sample.predecessors(r))
                if_labeled = [t not in self.actions for t in conds]
                #print('if_labeled', if_labeled)
                # 如果有没标注的就无法使用这个规则
                if False in if_labeled:
                    continue 
                else:
                    self.sample.nodes[r]['state'] = 'used'
                    self.unused_rules.remove(r)
                    can_infer = True
                    # 这个规则的结论情况
                    conc = list(self.sample.successors(r))[0]
                    conc_label = self.sample.edges[(r, conc)]['conclusion']
                    conc_node_label = self.sample.nodes[conc]['label']
                    # 检查入节点label是否满足规则条件
                    for cond in conds:
                        cond_label = self.sample.nodes[cond]['label']
                        rule_label = self.sample.edges[(cond, r)]['condition']
                        # 还要看结论是否已经被标注了
                        if cond_label != rule_label or conc_node_label != None:
                            can_infer = False 
                            break 
                    if can_infer:
                        if_infer = True 
                        self.update_triple(conc, conc_label)
                        rs = list(self.sample.successors(conc))
                        possible_rules += list(set(rs).intersection(set(self.unused_rules)))
                    else:
                        continue 
            return if_infer


    # 同时考虑self.sample和self.linking_graphs的推理，self.sample上有linking node
    def infer_linking(self) -> bool:
        # self.sample上的推理
        possible_rules = []
        result = dict()
        label_dict = nx.get_node_attributes(self.sample, 'label')
        labeled_nodes = [n for n in label_dict.keys() if label_dict[n] != None]
        for t in labeled_nodes:
            rs = list(self.sample.successors(t))
            possible_rules += list(set(rs).intersection(set(self.unused_rules)))
        if len(possible_rules) == 0:
            return False 
        else:
            if_infer = False
            for r in possible_rules:
                # 先检查下规则是否真的没被使用
                if r not in self.unused_rules:
                    continue 

                conds = list(self.sample.predecessors(r))
                if_labeled = [t not in self.actions for t in conds]
                # 如果有没标注的就无法使用这个规则
                if False in if_labeled:
                    continue 
                else:
                    self.sample.nodes[r]['state'] = 'used'
                    self.unused_rules.remove(r)
                    # 检查这个rule能不能用
                    can_infer = True
                    # 这个规则的结论情况
                    conc = list(self.sample.successors(r))[0]
                    conc_label = self.sample.edges[(r, conc)]['conclusion']
                    conc_node_label = self.sample.nodes[conc]['label']
                    # 检查入节点label是否满足规则条件
                    for cond in conds:
                        cond_label = self.sample.nodes[cond]['label']
                        rule_label = self.sample.edges[(cond, r)]['condition']
                        if cond_label != rule_label or conc_node_label != None:
                            can_infer = False 
                            break 
                    if can_infer:
                        if_infer = True 
                        
                        # 如果结论是triple node
                        if len(conc) == 3:
                            self.update_triple(conc, conc_label)
                        elif len(conc) == 2:
                            # 如果结论是linking node
                            #self.update_linking(conc, conc_label)
                            result[conc] = conc_label
                            self.sample.nodes[conc]['label'] = conc_label
                        else:
                            print('invalid conclusion!')
                        rs = list(self.sample.successors(conc))
                        possible_rules += list(set(rs).intersection(set(self.unused_rules)))
                    else:
                        continue
            
            if if_infer == False:
                return False
            else:
                return result 

    # finish, labels里没有None，返回True
    def check_finish(self) -> bool:
        labels = list(nx.get_node_attributes(self.sample, 'label').values())
        #print(nx.get_node_attributes(self.sample, 'label'))
        if not self.if_linking:
            return not (None in labels)
        else:
            if_finish = True 
            for entity in self.linking_graphs.keys():
                graph = self.linking_graphs[entity]
                ls = list(nx.get_edge_attributes(graph, 'label').values())
                if None in ls:
                    if_finish = False
                    break 
            return (not (None in labels)) and if_finish
    
    def output(self) -> tuple:
        if self.if_linking:
            return (self.sample, self.linking_graphs)
        else:
            return (self.sample,)
    
    def if_new_note(self, mention, note):
        entity = self.mention_entity[mention]
        linking_graph = self.linking_graphs[entity]
        notes = set(nx.get_node_attributes(linking_graph, 'note').values())
        return note not in notes 
    
    def get_triple_num(self) -> int:
        return len(nx.get_node_attributes(self.sample, 'triple').keys())

    def get_linking_num(self) -> int:
        cnt = 0
        for lg in self.linking_graphs.values():
            # g.size(): g中的边数
            cnt += lg.size()
        return cnt 
    
    def get_mention_num(self) -> int:
        return self.mention_num 

    def report(self) -> None:
        print('=== Current IG situation ===')
        print('#Triples:', self.get_triple_num())
        if self.if_linking:
            print('#Linking pairs:', self.get_linking_num())
            print('#Mentions:', len(self.mention_entity.keys()))
        print()


if __name__ == "__main__":
    graph = nx.DiGraph()
    graph.add_node((1, 'r1', 3), origin=((1,1), 'r1', (3,0)), true_label=True, label=None, confidence=100, doc_id=1, sentence='sent', triple=True)
    graph.add_node((1, 'r2', 2), origin=((1,0), 'r2', (2,0)), true_label=True, label=None, confidence=100, doc_id=1, sentence='sent', triple=True)
    graph.add_node((2, 'r3', 3), origin=((2,1), 'r3', (3,1)), true_label=True, label=None, confidence=99, doc_id=2, sentence='sent', triple=True)
    graph.add_node('rule', state=None)
    graph.add_edge((1, 'r1', 3), 'rule', condition=True)
    graph.add_edge((1, 'r2', 2), 'rule', condition=True)
    graph.add_edge('rule', (2, 'r3', 3), conclusion=True)

    helper = Helper(graph, True)
    print()
    
        
