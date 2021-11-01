import random 
import networkx as nx 
import copy 
from paras import TIME_THRE, SIMU_THRE
import time 
from Timers import Timer
from paras import A1, A2, A3, A4
from paras import C
import numpy as np
from math import log

class mcts(object):
    
    # 如果考虑linking，有args，为entity_mention, mention_entity
    def __init__(self, state, actions, unused_rules, if_linking, thre_type, p1, *args) -> None:
        self.if_linking = if_linking
        self.thre_type = thre_type
        if if_linking:
            self.sample = state[0]
            self.linking_graphs = state[1]
            self.entity_mention = args[0]
            self.mention_entity = args[1]
            self.create_prob = args[2]
        else:
            self.sample = state
        
        # 这两个在之后模拟过程中会被深拷贝然后更新
        self.actions = actions
        self.unused_rules = unused_rules

        # 需要现在进行深拷贝，之后select的时候会直接更新
        self.unsimulated = copy.deepcopy(actions)

        self.UCBs = dict() 
        self.rewards = dict()
        self.p1 = p1
        self.rewards = dict()
        self.N = 0
    
    def select(self) -> tuple:
        if len(self.unsimulated) != 0:
            action = random.choice(self.unsimulated)
            self.unsimulated.remove(action)
        else:
            action = max(self.UCBs, key=self.UCBs.get)
        return action 

    def generate_label(self):
        rand = random.random()
        #print(self.p1)
        if rand < self.p1:
            return True 
        else:
            return False 

    def generate_note(self, mention):
        #print(self.mention_entity)
        #print(self.entity_mention)
        entity = self.mention_entity[mention]
        linking_graph = self.linking_graphs[entity]
        notes = list(nx.get_node_attributes(linking_graph, 'note').values())
        notes = [n for n in notes if n != None]
        
        if len(notes) == 0:
            return str(time.time())
        
        rand = random.random()
        # 新建实体
        if rand < self.create_prob:
            # 用时间戳随机生成note
            return str(time.time())
        # 实体归类，等概率随机选一个有note的mention，返回它的note
        else:
            note = random.choice(notes)
            return note
    
    def if_continue(self, t) -> bool:
        if self.thre_type == 'time':
            return t < TIME_THRE
        elif self.thre_type == 'simu':
            return self.N < SIMU_THRE
        else:
            print('Invalid threshold type! (not time or simu)')

    def simulate(self, action) -> None:
        timer = Timer(A1, A2, A3, A4)
        
        if self.if_linking:
            state = (self.sample, self.linking_graphs)
            helper = simulationHelper(self.if_linking, state, self.actions, self.unused_rules, self.entity_mention, self.mention_entity)
        else:
            state = self.sample
            helper = simulationHelper(self.if_linking, state, self.actions, self.unused_rules)
        
        # 先根据给定的action走一步
        if len(action) == 2:
            note = self.generate_note(action)
            helper.update_entity(action, note)
        elif len(action) == 3:
            label = self.generate_label()
            helper.update_triple(action, label)
        else:
            print('Invalid action!')
        
        if action in self.unsimulated:
            self.unsimulated.remove(action)
        
        while not helper.check_finish():
            action_selected = helper.select_one()
            #print(action_selected)
            # return: ('triple', triple, (origin, doc_id, sentence)) or ('mention', mention, (doc_id, sentence, entity_linking_graph))
            t = action_selected[0]
            information = action_selected[2]
            if t == 'triple':
                triple = action_selected[1]
                label = self.generate_label()
                helper.update_triple(triple, label)
                # update timer
                doc_id = information[1]
                timer.update(t, doc_id)
                #print(timer.triple_num)
            else:
                mention = action_selected[1]
                note = self.generate_note(mention)
                helper.update_entity(mention, note)
                # update timer
                doc_id = information[0]
                if_new = helper.if_new_note(mention, note)
                timer.update(t, doc_id, if_new)
            helper.infer()

        
        # -total_cost就是reward，耗时越少reward越大
        reward = (-1) * timer.total_cost()

        if action not in self.rewards:
            self.rewards[action] = []
        self.rewards[action].append(reward)

        self.N += 1

    # UCB = \bar{R} + C * \sqrt{\frac{2\ln(n)}{n(a)}}
    def update(self, action) -> None:
        #print(self.rewards)
        UCB = np.mean(self.rewards[action]) + C * (2 * log(self.N) / len(self.rewards[action])) ** 0.5 
        self.UCBs[action] = UCB

    def run(self) -> tuple:
        simu_time = 0
        start = time.time()
        t = time.time() - start 
        while self.if_continue(t):
            action = self.select()
            self.simulate(action)
            self.update(action)
            t = time.time() - start 
        # 动作选择标准：UCB最大的
        return max(self.UCBs, key=self.UCBs.get)


class simulationHelper(object):
    
    # 如果考虑linking，有args，为entity_mention, mention_entity
    def __init__(self, if_linking, state, actions, unused_rules, *args) -> None:
        self.if_linking = if_linking
        if self.if_linking:
            self.sample = copy.deepcopy(state[0])
            self.linking_graphs = copy.deepcopy(state[1])
        else:
            self.sample = copy.deepcopy(state)

        self.actions = copy.deepcopy(actions)
        self.unused_rules = copy.deepcopy(unused_rules)

        if self.if_linking:
            self.entity_mention = args[0]
            self.mention_entity = args[1]
    

    # MCTS中下一个任务完全随机选择
    # select_one会返回额外信息！
    def select_one(self) -> tuple:
        # 选到的可能是mention也可能是triple
        #print('action:', self.actions)
        #print('label situation:', dict(nx.get_node_attributes(self.sample, 'label')))
        action = random.choice(self.actions)

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
        
        raise Exception('Invalid action (not mention or triple)')

    def update_triple(self, triple, label) -> None:
        nx.set_node_attributes(self.sample, {triple: {'label': label}})
        # 更新self.actions
        self.actions.remove(triple)
    
    def update_entity(self, mention, note) -> None:
        entity = self.mention_entity[mention]
        graph = self.linking_graphs[entity]
        # 更新当前mention的note
        nx.set_node_attributes(graph, {mention: {'note': note}})
        # 更新self.actions
        self.actions.remove(mention)
        neighbors = list(graph.neighbors(mention))
        #print('begin update!')
        #print('linking graph situation:', dict(nx.get_node_attributes(graph, 'note')))
        #print('linking graph edge labels:', dict(nx.get_edge_attributes(graph, 'label')))
        for n in neighbors:
            # 看是否能更新边
            #print('1:', graph.nodes[n]['note'])
            #print('2:', graph.edges[(n, mention)])
            if graph.nodes[n]['note'] != None:
                if graph.edges[(n, mention)]['label'] == None:
                    #print('True update!')
                    label = (graph.nodes[n]['note'] == note)
                    attr = {'label': label}
                    nx.set_edge_attributes(graph, {(n, mention): attr})
                    # 如果更新了，再更新self.sample
                    # self.sample中的linking点中mention是要考虑顺序的，都按(min, max)的顺序
                    linking_node = (min(n, mention), max(n, mention))
                    #print('linking node:', linking_node)
                    nx.set_node_attributes(self.sample, {linking_node: attr})
            # 看有label的边能不能更新没有note的别的mention，如果边的label是True则可以更新note
            if graph.edges[(n, mention)] == True:
                if graph.nodes[n]['note'] == None:
                    attr = {'note': note}
                    nx.set_node_attributes(graph, {n: attr})
                    # 更新self.actions
                    self.actions.remove(n)
    
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
                self.update_entity(mention2, note1)
                #linking_graph.nodes[mention2]['note'] = note1 
                #self.actions.remove(mention2)
            elif note2 != None and note1 == None:
                self.update_entity(mention1, note2)
                #linking_graph.nodes[mention1]['note'] = note2
                #self.actions.remove(mention1)
    
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
    
    """ def infer(self) -> None:
        # 循环推理至没有新的可推理处
        if self.if_linking:
            while self.infer_linking():
                pass
        else:
            while self.infer_simple():
                pass """
    
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
        labeled_triples = list(set(triples).difference(set(self.actions)))
        for t in labeled_triples:
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

    """ def check_finish(self) -> bool:
        labels = list(nx.get_node_attributes(self.sample, 'label').values())
        return not (None in labels) """
    
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