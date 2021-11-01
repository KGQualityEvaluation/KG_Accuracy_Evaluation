from os import link
from paras import INITIAL_STRATIFICATION
import random 
from networkx import nx
import numpy as np  
from scipy import stats as st
from paras import EPSILON1, EPSILON2, ALPHA1, ALPHA2
from scipy.special import comb
import copy 

# 这里的random指的是不加权抽样的版本，最简单的
class randomCollector(object):
    def __init__(self, if_linking, M1, N1, *para) -> None:
        self.triple_nums = []
        self.triple_estimator = None
        self.triple_var = float('inf')
        self.M1 = M1  # 三元组总数
        #print('M1', self.M1)
        self.N1 = N1  # IG总数
        #print('N1', self.N1)
        self.if_linking = if_linking
        if if_linking:
            self.linking_nums = []
            self.linking_estimator = None
            self.linking_var = float('inf')
            self.M2 = para[0]
    
    # 真triple数
    def get_triple_num(self, sample) -> int:
        labels = list(nx.get_node_attributes(sample, 'label').values())
        return sum(labels)
    
    # 真linking数
    def get_linking_num(self, linking_graphs) -> int:
        num = 0
        for graph in linking_graphs:
            labels = list(nx.get_edge_attributes(graph, 'label').values())
            num += sum(labels)
        return num 

    def set_num(self, helper_output) -> None:
        sample = helper_output[0]
        self.triple_nums.append(self.N1 / self.M1 * self.get_triple_num(sample))
        if self.if_linking:
            linking_graphs = helper_output[1]
            self.linking_nums.append(self.N1 / self.M2 * self.get_linking_num(linking_graphs))
    
    def update_triple(self) -> None:
        self.triple_estimator = np.mean(self.triple_nums)
        if len(self.triple_nums) > 1:
            self.triple_var = np.var(self.triple_nums, ddof=1) / len(self.triple_nums)
        else:
            self.triple_var = float('inf')

    def update_linking(self) -> None:
        self.linking_estimator = np.mean(self.linking_nums)
        if len(self.linking_var) > 1:
            self.linking_var = np.var(self.linking_nums, ddof=1) / len(self.linking_nums)
        else:
            self.linking_var = float('inf')

    def update(self) -> None:
        self.update_triple()
        if self.if_linking:
            self.update_linking()

    def MoE(self) -> tuple:
        z1 = st.norm.isf(ALPHA1 / 2)
        MoE1 = z1 * (self.triple_var ** 0.5)
        if not self.if_linking:
            return MoE1
        else:
            z2 = st.norm.isf(ALPHA2 / 2)
            MoE2 = z2 * (self.linking_var ** 0.5) 
            return (MoE1, MoE2)
    
    def CI(self) -> tuple:
        if self.if_linking:
            MoE1, MoE2 = self.MoE()
        else:
            MoE1 = self.MoE()
        ci1 = (self.triple_estimator - MoE1, self.triple_estimator + MoE1)
        if not self.if_linking:
            return ci1
        else:
            ci2 = (self.linking_estimator - MoE2, self.linking_estimator + MoE2)
            return (ci1, ci2)
    
    def triple_acc(self) -> float:
        return self.triple_estimator
    
    def linking_acc(self) -> float:
        return self.linking_estimator


class randomSampler(object):
    def __init__(self, samples, if_linking) -> None:
        self.samples = samples 
        self.if_linking = if_linking
        
        self.M1, self.N1, self.N2 = 0, 0, 0
        self.para_init()
        
        if self.if_linking:
            self.collector = randomCollector(if_linking, self.M1, self.N1, self.M2)
        else:
            self.collector = randomCollector(if_linking, self.M1, self.N1)
     
    def linking_num(self, sample) -> int:
        triples = list(nx.get_node_attributes(sample, 'triple').keys())
        origin_dict = nx.get_node_attributes(sample, 'origin')
        entity_mention_dict = dict()
        for triple in triples:
            origin = origin_dict[triple]
            e1, _, e2 = triple
            m1, _, m2 = origin
            if e1 not in entity_mention_dict:
                entity_mention_dict[e1] = []
            if e2 not in entity_mention_dict:
                entity_mention_dict[e2] = []
            entity_mention_dict[e1].append(m1)
            entity_mention_dict[e2].append(m2)
        
        cnt = 0
        for k, v in entity_mention_dict.enumerate():
            num = comb(len(set(v)), 2)
            cnt += num 

        return cnt 
    
    # 计算self.M1（总三元组数）,self.N1（总IG数）,self.N2（总实体链接对数）
    def para_init(self):
        self.N1 = len(self.samples)
        for sample in self.samples:
            self.M1 += len(nx.get_node_attributes(sample, 'triple').keys())
            if self.if_linking:
                self.N2 += self.linking_num(sample)
    
    def get_one(self) -> nx.DiGraph:
        s = random.choice(self.samples)
        self.samples.remove(s)
        return s
    
    def get_some(self, m) -> list:
        some = []
        for i in range(m):
            some.append(self.get_one())
        return some 
    
    def no_more(self) -> bool:
        return len(self.samples) == 0


# HH estimator，加权抽样
class weightedRandomCollector(object):
    def __init__(self, if_linking) -> None:
        self.triple_accs = []
        self.triple_estimator = None
        self.triple_var = float('inf')
        #  # 三元组总数
        #print('M1', self.M1)
        #self.N1 = N1  # IG总数
        #print('N1', self.N1)
        self.if_linking = if_linking
        if if_linking:
            self.linking_accs = []
            self.linking_estimator = None
            self.linking_var = float('inf')
            #self.M2 = para[0]
        # 记录已经标注过的所有三元组数量
        self.triple_number = 0
    
    def get_triple_acc(self, sample) -> int:
        labels = list(nx.get_node_attributes(sample, 'label').values())
        #print(sum(labels) / len(labels))
        self.triple_number += len(labels)
        return sum(labels) / len(labels)
    
    def get_linking_acc(self, linking_graphs) -> int:
        total_num = 0
        true_num = 0
        for graph in linking_graphs:
            labels = list(nx.get_edge_attributes(graph, 'label').values())
            total_num += len(labels)
            true_num += sum(labels)
        return true_num / total_num

    def set_num(self, helper_output) -> None:
        sample = helper_output[0]
        self.triple_accs.append(self.get_triple_acc(sample))
        if self.if_linking:
            linking_graphs = helper_output[1]
            self.linking_accs.append(self.get_linking_acc(linking_graphs))
    
    def update_triple(self) -> None:
        self.triple_estimator = np.mean(self.triple_accs)
        if len(self.triple_accs) > 1:
            self.triple_var = np.var(self.triple_accs, ddof=1) / len(self.triple_accs)
        else:
            self.triple_var = float('inf')

    def update_linking(self) -> None:
        self.linking_estimator = np.mean(self.linking_accs)
        if len(self.linking_accs) > 1:
            self.linking_var = np.var(self.linking_accs, ddof=1) / len(self.linking_accs)
        else:
            self.linking_var = float('inf')

    def update(self) -> None:
        self.update_triple()
        if self.if_linking:
            self.update_linking()

    def MoE(self) -> tuple:
        z1 = st.norm.isf(ALPHA1 / 2)
        MoE1 = z1 * (self.triple_var ** 0.5)
        if not self.if_linking:
            return MoE1
        else:
            z2 = st.norm.isf(ALPHA2 / 2)
            MoE2 = z2 * (self.linking_var ** 0.5) 
            return (MoE1, MoE2)
    
    def CI(self) -> tuple:
        if self.if_linking:
            MoE1, MoE2 = self.MoE()
        else:
            MoE1 = self.MoE()
        ci1 = (self.triple_estimator - MoE1, self.triple_estimator + MoE1)
        if not self.if_linking:
            return ci1
        else:
            ci2 = (self.linking_estimator - MoE2, self.linking_estimator + MoE2)
            return (ci1, ci2)
    
    def triple_acc(self) -> float:
        return self.triple_estimator
    
    def linking_acc(self) -> float:
        return self.linking_estimator


class weightedRandomSampler(object):
    def __init__(self, samples, if_linking) -> None:
        self.samples = samples 
        self.if_linking = if_linking
        
        self.M1 = 0
        self.weights = []
        self.para_init()
        
        self.collector = weightedRandomCollector(if_linking)

    
    # 计算self.M1（总三元组数）,self.weights
    def para_init(self):
        self.N1 = len(self.samples)
        for sample in self.samples:
            sz = len(nx.get_node_attributes(sample, 'triple').keys())
            self.M1 += sz
            self.weights.append(sz)
        self.weights = list(np.divide(self.weights, self.M1))
    
    def get_one(self) -> nx.DiGraph:
        # 加权随机抽样
        # 抽index，然后找到对应sample，用index从self.samples和self.weights里移除对应元素
        idx = random.choices(population=list(range(len(self.samples))), weights=self.weights, k=1)[0]
        s = self.samples[idx]
        # 改成有放回其实就是把下面两个del删了
        #del self.samples[idx]
        #del self.weights[idx]
        return s
    
    def get_some(self, m) -> list:
        some = []
        for i in range(m):
            some.append(self.get_one())
        return some 
    
    def no_more(self) -> bool:
        return len(self.samples) == 0


# 加权分层抽样       
class weightedStratifiedSampler(object):
    def __init__(self, samples, if_linking) -> None:
        self.if_linking = if_linking
        self.stratification = INITIAL_STRATIFICATION

        self.samples = dict()
        self.sizes = dict()
        self.samples_init(samples)

        self.weights = self.get_W()

        self.collector = weightedStratifiedCollector(if_linking, self.stratification, self.weights)
    
    # 得到每一层的权重Wh
    def get_W(self) -> dict():
        weights_list = []
        for s in self.stratification:
            weights_list.append(sum(self.sizes[s]))
        print(weights_list)
        weights_list = list(np.divide(weights_list, sum(weights_list)))
        weights_dict = dict(zip(self.stratification, weights_list))
        return weights_dict

    def confidence(self, sample) -> float:
        cons = list(nx.get_node_attributes(sample, 'confidence_score').values())
        #return np.mean(cons)
        return np.median(cons)

    def initial_samples(self, m):
        sss = []
        for stra in self.stratification:
            for i in range(m):
                s = self.get_one_stra(stra)
                sss.append((stra, s))
        return sss
    
    def which_stra(self, sample) -> tuple:
        con = self.confidence(sample)
        for stra in self.stratification:
            if con >= stra[0] and con < stra[1]:
                return stra 
    
    # 根据samples和初始分层，把样本变成每个层对应的
    def samples_init(self, samples) -> None:
        for stra in self.stratification:
            self.samples[stra] = []
            self.sizes[stra] = []
        for sample in samples:
            stra = self.which_stra(sample)
            self.samples[stra].append(sample)
            self.sizes[stra].append(len(nx.get_node_attributes(sample, 'triple')))
    
    def get_one_stra(self, stra):
        stra_samples = self.samples[stra]
        if len(stra_samples) == 1:
            return stra_samples[0]
        
        sizes = self.sizes[stra]
        # 这里的weights是在一层中根据size加权抽样的权重
        weights = list(np.divide(sizes, sum(sizes)))
        
        s = random.choices(population=stra_samples, weights=weights, k=1)[0]
        return s
    
    def get_some_stra(self, stra, m) -> list:
        some = []
        for i in range(m):
            some.append(self.get_one_stra(stra))
        return some 

    def get_one(self) -> tuple:
        stra = self.collector.select_stra()
        return (stra, self.get_one_stra(stra))
    
    def get_some(self, m):
        stra = self.collector.select_stra()
        return self.get_some_stra(stra, m)
    
    def adjust(self) -> bool:
        result = self.select_stras()
        if result == False:
            return False 
        else:
            stra1, stra2 = result 
            self.combine(stra1, stra2)
            return True 

    # 如果能调整的话就返回包含两个待合并层的元组，不能就返回False
    def select_stras(self):
        if len(self.stratification) <= 3:
            print('Only three stratum. Adujstment is not necessary.')
            return False
        score_dict = dict()
        for i in range(len(self.stratification) - 1):
            stra1 = self.stratification[i]
            stra2 = self.stratification[i+1]
            score = self.collector.simulate_var(stra1, stra2)
            #print(score)
            #print(self.stratification)
            if score > 0:
                score_dict[(stra1, stra2)] = score
        if len(score_dict.keys()) == 0:
            print('No better adjustment in this round. Adujstment is not necessary.')
            return False
        best = max(score_dict, key=score_dict.get)
        print('Adjust:', best)
        return best

    # 更新sampler，也更新sampler.collector
    def combine(self, stra1, stra2):
        new_stra = (stra1[0], stra2[1])
        # stratification
        idx = self.stratification.index(stra1)
        self.stratification[idx] = new_stra
        self.stratification.remove(stra2)
        # weights
        self.weights[new_stra] = self.weights[stra1] + self.weights[stra2]
        del self.weights[stra1]
        del self.weights[stra2]
        # samples
        self.samples[new_stra] = self.samples[stra1] + self.samples[stra2]
        del self.samples[stra1]
        del self.samples[stra2]
        # sizes
        self.sizes[new_stra] = self.sizes[stra1] + self.sizes[stra2]
        del self.sizes[stra1]
        del self.sizes[stra2]

        # self.collector
        self.collector.combine(stra1, stra2)

        

class weightedStratifiedCollector(object):
    def __init__(self, if_linking, stratification, weights) -> None:
        self.stratification = stratification
        self.weights = weights
        self.if_linking = if_linking
        
        self.triple_accs = dict(zip(self.stratification, [[] for i in range(len(self.stratification))]))
        self.triple_estimators = dict(zip(self.stratification, [None for i in range(len(self.stratification))]))
        self.triple_vars = dict(zip(self.stratification, [float('inf') for i in range(len(self.stratification))]))
        self.triple_estimator = None 
        self.triple_var = float('inf')

        if self.if_linking:
            self.linking_accs = dict(zip(self.stratification, [[] for i in range(len(self.stratification))]))
            self.linking_estimators = dict(zip(self.stratification, [None for i in range(len(self.stratification))]))
            self.linking_vars = dict(zip(self.stratification, [float('inf') for i in range(len(self.stratification))]))
            self.linking_estimator = None 
            self.linking_var = float('inf')
            self.linking_number = 0
        
        self.triple_number = 0
    
    # weights为dict
    def sampling_gain(self, stra) -> float:
        stra_var = self.triple_vars[stra]
        first = self.weights[stra] * stra_var
        tmp = 0
        for s in self.stratification:
            tmp += self.weights[s] * self.triple_vars[s]
        first /= tmp 
        second = len(self.triple_accs[stra])
        tmp = 0
        for s in self.stratification:
            tmp += len(self.triple_accs[s])
        second /= tmp 

        return first - second 

    def select_stra(self) -> tuple:
        sgs = dict()
        for stra in self.stratification:
            sg = self.sampling_gain(stra)
            sgs[stra] = sg
        
        return max(sgs, key=sgs.get)

    def get_triple_acc(self, sample) -> int:
        labels = list(nx.get_node_attributes(sample, 'label').values())
        self.triple_number += len(labels)
        return sum(labels) / len(labels)
    
    def get_linking_acc(self, linking_graphs) -> int:
        total_num = 0
        true_num = 0
        for graph in linking_graphs.values():
            #print(graph)
            labels = list(nx.get_edge_attributes(graph, 'label').values())
            total_num += len(labels)
            true_num += sum(labels)
        self.linking_number += total_num
        if total_num == 0:
            return None
        return true_num / total_num
    
    def set_num(self, stra, helper_output):
        sample = helper_output[0]
        self.triple_accs[stra].append(self.get_triple_acc(sample))
        if self.if_linking:
            linking_graphs = helper_output[1]
            #print('linking:', linking_graphs)
            acc = self.get_linking_acc(linking_graphs)
            if acc != None:
                self.linking_accs[stra].append(acc)
                #print(self.linking_accs)
            #else:
            #    self.linking_accs[stra].append(0)

    def stra_triple_estimator(self, stra):
        accs = self.triple_accs[stra]
        return np.mean(accs)
    
    def stra_triple_var(self, stra):
        accs = self.triple_accs[stra]
        if len(accs) <= 1:
            return float('inf')
        v = np.var(accs, ddof=1) / len(accs)
        return v

    def stra_linking_estimator(self, stra):
        accs = self.linking_accs[stra]
        return np.mean(accs)

    def stra_linking_var(self, stra):
        accs = self.linking_accs[stra]
        if len(accs) <= 1:
            return float('inf')
        v = np.var(accs, ddof=1) / len(accs)
        return v

    def update_triple(self):
        self.triple_estimators = dict(zip(self.stratification, [None for i in range(len(self.stratification))]))
        self.triple_vars = dict(zip(self.stratification, [float('inf') for i in range(len(self.stratification))]))

        for stra in self.stratification:
            self.triple_estimators[stra] = self.stra_triple_estimator(stra)
            self.triple_vars[stra] = self.stra_triple_var(stra)
                    
        self.triple_estimator = sum([self.triple_estimators[s] * self.weights[s] for s in self.stratification])

        if self.triple_number <= 1:
            self.triple_var = float('inf')
        else:
            self.triple_var = sum([self.weights[s] ** 2 * self.triple_vars[s] for s in self.stratification])
            #self.triple_var /= (self.triple_number * (self.triple_number - 1))

    def update_linking(self):
        self.linking_estimators = dict(zip(self.stratification, [None for i in range(len(self.stratification))]))
        self.linking_vars = dict(zip(self.stratification, [float('inf') for i in range(len(self.stratification))]))

        for stra in self.stratification:
            self.linking_estimators[stra] = self.stra_linking_estimator(stra)
            self.linking_vars[stra] = self.stra_linking_var(stra)
        
        #print(self.linking_estimators)
        #print(self.weights)
        self.linking_estimator = sum([self.linking_estimators[s] * self.weights[s] for s in self.stratification])

        if self.linking_number <= 1:
            self.linking_var = float('inf')
        else:
            self.linking_var = sum([self.weights[s] ** 2 * self.linking_vars[s] for s in self.stratification])
            #self.linking_var /= (self.linking_number * (self.linking_number - 1))

    def update(self):
        self.update_triple()
        if self.if_linking:
            self.update_linking()
    
    def MoE(self) -> tuple:
        z1 = st.norm.isf(ALPHA1 / 2)
        MoE1 = z1 * (self.triple_var ** 0.5)
        if not self.if_linking:
            return MoE1
        else:
            z2 = st.norm.isf(ALPHA2 / 2)
            MoE2 = z2 * (self.linking_var ** 0.5) 
            return (MoE1, MoE2)
    
    def CI(self) -> tuple:
        if self.if_linking:
            MoE1, MoE2 = self.MoE()
        else:
            MoE1 = self.MoE()
        ci1 = (self.triple_estimator - MoE1, self.triple_estimator + MoE1)
        if not self.if_linking:
            return ci1
        else:
            #print('estimator:', self.linking_estimator)
            #print('MoE:', MoE2)
            if self.linking_estimator != None:
                ci2 = (self.linking_estimator - MoE2, self.linking_estimator + MoE2)
                return (ci1, ci2)
            else:
                return (ci1, None)

    def triple_acc(self):
        return self.triple_estimator
    
    def linking_acc(self):
        return self.linking_estimator

    def simulate_var(self, stra1, stra2) -> float:
        new_stra = (stra1[0], stra2[1])
        new_triple_accs = copy.deepcopy(self.triple_accs)
        new_triple_accs[new_stra] = self.triple_accs[stra1] + self.triple_accs[stra2]
        del new_triple_accs[stra1]
        del new_triple_accs[stra2]

        new_weights = copy.deepcopy(self.weights)
        new_weights[new_stra] = self.weights[stra1] + self.weights[stra2]
        del new_weights[stra1]
        del new_weights[stra2]

        def stra_var(s):
            accs = new_triple_accs[s]
            if len(accs) <= 1:
                return float('inf')
            v = np.var(accs, ddof=1) / len(accs)
            return v
        
        t_vars = []
        for s in new_triple_accs.keys():
            t_vars.append(stra_var(s))
        ws = list(new_weights.values())
        
        new_var = sum([ws[i] ** 2 * t_vars[i] for i in range(len(ws))])

        return self.triple_var - new_var 

    def combine(self, stra1, stra2):
        new_stra = (stra1[0], stra2[1])
        # 因为是浅拷贝，所以sampler和collector的stratification和weights是自动同步的，不用改！
        """ # stratification
        print(self.stratification)
        idx = self.stratification.index(stra1)
        self.stratification[idx] = new_stra
        self.stratification.remove(stra2)
        # weights
        self.weights[new_stra] = self.weights[stra1] + self.weights[stra2]
        del self.weights[stra1]
        del self.weights[stra2] """
        # triple_accs
        self.triple_accs[new_stra] = self.triple_accs[stra1] + self.triple_accs[stra2]
        del self.triple_accs[stra1]
        del self.triple_accs[stra2]
        
        if self.if_linking:
            # linking_accs
            self.linking_accs[new_stra] = self.linking_accs[stra1] + self.linking_accs[stra2]
            del self.linking_accs[stra1]
            del self.linking_accs[stra2]
        
        # triple_estimators, triple_estimator, triple_vars, triple_var, linking_estimators, linking_estimator, linking_vars, linking_var
        self.update()
