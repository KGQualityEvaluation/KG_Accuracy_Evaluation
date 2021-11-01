from samplingMethods import randomSampler, weightedRandomSampler, weightedStratifiedSampler
from annotatorsWithHelper import autoAnnotator, userAnnotator, Helper
from paras import P1, P2
from paras import TRUE_LABEL_DICT, TRUE_NOTE_DICT
from paras import A1, A2, A3, A4
from paras import EPSILON1, EPSILON2
from Timers import Timer, timeCollector

class Framework(object):
    def __init__(self, if_text, if_true_label, if_linking, if_interact, if_infer, if_adjust, sampling_type, select_type, samples) -> None:
        
        if if_text == False and if_linking == True:
            raise Exception('Cannot verify entity linking without texts.')
        
        self.if_text = if_text
        self.if_true_label = if_true_label
        self.if_linking = if_linking
        self.if_interact = if_interact
        self.if_infer = if_infer
        self.if_adjust = if_adjust
        self.sampling_type = sampling_type
        self.select_type = select_type

        if sampling_type == 'random':
            self.sampler = randomSampler(samples, if_linking)
        elif sampling_type == 'weighted':
            self.sampler = weightedRandomSampler(samples, if_linking)
        elif sampling_type == 'wstratified':
            self.sampler = weightedStratifiedSampler(samples, if_linking)
        else:
            raise Exception('Invalid sampling method.')

        self.p1 = P1 
        if if_linking:
            self.p2 = P2
            if select_type == 'MCTS':
                self.create_num = 0
                self.classify_num = 0
        
        if self.if_true_label:
            if self.if_linking:
                self.annotator = autoAnnotator(TRUE_LABEL_DICT, self.if_linking, TRUE_NOTE_DICT)
            else:
                self.annotator = autoAnnotator(TRUE_NOTE_DICT, self.if_linking)
        else:
            self.annotator = userAnnotator()
        
        self.time_collector = timeCollector()
        
        self.total_triple_num = 0
        if self.if_linking:
            self.total_mention_num = 0


    def annotate_one(self, sample, *stra) -> None:

        if self.select_type == 'MCTS':
            if self.if_linking:
                if self.create_num + self.classify_num == 0:
                    create_prob = 0.5
                else:
                    create_prob = self.create_num / (self.create_num + self.classify_num)
                helper = Helper(sample, self.if_linking, self.select_type, self.p1, create_prob)
            else:
                helper = Helper(sample, self.if_linking, self.select_type, self.p1)
        else:
            helper = Helper(sample, self.if_linking, self.select_type)
        

        if self.if_interact:
            self.annotate_one_interact(helper, self.annotator, *stra)
        else:
            self.annotate_one_simple(helper, self.annotator, *stra)

        # 标注结束后更新self.total_triple_num和self.total_mention_num
        self.total_triple_num += helper.get_triple_num()
        if self.if_linking:
            self.total_mention_num += helper.get_mention_num()
    
    # annotate one IG (without interaction)
    def annotate_one_simple(self, helper, annotator, *stra) -> Timer:
        # report situation of IG
        helper.report()

        timer = Timer(A1, A2, A3, A4)
        while not helper.check_finish():
            action = helper.select_one()
            # return: ('triple', triple, (origin, doc_id, sentence)) or ('mention', mention, (doc_id, sentence, entity_linking_graph))

            # t: type, triple or mention
            t = action[0]
            information = action[2]
            if self.if_true_label:
                if t == 'triple':
                    origin = action[2][0]
                    triple = action[1]
                    if self.if_linking:
                        label = annotator.annotate_triple(origin)
                    else:
                        label = annotator.annotate_triple(triple)
                    helper.update_triple(triple, label)
                    # update timer
                    doc_id = information[1]
                    timer.update(t, doc_id)
                else:
                    mention = action[1]
                    note = annotator.annotate_entity(mention)
                    entity = helper.mention_entity[mention]
                    graph = helper.linking_graphs[entity]
                    graph.nodes[mention]['note'] = note
                    helper.update_graph(graph)
                    helper.synch_graph2sample(graph)
                    #helper.update_entity(mention, note)
                    # update timer
                    doc_id = information[0]
                    if_new = helper.if_new_note(mention, note)
                    # 为了估计create_prob
                    if self.select_type == 'MCTS':
                        if if_new:
                            self.create_num += 1
                        else:
                            self.classify_num += 1
                    timer.update(t, doc_id, if_new)
            else:
                if t == 'triple':
                    triple = action[1]
                    label = annotator.annotate_triple(triple, information)
                    helper.update_triple(triple, label)
                    # update timer
                    doc_id = information[1]
                    timer.update(t, doc_id)
                else:
                    mention = action[1]
                    note = annotator.annotate_entity(mention, information)
                    entity = helper.mention_entity[mention]
                    graph = helper.linking_graphs[entity]
                    graph.nodes[mention]['note'] = note
                    helper.update_graph(graph)
                    helper.synch_graph2sample(graph)

                    # update timer 
                    doc_id = information[0]
                    if_new = helper.if_new_note(mention, note)
                    timer.update(t, doc_id, if_new)
                    if self.select_type == 'MCTS':
                        # 为了估计create_prob
                        if if_new:
                            self.create_num += 1
                        else:
                            self.classify_num += 1
            if self.if_infer:
                helper.infer()
        
        if len(stra) == 0:
            self.sampler.collector.set_num(helper.output())
        else:
            self.sampler.collector.set_num(stra[0], helper.output())
        
        self.time_collector.input(timer)

        # report annotation situation & current cost
        helper.report()
        timer.report()


    def annotate_one_interact(self, helper, annotator, *stra) -> Timer:
        pass

    
    def annotate_some(self, sample_list) -> None:
        for sample in sample_list:
            self.annotate_one(sample)

    def update(self, adjust):
        self.sampler.collector.update()
        self.p1 = self.sampler.collector.triple_acc()
        if self.if_linking:
            self.p2 = self.sampler.collector.linking_acc()
        if self.sampling_type == 'wstratified' and self.if_adjust == True and adjust == True:
            self.sampler.adjust()

    def check_end(self):
        if self.if_linking:
            MoE1, MoE2 = self.sampler.collector.MoE()
            if MoE1 <= EPSILON1 and MoE2 <= EPSILON2:
                return True 
            else:
                return False 
        else:
            MoE = self.sampler.collector.MoE()
            if MoE <= EPSILON1:
                return True 
            else:
                return False 

    def report(self):
        # 根据sampler.collector
        # 估计结果的报告
        triple_acc = self.sampler.collector.triple_acc()
        print('=== Report of estimator(s) ===')
        print('Triple accuracy estimate:', triple_acc)

        if self.if_linking:
            linking_acc = self.sampler.collector.linking_acc()
            MoE1, MoE2 = self.sampler.collector.MoE()
            ci1, ci2 = self.sampler.collector.CI()
            print('MoE:', MoE1)
            print('CI:', ci1)
            
            print()
            print('Linking accuracy estimate', linking_acc)
            print('MoE:', MoE2)
            print('CI:', ci2)
        else:
            MoE = self.sampler.collector.MoE()
            ci = self.sampler.collector.CI()
            print('MoE:', MoE)
            print('CI:', ci)
        print()
        
        # 根据time_collector
        # 标注成本&推理效果的报告
        
        # 标注成本
        triple_num, new_entity_num, old_entity_num, bonus_num, total_cost = self.time_collector.output()
        print('=== Report of cost ===')
        if self.sampling_type == 'wstratified':
            print('#IGs:', sum([len(l) for l in self.sampler.collector.triple_accs.values()]))
        else:
            print('#IGs:', len(self.sampler.collector.triple_accs))
        
        print('#Triples waiting for verification:', self.total_triple_num)
        print('#Triples annotated by human:', triple_num)
        print('Inference rate:', 1 - triple_num / self.total_triple_num)

        if self.if_linking:
            print('#Mentions waiting for identification:', self.total_mention_num)
            print('#Mentions identified by human:', new_entity_num + old_entity_num)
            print('Inference rate:', 1 - (new_entity_num + old_entity_num) / self.total_mention_num)

            # 这两个量被维护了两份，实验初期可以验证代码有无错误，验证完就注释掉
            '''
            print()
            print('*** Verification: if the two pairs of numbers equal? ***')
            print(new_entity_num, old_entity_num)
            print(self.create_num, self.classify_num)
            print()
            '''
        
        print('#Continuous texts:', bonus_num)
        print('Total cost:', total_cost)
        if self.if_adjust:
            print('Stratification:', self.sampler.stratification)

    
    def run(self):
        # 要抽的三元组数大于等于20
        # 三元组数：self.sampler.collector.triple_number
        cnt = 0
        
        if self.sampling_type == 'wstratified':
            # 初始化要先从每个层抽一两个IG，这样将来才能利用sampling gain选层
            if not self.if_linking:
                sss = self.sampler.initial_samples(2)
                for ss in sss:
                    stra, s = ss
                    self.annotate_one(s, stra)
                    # 最开始的update应该不调整分层的
                    # 每次从一个层里抽多个样本标注，如果每个样本标完就更新不合适
                    self.update(adjust=False)
            else:
                for stra in self.sampler.stratification:
                    while len(self.sampler.collector.linking_accs[stra]) < 2:
                        s = self.sampler.get_one_stra(stra)
                        self.annotate_one(s, stra)
                        self.update(adjust=False)
            self.report()

        while not self.check_end() or self.sampler.collector.triple_number < 60:
            if self.sampling_type == 'wstratified':
                stra, s = self.sampler.get_one()
                self.annotate_one(s, stra)
                self.update(adjust=True)
            else:
                s = self.sampler.get_one()
                self.annotate_one(s)
                self.update(adjust=False)
            cnt += 1
            if cnt % 3 == 0:
                self.report()
        self.report()

    def log(self):
        triple_num, new_entity_num, old_entity_num, bonus_num, total_cost = self.time_collector.output()
        triple_acc = self.sampler.collector.triple_acc()
        # all triples, annotated by human, inference rate, acc
        l = str(self.total_triple_num) + '\t' + str(triple_num) + '\t' + str(1 - triple_num / self.total_triple_num) + '\t' + str(triple_acc) + '\n'
        return l
        
