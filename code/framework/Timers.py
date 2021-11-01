class Timer(object):
    def __init__(self, a1, a2, a3, a4) -> None:
        
        self.triple_num = 0
        self.new_entity_num = 0
        self.old_entity_num = 0
        self.bonus_num = 0
        self.last_doc = None 
        self.a1, self.a2, self.a3, self.a4 = a1, a2, a3, a4
        
    
    def update(self, t, doc_id, *if_new) -> None:
        if t == 'triple':
            self.update_triple()
        elif t == 'mention':
            self.update_entity(if_new)
        else:
            raise Exception('Invalid type (triple or entity).')
        
        self.update_bonus(doc_id)
    
    def update_triple(self) -> None:
        self.triple_num += 1
    
    def update_entity(self, if_new) -> None:
        if if_new:
            self.new_entity_num += 1
        else:
            self.old_entity_num += 1
    
    def update_bonus(self, doc_id) -> None:
        if doc_id == self.last_doc:
            self.bonus_num += 1
        self.last_doc = doc_id 
    
    def total_cost(self) -> float:
        return self.triple_num * self.a1 + self.new_entity_num * self.a2 + self.old_entity_num * self.a3 - self.bonus_num * self.a4

    def report(self) -> None:
        print('=== Time report of current IG annotation ===')
        print('Annotated triples by human:', self.triple_num)
        print('Annotated entities by human:', self.new_entity_num + self.old_entity_num)
        print('Bonus number:', self.bonus_num)
        print('Total cost:', self.total_cost())
        print()
    
    def output(self) -> tuple:
        return (self.triple_num, self.new_entity_num, self.old_entity_num, self.bonus_num, self.total_cost())


class timeCollector(object):
    def __init__(self) -> None:
        self.triple_num = 0
        self.new_entity_num = 0
        self.old_entity_num = 0
        self.bonus_num = 0
        self.total_cost = 0
    
    def input(self, timer) -> None:
        self.triple_num += timer.triple_num
        self.new_entity_num += timer.new_entity_num
        self.old_entity_num += timer.old_entity_num
        self.bonus_num += timer.bonus_num
        self.total_cost += timer.total_cost() 
    
    def output(self) -> tuple:
        return (self.triple_num, self.new_entity_num, self.old_entity_num, self.bonus_num, self.total_cost)