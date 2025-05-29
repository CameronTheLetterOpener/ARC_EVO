import numpy as np

def filter_overspecified(self, ruleset):
    #If a rule starts to produce incorrect outputs when the grid is expanded, it is overspecified
    #For each training grid, expand it by 4 pixels in each direction
    #Then apply each rule to the expanded grid and see if it remains correct
    #If it does not, remove the rule from the ruleset
    ios = [io for io in self.task.train]
    for i, io in enumerate(ios):
        #Expand the grid by 4 pixels in each direction
        expanded_grid = np.zeros((io[0].shape[0]+8, io[0].shape[1]+8))
        expanded_grid[4:-4, 4:-4] = io[0]
    
    #apply each rule to each grid
    new_ruleset = []
    for rule in ruleset:
        incorrect = False
        rule_copy = rule.make_copy()
        for i, io in enumerate(ios):
            #Apply the rule to the expanded grid
            out, out_cor = rule_copy.apply_to_io(io, i)
            #Check if the output is correct
            if np.any(np.logical_and(out != -1, out != io[1])):
                #If it is not correct, remove the rule from the ruleset
                incorrect = True
                break
        if not incorrect:
            #If the rule is correct, add it to the new ruleset
            new_ruleset.append(rule_copy)
    return new_ruleset

def filter_subsumption(self, ruleset, v=False):
    #Make a copy of the training grids to keep track of which outputs are satisfied.
    out_cors = [np.zeros_like(test_io[1]) for test_io in self.task.train]
    out_io = [test_io[1] for test_io in self.task.train]
    #Find the top scoring rules for each colour
    #Then add these rules one by one to the compact rulset and mark off the outputs they satisfy
    #If a rule is added that satisfies all the outputs of a previous rule, it is subsumed and not added

    #Sort the rules primarily by prop_correct (large to small) and secondarily by length (small to large)
    ruleset = sorted(ruleset, key=lambda x: (x.prop_correct, x.correct, -len(x.ref_cols)), reverse=True)    
    rule_list = []
    for rule in ruleset:
        added = []
        for k, (io, mask) in enumerate(zip(self.task.train, self.correct_masks)):
            out, rule_out_cor = rule.apply_to_io(io, k)
            #Are there elements of out of the out_col that are not already true in out_cors. This is a one directional check
            added.append(np.any(np.logical_and(rule_out_cor, np.logical_not(out_cors[k]))))
        if np.any(added):
            rule_list.append(rule)
            for k, (io, mask) in enumerate(zip(self.task.train, self.correct_masks)):
                out, rule_out_cor = rule.apply_to_io(io, k)
                out_cors[k] = np.logical_or(out_cors[k], rule_out_cor)
        if v:
            print(str(rule))
    
    return rule_list

def refine_ruleset(self, ruleset):
    new_rules = []
    for rule in ruleset:
        new_rule = self.refine_rule(rule)
        if new_rule:
            new_rules.extend(new_rule)
    ruleset.extend(new_rules)
    return ruleset

def refine_rule(self, rule):
    #Remove one ref col at a time and see if it degrades the performance
    #If it does not, remove it
    new_rules = []
    if len(rule.ref_cols) > 1:
        for i in range(len(rule.ref_cols)):
            rule_copy = rule.make_copy()
            rule_copy.ref_cols.pop(i)
            rule_copy.angles.pop(i)
            rule_copy.distances.pop(i)
            self.score_rule(rule_copy)
            if rule_copy.prop_correct >= rule.prop_correct:
                new_rules.append(rule)
    return new_rules

