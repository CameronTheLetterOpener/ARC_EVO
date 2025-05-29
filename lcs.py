import numpy as np
import random
import multiprocessing
import copy
import pickle

class LCS_ARC:
    def __init__(self, task, allow_background_in_outcol=False):
        self.rules = []
        self.perfect_rules = []
        self.saved_perfect_rules = []
        self.task = task
        self.correct_masks = [np.zeros_like(io[1]) for io in self.task.train]
        self.train_col = self.count_training_colours()
        self.background = 5
        self.allow_background = allow_background_in_outcol
        
    def count_training_colours(self):
            colours = [{}, {}]
            for io in self.task.train:
                for j in range(len(io)):
                    for i in io[j].flatten():
                        i = int(i)
                        if i not in colours:
                            colours[j][i] = 1
                        else:
                            colours[j][i] += 1
            return colours
    
    #Create a custom iterator for the ruleset, that yields rules in order, but prioritises rule.level, from lowest to highest
    def iter_ruleset(self):
        for rule in sorted(self.rules, key=lambda x: x.level):
            yield rule
    
    def generate_rule(self, io, max_ref_length=4, max_out_length=1):
        #Generate random states for the rule
        rule_length = random.randint(1, max_ref_length)
        ref_cols = []
        ref_xs = []
        ref_ys = []
        for i in range(rule_length):
            ref_col = random.choice(list(self.train_col[0].keys()))
            ref_cols.append(ref_col)
            ref_x = random.randint(0, io[0].shape[1]) - io[0].shape[1]//2
            ref_y = random.randint(0, io[0].shape[0]) - io[0].shape[0]//2
            ref_xs.append(ref_x)
            ref_ys.append(ref_y)
        cols = list(self.train_col[1].keys())
        if not self.allow_background and self.background in cols:
            cols.remove(self.background)
        out_col_length = random.randint(1, max_out_length)
        out_cols = []
        out_xs = []
        out_ys = []
        #First entry is the central pixel
        out_col = random.choice(cols)
        out_cols.append(out_col)
        out_x = random.randint(0, io[0].shape[1]) - io[0].shape[1]//2
        out_y = random.randint(0, io[0].shape[0]) - io[0].shape[0]//2
        out_xs.append(out_x)
        out_ys.append(out_y)
        for i in range(1, out_col_length):
            out_col = random.choice(cols)
            out_cols.append(out_col)
            out_x = random.randint(0, io[0].shape[1]) - io[0].shape[1]//2
            out_y = random.randint(0, io[0].shape[0]) - io[0].shape[0]//2
            out_xs.append(out_x)
            out_ys.append(out_y)
        return MultiRule.from_coords(ref_cols, out_cols, ref_xs, ref_ys, out_xs, out_ys, self.task)

    def xy_pairs_from_heatmap(self, heatmap):
        #Get the x and y coordinates of the heatmap
        coords = np.argwhere(heatmap > 5)
        xs = coords[:, 1]
        ys = coords[:, 0]
        return xs, ys
    
    def get_changed_out_pos(self, io):
        #Get the x and y coordinates of the changed output positions
        coords = np.argwhere(io[1] != io[0])
        #Then randomly select one of these coordinates
        coords = coords[random.randint(0, len(coords)-1)]
        xs = coords[1]
        ys = coords[0]
        #Get the colour of the pixel
        col = io[1][ys][xs]
        return xs, ys, col

    def get_close_in_pos(self, io, out_x, out_y):
        #Return a random input pos biased to be close to the output
        distribution = np.zeros((io[0].shape[0], io[0].shape[1]))
        for i in range(io[0].shape[0]):
            for j in range(io[0].shape[1]):
                distribution[i][j] = np.sqrt((i-out_y)**2 + (j-out_x)**2)
        distribution = np.exp(-distribution)
        distribution = distribution/np.sum(distribution)
        #Select a random pixel from the distribution
        idx = np.random.choice(np.arange(io[0].shape[0]*io[0].shape[1]), p=distribution.flatten())
        ref_x = idx % io[0].shape[1]
        ref_y = idx // io[0].shape[1]
        #Get the colour of the pixel
        col = io[0][ref_y][ref_x]
        return ref_x, ref_y, col

    def generate_rule_from_grid(self, io, max_ref_length=4, max_out_length=1, heatmap=None, incorrect_coords=[]):
        #Generate a rule chosen from the io grids
        ref_length = random.randint(1, max_ref_length)
        out_length = random.randint(1, max_out_length)
        ref_cols = []
        ref_xs = []
        ref_ys = []
        out_cols = []
        out_xs = []
        out_ys = []
        if(heatmap is not None):
            xs, ys = self.xy_pairs_from_heatmap(heatmap[1])
        for i in range(out_length):
            if(heatmap is not None):
                #Select a random pixel from the heatmap
                idx = random.randint(0, len(xs)-1)
                out_x = xs[idx]
                out_y = ys[idx]
            else:
                #Select a random pixel from the output grid
                if(incorrect_coords != [] and len(incorrect_coords[0][0]) != 0 and random.random() < 0.75 and i == 0):
                    idx = random.randint(0, len(incorrect_coords[0][0])-1)
                    out_x = incorrect_coords[0][1][idx]
                    out_y = incorrect_coords[0][0][idx]
                else:
                    #Get another output pixel if self.allow_background is false
                    out_x, out_y, col = self.get_changed_out_pos(io)
                    while not self.allow_background and col == self.background:
                        out_x, out_y, col = self.get_changed_out_pos(io)
            out_xs.append(out_x)
            out_ys.append(out_y)
            out_cols.append(io[1][out_y][out_x])
        if(heatmap is not None):
            xs, ys = self.xy_pairs_from_heatmap(heatmap[0])
        #The first entry always has to be the central pixel
        ref_xs.append(out_xs[0])
        ref_ys.append(out_ys[0])
        ref_cols.append(io[0][out_ys[0]][out_xs[0]])
        
        for i in range(ref_length-1):
            if(heatmap is not None):
                #Select a random pixel from the heatmap
                idx = random.randint(0, len(xs)-1)
                ref_x = xs[idx]
                ref_y = ys[idx]
            else:
                #Select a random pixel from the input grid or one of the incorrect coords
                ref_x, ref_y, col = self.get_close_in_pos(io, out_xs[0], out_ys[0])
            ref_xs.append(ref_x)
            ref_ys.append(ref_y)
            #Get the colour of the pixel
            ref_col = io[0][ref_y][ref_x]
            ref_cols.append(ref_col)
        rule = MultiRule.from_coords(ref_cols, out_cols, ref_xs, ref_ys, out_xs, out_ys, self.task)
        return rule

    def apply_self(self, rulesets, io, i, grid=None):
        #Does this need to be able to take in a pre-existing output grid?
        io_copy = (io[0].copy(), io[1].copy())
        if grid is None:
            grid_copy = np.ones_like(io[1])*-1
        else:
            grid_copy = grid.copy()
            io_copy = (grid_copy, io[1])
        for ruleset in rulesets:
            for rule in ruleset:
                out, our_cor = rule.apply_to_io(io_copy, i)
                idxs = np.where(out != -1)
                grid_copy[idxs] = out[idxs]
            io_copy = (grid_copy, io[1])
            if(not self.allow_background):
                grid_copy[grid_copy == -1] = self.background
        return grid_copy
    
    #The 2 -> 2 rule should be showing up in the perfect rules
    def generate_base_rules(self):
        #Generate rules that take each of the colours in the puzzle and output them with 0 distance and angle
        rules = []
        for col in self.train_col[0].keys():
            rule = MultiRule([col], [col], [0], [0], [0], [0], self.task)
            rules.append(rule)
        return rules

    def generate_initial_population(self, n_rules=100, background=False, max_ref_length=4, max_out_length=1, heatmaps=None):
        self.rules.extend(self.generate_base_rules())
        for i in range(n_rules):
            k = random.randint(0, len(self.task.train)-1)
            if(heatmaps is None):
                self.rules.append(self.generate_rule_from_grid(self.task.train[k], max_ref_length=max_ref_length, max_out_length=max_out_length))
            else:
                self.rules.append(self.generate_rule_from_grid(self.task.train[k], max_ref_length=max_ref_length, max_out_length=max_out_length, heatmap=heatmaps[k]))

    def score_rule(self, rule):
        rule.correct = 0
        rule.incorrect = 0
        rule.prop_correct = 0
        for i, (io, mask) in enumerate(zip(self.task.train, self.correct_masks)):
            out, out_cor = rule.apply_to_io(io, i, score=True)
        if(rule.correct + rule.incorrect != 0):
            rule.prop_correct = rule.correct/(rule.correct + rule.incorrect)

    def evaluate_ruleset(self, ruleset):
        for rule in ruleset:
            self.score_rule(rule)

    #Rewrite the above for multiprocessing
    def evaluate_rule(self, rule):
        self.score_rule(rule)
        return rule

    def evaluate_ruleset_parallel(self, ruleset):
        with multiprocessing.Pool(16) as p:
            ruleset = p.map(self.evaluate_rule, ruleset)
        return ruleset

    def save_perfect_rules(self):
        i = 0
        for rule in self.rules:
            if(rule.prop_correct == 1):
            # if rule.correct > 1 and rule.incorrect == 0:
                self.perfect_rules.append(rule)
                i += 1
        print('Number of perfect rules:', i)

    def sort_ruleset(self, ruleset):
        #Sort the ruleset by prop_correct, then by length
        ruleset = sorted(ruleset, key=lambda x: (x.prop_correct, x.correct, -len(x.ref_cols)), reverse=True)
        return ruleset

    def tournament_selection(self, n=1000):
        return random.sample(self.rules, n)
    
    def crossover(self, parent1, parent2):
        return parent1.crossover(parent2)
    
    def remove_duplicates(self, ruleset):
        #Remove rules that are the same
        new_ruleset = []
        for rule in ruleset:
            if rule not in new_ruleset:
                new_ruleset.append(rule)
        return new_ruleset
    
    def mutate(self, rule):
        #Mutate a rule in place
        #Add a new ref
        if(random.random() < 0.1):
            ref_col = random.choice(list(self.train_col[0].keys()))
            ref_x = random.randint(0, self.task.train[0][0].shape[1]) - self.task.train[0][0].shape[1]//2
            ref_y = random.randint(0, self.task.train[0][0].shape[0]) - self.task.train[0][0].shape[0]//2
            rule.ref_cols.append(ref_col)
            #Calculate the angle and distance
            angle = np.arctan2(ref_y, ref_x)
            distance = np.sqrt((0 - ref_x)**2 + (0 - ref_y)**2)
            rule.angles.append(angle)
            rule.distances.append(distance)
        #Remove a ref
        if(random.random() < 0.1):
            if(len(rule.ref_cols) > 1):
                idx = random.randint(0, len(rule.ref_cols)-1)
                rule.ref_cols.pop(idx)
                rule.angles.pop(idx)
                rule.distances.pop(idx)
        #Mutate an out
        if(random.random() < 0.1):
            cols = list(self.train_col[1].keys())
            if not self.allow_background and self.background in cols:
                cols.remove(self.background)
            out_col = random.choice(cols)
            out_x = random.randint(0, self.task.train[0][0].shape[1]) - self.task.train[0][0].shape[1]//2
            out_y = random.randint(0, self.task.train[0][0].shape[0]) - self.task.train[0][0].shape[0]//2
            rule.out_cols.append(out_col)
            #Calculate the angle and distance
            angle = np.arctan2(out_y, out_x)
            distance = np.sqrt((0 - out_x)**2 + (0 - out_y)**2)
            rule.out_angles.append(angle)
            rule.out_distances.append(distance)
        #Remove an out
        if(random.random() < 0.1):
            if(len(rule.out_cols) > 1):
                idx = random.randint(0, len(rule.out_cols)-1)
                rule.out_cols.pop(idx)
                rule.out_angles.pop(idx)
                rule.out_distances.pop(idx)

    def combine_rules(self, ruleset):
        #For each rule in the ruleset, combine it with each other rule
        #If the rules don't overlap, and the combined rule doesn't decrease the performance, add it to the new ruleset
        new_rules = []
        for i in range(len(ruleset)):
            for j in range(i+1, len(ruleset)):
                rule1 = ruleset[i]
                rule2 = ruleset[j]
                if(rule1 != rule2):
                    #Combine the rules
                    new_rule = rule1.combine(rule2)
                    #Check if the new rule decreases the performance
                    # print(new_rule)
                    self.score_rule(new_rule)
                    # print(rule1)
                    # print(rule2)
                    # print('-------------------')
                    if(new_rule.correct >= rule1.correct and new_rule.correct >= rule2.correct):
                        new_rules.append(new_rule)
        return new_rules

    def train(self, n_generations=10, move_to_next_gen_threshold=100, vis_after=False, iteration=0, heatmaps=None):
        max_ref_length = 5
        max_out_length = 1
        max_ruleset_size = 10000
        self.generate_initial_population(n_rules=max_ruleset_size, max_ref_length=max_ref_length, max_out_length=max_out_length, heatmaps=heatmaps)
        gens_of_no_improvement = 0
        scores = []
        original_task = copy.deepcopy(self.task)
        incorrect_coords = [[] for _ in range(len(self.task.train))]
        for i in range(n_generations):
            if(len(self.saved_perfect_rules) > 0):
                print('a')
            print(f'Generation {i}')
            print('Evaluating')
            #self.rules = [MultiRule([5, 5], [2], [-np.pi/2, 0], ['#', '#'], [0], [0], self.task, hash=False)]*1000
            #self.rules.append(MultiRule([5, 5], [2], [-np.pi/2, 0], ['#', '#'], [0], [0], self.task, hash=False))
            self.rules = self.evaluate_ruleset_parallel(self.rules)
            #Print the number of rules with a '#' in them
            num_hash = 0
            for rule in self.rules:
                if rule.distances.count('#') > 1:
                    num_hash += 1
                    # if(i > 0):
                    #     print(rule)
                    #     rule.visualise_rule()
            print('Number of rules with #:', num_hash)
            self.save_perfect_rules()
            if(vis_after <= i):
                print('Visualising')
                self.visualise_pop_stats()

            new_rules = []
            #Rules with a prop_correct of 1 are kept
            new_rules.extend(self.perfect_rules)

            #Generate 1 10th of the population through random generation
            print('Random Generation')
            if(i == 1):
                print('a')
            for j in range(max_ruleset_size//10):
                k = random.randint(0, len(self.task.train)-1)
                if(heatmaps is None):
                    new_rules.append(self.generate_rule_from_grid(self.task.train[k], max_ref_length=max_ref_length, max_out_length=max_out_length, incorrect_coords=incorrect_coords[k]))
                else:
                    new_rules.append(self.generate_rule_from_grid(self.task.train[k], max_ref_length=max_ref_length, max_out_length=max_out_length, heatmap=heatmaps[k]))
            incorrect_coords = [[] for _ in range(len(self.task.train))]

            #The rest are generated through crossover and mutation
            print('Selection')
            tournament = self.tournament_selection()
            tournament = sorted(tournament, key=lambda x: x.prop_correct, reverse=True)
            probabilities = np.linspace(0.1, 0.8, len(tournament))
            pairs = []
            for j in range(len(self.rules)-len(self.perfect_rules)):
                parent1 = random.choices(tournament, weights=probabilities)[0]
                parent2 = random.choices(tournament, weights=probabilities)[0]
                pairs.append((parent1, parent2))

            print('Crossover and mutation')
            for parents in pairs:
                new_rules.append(self.cross_mutate(parents))

            print('Rotating')
            #Rotate some of the rules and store the rotated rules as a new rule
            for rule in new_rules:
                if(random.random() < 0.1):
                    rule_copy = rule.make_copy()
                    if(random.random() < 0.5):
                        rule_copy.rotate_45()
                    else:
                        rule_copy.rotate_90()
                    new_rules.append(rule_copy)

            print('Extending')
            #Randomly set some of the distances and angles to be "#"
            if(use_lines):
                for rule in new_rules:
                    if(random.random() < 0.3 and len(rule.ref_cols) > 1):
                        rule_copy = rule.make_copy()
                        #Sample one of the distances that isn't already '#'
                        idx = random.randint(0, len(rule_copy.distances)-1)
                        l = 0
                        while(rule_copy.distances[idx] == '#'):
                            idx = random.randint(0, len(rule_copy.distances)-1)
                            l += 1
                            if(l > 5):
                                break
                        rule_copy.ref[idx].distance = '#'
                        if(l > 5):
                            continue
                        else:
                            # if(i > 0):
                                # print(rule)
                                # print(rule_copy)
                                # print('a')
                            new_rules.append(rule_copy)

            print('Evaluating New Population')
            self.rules = new_rules
            self.rules = self.evaluate_ruleset_parallel(self.rules)

            if(i > 0):
                print('a')
            self.save_perfect_rules()
            
            #Compact all perfect rules and test them on the examples
            print('Length of perfect rules:', len(self.perfect_rules))
            self.perfect_rules = self.remove_duplicates(self.perfect_rules)
            #Check to see if any of the rules fail when expanding the grid
            self.perfect_rules = self.filter_overspecified(self.perfect_rules)
            print('Filtering perfect rules')
            self.perfect_rules = self.filter_subsumption(self.perfect_rules)
            #Refine the perfect rules every other generation
            print('Refining perfect rules')
            self.perfect_rules = self.refine_ruleset(self.perfect_rules)
            # #Recentre the perfect rules
            # print('Recentering perfect rules')
            # for rule in self.perfect_rules:
            #     rule.recenter()
            # print('Combining perfect rules')
            # self.perfect_rules.extend(self.combine_rules(self.perfect_rules))
            prev_scores = scores
            scores = []
            for j, io in enumerate(self.task.train):
                grid = self.apply_self([self.perfect_rules], io, j, grid=io[0])
                scores.append(sum(sum(grid == io[1])))
                incorrect_coords[j].append(np.where(grid != io[1]))
                print(incorrect_coords[j])
                print(f'Score on Example {j}: {scores[j]}/{io[1].size}')

            print('Total Score:', sum(scores), '/', sum([io[1].size for io in self.task.train]))
            print('Previous Total Score:', sum(prev_scores))
            if(sum(scores) > sum(prev_scores)-2*len(self.task.train) and sum(scores) < sum(prev_scores)+2*len(self.task.train)):
                if(len(self.perfect_rules) != 0):
                    gens_of_no_improvement += 1
                    print('No Improvement')
            else:
                gens_of_no_improvement = 0

            self.perfect_rules = self.remove_duplicates(self.perfect_rules)

            print('Length of perfect rules:', len(self.perfect_rules))
            if(vis_after <= i):
                print('Visualising')
                self.vis_perfect_rules()

            if(vis_after <= i):
                for rule in self.perfect_rules:
                    print(rule)
                    rule.visualise_rule()
                    print(rule.correct, rule.incorrect, rule.prop_correct)
                    #rule.apply_to_io(self.task.train[5], np.zeros_like(self.task.train[5][1]))
                    print('---')

            if(vis_after <= i):
                print('Visualising')
                self.visualise_pop_stats()

            if sum(scores) == sum([io[1].size for io in self.task.train]):
                print('All examples solved')
                self.saved_perfect_rules.append(self.perfect_rules)
                for j, io in enumerate(original_task.train):
                    grid = self.apply_self(self.saved_perfect_rules, io, j)
                    if(vis_after <= i):
                        self.visualise_grid(grid)
                break

            if(gens_of_no_improvement >= move_to_next_gen_threshold):
                print('---------No Improvement---------')
                self.saved_perfect_rules.append(self.perfect_rules)
                self.perfect_rules = []
                for j, io in enumerate(original_task.train):
                    # init_grid = io[1]
                    # init_grid[~self.correct_masks[j]] = -1
                    grid = self.apply_self(self.saved_perfect_rules, io, j)
                    self.correct_masks[j] = grid == io[1]
                    self.task.train[j] = (grid, io[1])
                    if(vis_after <= i):
                        self.visualise_grid(self.task.train[j][0])
                self.rules = []
                self.generate_initial_population(n_rules=max_ruleset_size, max_ref_length=max_ref_length, max_out_length=max_out_length)
                gens_of_no_improvement = 0

        #Remove any duplicates from the saved perfect rules
        self.saved_perfect_rules.append(self.perfect_rules)
        self.saved_perfect_rules = [self.remove_duplicates(ruleset) for ruleset in self.saved_perfect_rules]
        self.task = original_task
        #Apply each iteration of perfect rules to the test set
        for i, io in enumerate(self.task.test):
            grid = self.apply_self(self.saved_perfect_rules, io, i)
            #self.visualise_grid(grid)
            print(f'Score on Example {i}: {sum(sum(grid == io[1]))}/{io[1].size}')

        # #Save the perfect rules as a pickle
        with open(path_folder + id + '_rulesets/perfect_rules'+str(iteration)+'.pkl', 'wb') as f:
            pickle.dump(self.saved_perfect_rules, f)
        
        # #Show the final ruleset
        # for i, ruleset in enumerate(self.saved_perfect_rules):
        #     print(f'Ruleset {i}')
        #     for rule in ruleset:
        #         print(rule)
        #         rule.visualise_rule()
        #         print(rule.correct, rule.incorrect, rule.prop_correct)
        #         print('---')

    #allow crossover and mutation to be passed in as functions to do in paralell
    def cross_mutate(self, parents):
        child = self.crossover(parents[0], parents[1])
        self.mutate(child)
        return child