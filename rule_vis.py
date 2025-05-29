import matplotlib.pyplot as plt
import numpy as np
import os, pickle, arckit

ARC_rgb_triplets = [(0, 0, 0), 
                    (0, 45.5, 85.1), 
                    (100.0, 25.5, 21.2), 
                    (18.0, 80.0, 25.1), 
                    (100.0, 86.3, 0), 
                    (66.7, 66.7, 66.7), 
                    (94.1, 7.1, 74.5), 
                    (100.0, 52.2, 10.6), 
                    (49.8, 85.9, 100.0),
                    (52.9, 4.7, 14.5)]

def arc_to_rgb(grid):
    #Convert a grid of ARC colours to an RGB grid
    plot_grid = np.zeros((grid.shape[0], grid.shape[1], 3))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if(grid[i][j] != -1):
                plot_grid[i][j] = np.array(ARC_rgb_triplets[int(grid[i][j])])/100
            else:
                plot_grid[i][j] = [1, 1, 1]
    return plot_grid

def visualise_grid(in_grid, show=True, title=''):
    #Show a grid with ARC colours as in visualise rule
    plot_grid = arc_to_rgb(in_grid)
    plt.imshow(plot_grid)
    plt.title(title)
    if show:
        plt.show()

def fill_pixels(grid, xs, ys, cols):
    for i, (x_list, y_list) in enumerate(zip(xs, ys)):
        for x, y in zip(x_list, y_list):
            if x >= 0 and x < grid.shape[1] and y >= 0 and y < grid.shape[0]:
                grid[y][x] = cols[i]
    return grid

def visualise_rule(rule, v=False):
    if(v):
        print(rule)
    #Generate a grid and show the relative positions of the refs to the out
    ref_distances = [d if d != '#' else max(rule.task.train[0][0].shape)//2 for d in rule.distances]
    max_dist = np.ceil(max(max(ref_distances), max(rule.out_distances))+0.01)
    in_grid = np.ones((int(max_dist*2)+1, int(max_dist*2)+1))*-1
    out_grid = np.ones((int(max_dist*2)+1, int(max_dist*2)+1))*-1

    ref_x, ref_y = rule.ref_loc(int(max_dist), int(max_dist), -1)
    in_grid = fill_pixels(in_grid, ref_x, ref_y, rule.ref_cols)
    #Replace the grid cols with the colours in rgb_triplets
    plt.subplot(1, 2, 1)
    visualise_grid(in_grid, show=False, title='In Grid')
    #Generate a grid and show the relative positions of the refs to the out
    out_x, out_y = rule.out_loc(int(max_dist), int(max_dist), -1)
    out_grid = fill_pixels(out_grid, out_x, out_y, rule.out_cols)
    #Replace the grid cols with the colours in rgb_triplets
    plt.subplot(1, 2, 2)
    visualise_grid(in_grid, show=False, title='Out Grid')
    plt.show()

def vis_heatmaps(id, path_folder, vis=False, vis_rules=False, target_pixel=None):
    task = arckit.load_single(id)
    in_heatmaps = []
    out_heatmaps = []
    for io in task.train:
        in_heatmap = np.zeros_like(io[0])
        out_heatmap = np.zeros_like(io[1])
        in_heatmaps.append(in_heatmap)
        out_heatmaps.append(out_heatmap)
    for j in range(len(os.listdir(path_folder + id + '_rulesets'))):

        #load the pickle file
        with open(path_folder + id + '_rulesets/perfect_rules' + str(j) + '.pkl', 'rb') as f:
            perfect_rules = pickle.load(f)

        #Generate heatmaps of which pixels the rules reference on each of the input grids
        #Generate heatmaps of the output grid as well
        for i, io in enumerate(task.train):
            for ruleset in perfect_rules:
                for rule in ruleset:                    
                    in_h, out_h = rule.gen_heatmaps(io, i, target_pixel)
                    in_heatmaps[i] += in_h
                    out_heatmaps[i] += out_h
    for i, io in enumerate(task.train):
        if(vis):
            plt.subplot(1, 2, 1)
            plt.imshow(in_heatmaps[i])
            plt.title('In Heatmap')
            plt.subplot(1, 2, 2)
            plt.imshow(out_heatmaps[i])
            plt.title('Out Heatmap')
            plt.show()

    if(vis_rules):
        for i, ruleset in enumerate(perfect_rules):
                print(f'------------Ruleset {i}------------')
                for rule in ruleset:
                    print(rule)
                    rule.visualise_rule()
                    print(rule.correct, rule.incorrect, rule.prop_correct)
                    print('---')

    #Generate heatmaps of the output grid as well
    heatmaps = []
    for i, o in zip(in_heatmaps, out_heatmaps):
        heatmaps.append((i, o))

    return heatmaps

def visualise_pop_stats(self):
    noise = (np.random.rand(len(self.rules))-0.5)*2
    correct = np.array([rule.correct for rule in self.rules]) + noise
    incorrect = np.array([rule.incorrect for rule in self.rules]) + noise
    prop_correct = [rule.prop_correct for rule in self.rules]
    plt.scatter(correct, incorrect, c=prop_correct, cmap='viridis')
    plt.xlabel('Correct')
    plt.ylabel('Incorrect')
    plt.colorbar()
    plt.show()

def vis_perfect_rules(self):
    #Show distribution of perfect rule length both ref and out
    ref_lengths = [len(rule.ref_cols) for rule in self.perfect_rules]
    out_lengths = [len(rule.out_cols) for rule in self.perfect_rules]
    plt.hist(ref_lengths, bins=range(1, 10), alpha=0.5, label='Ref Lengths')
    plt.hist(out_lengths, bins=range(1, 10), alpha=0.5, label='Out Lengths')
    plt.legend()
    plt.show()

    #Plot perfect rule angles and distances, both ref and out with num correct on y axis
    #Angles and distances come as lists, unpack these into a single list
    #Do top angles and bottom distances using subplots
    ref_angles_lists = [rule.angles for rule in self.perfect_rules]
    out_angles_lists = [rule.out_angles for rule in self.perfect_rules]
    ref_distances_lists = [rule.distances for rule in self.perfect_rules]
    out_distances_lists = [rule.out_distances for rule in self.perfect_rules]
    corrects = [rule.correct for rule in self.perfect_rules]
    
    ref_angles = []
    out_angles = []
    ref_distances = []
    out_distances = []
    ref_corrects = []
    out_corrects = []
    #Since each rule has different numbers of refs and outs, make 2 correct lists
    for angle_list, distance_list, correct in zip(ref_angles_lists, ref_distances_lists, corrects):
        angle_list = [angle if angle != '#' else 8 for angle in angle_list]
        distance_list = [distance if distance != '#' else 100 for distance in distance_list]
        ref_angles.extend(angle_list)
        ref_distances.extend(distance_list)
        ref_corrects.extend([correct]*len(angle_list))
    for angle_list, distance_list, correct in zip(out_angles_lists, out_distances_lists, corrects):
        out_angles.extend(angle_list)
        out_distances.extend(distance_list)
        out_corrects.extend([correct]*len(angle_list))

    ref_y_noise = (np.random.rand(len(ref_angles))-0.5)*0.2
    out_y_noise = (np.random.rand(len(out_angles))-0.5)*0.2
    ref_corrects = np.array(ref_corrects) + ref_y_noise
    out_corrects = np.array(out_corrects) + out_y_noise
        
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].scatter(ref_angles, ref_corrects)
    axs[0, 0].set_title('Ref Angles')
    axs[0, 1].scatter(ref_distances, ref_corrects)
    axs[0, 1].set_title('Ref Distances')
    axs[1, 0].scatter(out_angles, out_corrects)
    axs[1, 0].set_title('Out Angles')
    axs[1, 1].scatter(out_distances, out_corrects)
    axs[1, 1].set_title('Out Distances')
    plt.show()