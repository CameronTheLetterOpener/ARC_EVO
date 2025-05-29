import numpy as np
import arckit, os, random

def run_train(id, iteration=0, vis_after=False, heatmaps=None):
    task = arckit.load_single(id)
    lcs = LCS_ARC(task, allow_background_in_outcol=False)
    lcs.train(vis_after=vis_after, iteration=iteration, heatmaps=heatmaps, n_generations=1)

def test_rule_on_io(id):
    task = arckit.load_single(id)
    #Load the perfect rules
    ref_cols = [3, 3, 3, 3, 0]
    out_cols = [4]
    angles = [-np.pi/2, 0, np.pi, np.pi/2, 0]
    distances = ['#', '#', '#', '#', 0]
    out_angles = [0]
    out_distances = [0]
    rule = MultiRule(ref_cols, out_cols, angles, distances, out_angles, out_distances, task)
    for idx, io in enumerate(task.train):
        #Apply the rule to the io
        out, out_cor = rule.apply_to_io(io, idx, score=True)
        #Visualise the rule
        print(out)
    print(rule)
    rule.visualise_rule()

heatmap_path_folder = 'ARC_rulesets/'
path_folder = 'ARC_rulesets/'
use_lines = True
scenario = 0

if __name__ == '__main__':
    if(scenario == 0):
        # ids = ['25d8a9c8', 
        #     '6e19193c', 'd406998b', '54d9e175', '7df24a62', '00d62c1b', 'e8593010', '60b61512', '4612dd53', '5521c0d9', '1caeab9d']
        # ids = ['00d62c1b', 'e8593010', '60b61512', '4612dd53', '5521c0d9', '1caeab9d']
        # ids = ['0ca9ddb6']
        # ids = ['54d9e175', '41e4d17e', '2281f1f4', 'ae3edfdc', '4612dd53', '5521c0d9', '1caeab9d']
        ids = ['e8593010']
        # ids = ['2281f1f4']#, 'ae3edfdc']
        # ids = ['4612dd53']
        # ids = ['5521c0d9']
        # ids = ['1caeab9d']
        # ids = ['a48eeaf7']
        # ids = ['60b61512']
        # ids = ['50cb2852', '54d82841', '42a50994', '7e0986d6']
        for id in ids:
            path = path_folder + id + '_rulesets'
            if not os.path.exists(path):
                os.makedirs(path)
            for i in range(0, 6):
                seed = i
                random.seed(seed)
                np.random.seed(seed)

                run_train(id, iteration=i, vis_after=100)
    if(scenario == 1):
        use_lines = False
        ids = ['50cb2852']
        ids = ['e8593010']
        #7df24a62 is an interesting example. IO pair 2 (3) has a unique pattern to it which makes the system learn better
        #The system is overall weak for puzzles that require many long range references e.g. 00d62c1b and d406998b
        #e8593010 is an example where the background colour needs to be dynamic.
        #TODO: rerun above with correct background colour and try to construct objects
        for id in ids:
            vis_heatmaps(id, vis=1, vis_rules=1, target_pixel=(6, 2))
    if(scenario == 2):
        ids = ['a48eeaf7']
        for id in ids:
            path = path_folder + id + '_rulesets'
            if not os.path.exists(path):
                os.makedirs(path)
            for i in range(0, 6):
                seed = i
                random.seed(seed)
                np.random.seed(seed)

                heatmaps = vis_heatmaps(id, vis=False)

                run_train(id, iteration=i, vis_after=11, heatmaps=heatmaps)
    if(scenario == 3):
        ids = ['00d62c1b']
        use_lines = False
        for id in ids:
            test_rule_on_io(id)