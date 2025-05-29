import numpy as np

def out_condition(self, io, i_outs, j_outs):
    conditions = []
    successful_outs = []
    for k, (i_out_list, j_out_list, col) in enumerate(zip(i_outs, j_outs, self.out_cols)):
        conditions.append(False)
        #This inner loop if for outs with wildcards, if all are true, then the out returns true
        inner_conditions = []
        inner_successful_outs = []
        for i_out, j_out in zip(i_out_list, j_out_list):
            if i_out >= 0 and i_out < len(io[1][0]) and j_out >= 0 and j_out < len(io[1]):
                if(io[1][j_out][i_out] == col):
                    inner_conditions.append(True)
                    inner_successful_outs.append((i_out, j_out))
                else:
                    inner_conditions.append(False)
        #If all the inner conditions are true, then the out condition is true
        if all(inner_conditions):
            conditions[k] = True
            #Add the successful reference to the list
            successful_outs.append(inner_successful_outs)
                    
    return conditions, successful_outs

def eval_condition(self, io, i_refs, j_refs):
    conditions = []
    successful_refs = []
    for k, (i_ref_list, j_ref_list, col) in enumerate(zip(i_refs, j_refs, self.ref_cols)):
        conditions.append(False)
        #This inner loop if for refs with wildcards, if any are true, then the ref returns true
        for i_ref, j_ref in zip(i_ref_list, j_ref_list):
            if i_ref >= 0 and i_ref < len(io[1][0]) and j_ref >= 0 and j_ref < len(io[1]):
                if(io[0][j_ref][i_ref] == col or self.rel_ref_key[k]):
                    conditions[k] = True
                    if(self.rel_ref_key[k]):
                        conditions[k] = col
                    #Add the successful reference to the list
                    successful_refs.append((i_ref, j_ref))
                    break
        
    return conditions, successful_refs 

def apply_to_io(rule, io, io_i, score=False):
    #In the voting scheme, the out is this rule's votes
    out = np.ones_like(io[1])*-1
    out_cor = np.zeros_like(io[1])
    
    for j_out in range(len(io[1])):
        for i_out in range(len(io[1][0])):
            # if(j_out == 2 and i_out == 11 and io_i == 0):
            #     print('a')
            i_refs, j_refs = rule.ref_loc(i_out, j_out, io_i)
            #Conditions marks whether each of the refs are satisfied
            conditions, successful_refs = rule.eval_condition(io, i_refs, j_refs)
            if rule.check_condition(conditions):
                #Rewrite the above to allow for multiple out cols
                out_xs, out_ys = rule.out_loc(i_out, j_out, io_i)
                out_conditions, successful_outs = rule.out_condition(io, out_xs, out_ys)
                for out_list in successful_outs:
                    for i_out_ref, j_out_ref in out_list:
                        out_cor[j_out_ref][i_out_ref] = True
                #Update the output grid
                for i, (out_x_list, out_y_list) in enumerate(zip(out_xs, out_ys)):
                    for out_x, out_y in zip(out_x_list, out_y_list):
                        if out_x >= 0 and out_x < len(io[1][0]) and out_y >= 0 and out_y < len(io[1]):
                            out[j_out][i_out] = rule.out_cols[i]

                if score:
                    if all(out_conditions):
                        rule.correct += len(out_conditions)
                    else:
                        rule.incorrect += len(out_conditions)
    return out, out_cor

def gen_heatmaps(self, io, io_i, target_pixel, out_pix=None):
    in_heatmap = np.zeros_like(io[0])
    out_heatmap = np.zeros_like(io[1])
    for j_out in range(len(io[1])):
        for i_out in range(len(io[1][0])):
            i_refs, j_refs = self.ref_loc(i_out, j_out, io_i)
            #Conditions marks whether each of the refs are satisfied
            conditions = []
            successful_refs = []
            for k, (i_ref_list, j_ref_list, col) in enumerate(zip(i_refs, j_refs, self.ref_cols)):
                conditions.append(False)
                for i_ref, j_ref in zip(i_ref_list, j_ref_list):
                    if i_ref >= 0 and i_ref < len(io[1][0]) and j_ref >= 0 and j_ref < len(io[1]):
                        conditions[k] = io[0][j_ref][i_ref] == col
                        #Add the successful reference to the list
                        successful_refs.append((i_ref, j_ref))
                        break
            # if(i_out > 11):
            #     print('a')
            if all(conditions) and (out_pix is None or (out_pix[0] == i_out and out_pix[1] == j_out)):
                j_refs = [j_ref for j_ref_list in j_refs for j_ref in j_ref_list]
                i_refs = [i_ref for i_ref_list in i_refs for i_ref in i_ref_list]
                if(target_pixel is None):
                    in_heatmap[j_refs, i_refs] += 1
                out_xs, out_ys = self.out_loc(i_out, j_out, io_i)
                for i, (out_x_list, out_y_list) in enumerate(zip(out_xs, out_ys)):
                    for out_x, out_y in zip(out_x_list, out_y_list):
                        if out_x >= 0 and out_x < len(io[1][0]) and out_y >= 0 and out_y < len(io[1]):
                            if target_pixel is None:
                                out_heatmap[out_y][out_x] += 1
                            else:
                                if target_pixel[0] == out_x and target_pixel[1] == out_y:
                                    out_heatmap[out_y][out_x] += 1
                                    in_heatmap[j_refs, i_refs] += 1
                
    return in_heatmap, out_heatmap