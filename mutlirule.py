import random
import numpy as np

class MultiRule:
    #Not currently set up for generalisation
    def __init__(self, ref_cols, out_cols, angles, distances, out_angles, out_distances, task, deg=False, hash=True, relative_refs=False):
        #Rel coords have the output pixel as the end.
        for i in range(len(ref_cols)):
            if(random.random() < 0.1 and use_lines and hash):
                distances[i] = '#'
                #Round the angle to the nearest 45 degrees
                if(deg):
                    angles[i] = int(round(angles[i]/np.pi*180/45)*45)
                else:
                    #Now in radians
                    angles[i] = round(angles[i]/np.pi*180/45)*np.pi/180
        self.ref = []
        for i in range(len(ref_cols)):
            self.ref.append(Reference(ref_cols[i], angles[i], distances[i]))
        self.out = []
        for i in range(len(out_cols)):
            self.out.append(Output(out_cols[i], out_angles[i], out_distances[i]))

        self.sort_rule()

        self.level = random.randint(0, 3)

        self.correct = 0
        self.incorrect = 0
        self.prop_correct = 0

        self.task = task
        if(hash and random.random() < 0.5):
        # if(relative_refs):
            self.rel_ref_key = np.ones_like(self.ref_cols, dtype=bool).tolist()
        else:
            self.rel_ref_key = np.zeros_like(self.ref_cols, dtype=bool).tolist()

        # if(len(ref_cols) == 2):
        #     self.visualise_rule()

        self.ref_stats = []
        self.out_stats = []
        for i, io in enumerate(task.train):
            self.ref_stats.append(np.zeros_like(io[0]))
            self.out_stats.append(np.zeros_like(io[1]))

    @property
    def ref_cols(self):
        return [r.col for r in self.ref]
    
    @property
    def angles(self):
        return [r.angle for r in self.ref]
    
    @property
    def distances(self):
        return [r.distance for r in self.ref]
    
    @property
    def out_cols(self):
        return [o.col for o in self.out]
    
    @property
    def out_angles(self):
        return [o.angle for o in self.out]
    
    @property
    def out_distances(self):
        return [o.distance for o in self.out]

    def sort_rule(self):
        obj = np.rec.fromarrays([self.ref_cols, self.angles, self.distances], names='ref_cols, angles, distances')
        #Output an array of indexes to reorder the refs
        idxs = np.argsort(obj, order=['distances', 'angles', 'ref_cols'])
        self.ref = [self.ref[i] for i in idxs]

        obj = np.rec.fromarrays([self.out_cols, self.out_angles, self.out_distances], names='out_cols, out_angles, out_distances')
        #Output an array of indexes to reorder the outs
        idxs = np.argsort(obj, order=['out_distances', 'out_angles', 'out_cols'])
        self.out = [self.out[i] for i in idxs]

    def to_angles(ref_xs, ref_ys, out_xs, out_ys):
        angles = []
        distances = []
        first_out_x = out_xs[0]
        first_out_y = out_ys[0]
        for ref_x, ref_y in zip(ref_xs, ref_ys):
            angle = np.arctan2(ref_y - first_out_y, ref_x - first_out_x)
            distance = np.sqrt((first_out_x - ref_x)**2 + (first_out_y - ref_y)**2)
            angles.append(angle)
            distances.append(distance)
        out_angles = []
        out_distances = []
        for out_x, out_y in zip(out_xs, out_ys):
            angle = np.arctan2(out_y - first_out_y, out_x - first_out_x)
            distance = np.sqrt((first_out_x - out_x)**2 + (first_out_y - out_y)**2)
            out_angles.append(angle)
            out_distances.append(distance)
        return angles, distances, out_angles, out_distances

    @classmethod
    def from_coords(cls, ref_cols, out_cols, ref_xs, ref_ys, out_xs, out_ys, task):
        angles, distances, out_angles, out_distances = cls.to_angles(ref_xs, ref_ys, out_xs, out_ys)
        return cls(ref_cols, out_cols, angles, distances, out_angles, out_distances, task)
    
    def make_copy(self, hash=False):
            ref_cols = self.ref_cols.copy()
            angles = self.angles.copy()
            distances = self.distances.copy()
            out_cols = self.out_cols
            out_angles = self.out_angles.copy()
            out_distances = self.out_distances.copy()
            return MultiRule(ref_cols, out_cols, angles, distances, out_angles, out_distances, self.task, hash=hash, relative_refs=any(self.rel_ref_key))
    
    def crossover(self, other):
        #Rules can be different lengths
        cross_point = random.randint(0, min(len(self.ref_cols), len(other.ref_cols)))
        ref_cols = self.ref_cols[:cross_point] + other.ref_cols[cross_point:]
        angles = self.angles[:cross_point] + other.angles[cross_point:]
        distances = self.distances[:cross_point] + other.distances[cross_point:]
        out_cross_point = random.randint(0, min(len(self.out_cols), len(other.out_cols)))
        out_cols = self.out_cols[:out_cross_point] + other.out_cols[out_cross_point:]
        out_angles = self.out_angles[:out_cross_point] + other.out_angles[out_cross_point:]
        out_distances = self.out_distances[:out_cross_point] + other.out_distances[out_cross_point:]
        rel_refs = self.rel_ref_key[:cross_point] + other.rel_ref_key[cross_point:]
        return MultiRule(ref_cols, out_cols, angles, distances, out_angles, out_distances, self.task, relative_refs=any(rel_refs))

    def ref_loc(self, out_x, out_y, i):
        ref_xs = []
        ref_ys = []
        if(i == -1):
            grid = np.zeros((out_x*2, out_y*2))
        else:
            grid = self.task.train[i][0]
        for ref in self.ref:
            ref_x, ref_y = ref.get_xy(out_x, out_y, grid)
            ref_xs.append(ref_x)
            ref_ys.append(ref_y)
        return ref_xs, ref_ys
    
    def out_loc(self, out_x, out_y, i):
        out_xs = []
        out_ys = []
        if(i == -1):
            grid = np.zeros((out_x*2, out_y*2))
        else:
            grid = self.task.train[i][0]
        for out in self.out:
            out_x, out_y = out.get_xy(out_x, out_y, grid)
            out_xs.append(out_x)
            out_ys.append(out_y)
        return out_xs, out_ys
    
    def df_row(self):
        return [self.ref_cols, self.out_cols, list(np.round(self.angle_deg, 3)), list(np.round(self.distances, 3)), self.correct, self.incorrect, self.prop_correct, list(np.round(self.rel_coords_x, 3)), list(np.round(self.rel_coords_y))]
    
    @property
    def angle_deg(self):
        return [np.degrees(a) for a in self.angles]

    @property
    def out_angle_deg(self):
        return [np.degrees(a) for a in self.out_angles]
    
    def __str__(self):
        return f"Rule: {np.array(self.ref_cols)} -> {np.array(self.out_cols)} at ref_angles {np.array(self.angle_deg)}, ref_distances {np.array(self.distances)}, rel_refs {np.array(self.rel_ref_key)}, out_angles {np.array(self.out_angle_deg)} and out_distances {np.array(self.out_distances)} with correct {self.correct} and incorrect {self.incorrect} and prop_correct {self.prop_correct}"
    
    def check_condition(self, condition):
        cond_arr = np.array(condition)
        ref_arr = np.array(self.ref_cols)
        rel_arr = np.array(self.rel_ref_key)
        #Where rel_arr is True, cond_arr and ref_arr must match pattern-wise.
        rel_mask = np.where(rel_arr)[0]
        if np.any(rel_arr):
            #Make sure that the len(np.unique) of cond_arr matches ref_array
            unique_cond = np.unique(cond_arr[rel_mask])
            unique_ref = np.unique(ref_arr[rel_mask])
            if len(unique_cond) != len(unique_ref):
                return False
        return np.all(cond_arr[~rel_arr])

    def __eq__(self, other):
        return self.ref_cols == other.ref_cols and self.out_cols == other.out_cols and self.angles == other.angles and self.distances == other.distances

    #Provide the ability to combine rules
    def combine(self, other):
        #Check the ref_stats to see if any of the refs are the same
        ref_pairs = []
        for i in range(len(self.ref_cols)):
            for j in range(len(other.ref_cols)):
                #Check if the ref_stats are the same
                conditions = []
                for k in range(len(self.ref_stats)):
                    if np.all((self.ref_stats[k] == i+1) == (other.ref_stats[k] == j+1)):
                        conditions.append(True)
                    else:
                        conditions.append(False)
                if all(conditions):
                    ref_pairs.append((i, j))
        #If there are any overlapping refs, combine them, re-aligning the angles and distances to these refs
        if len(ref_pairs) > 0:
            #Realign the angles and distances to the first ref_pair
            ref_pair = ref_pairs[0]
            ref_x, ref_y = self.ref_loc(0, 0, 0)
            ref_x2, ref_y2 = other.ref_loc(0, 0, 0)
            ref_pair = ref_pairs[0]
            ref_xs, ref_ys, out_xs, out_ys = from_angles(self.angles, self.distances, self.out_angles, self.out_distances)
            ref_xs2, ref_ys2, out_xs2, out_ys2 = from_angles(other.angles, other.distances, other.out_angles, other.out_distances)
            #Minus the first ref pair from the rest of the refs
            for i in range(len(ref_xs)):
                ref_xs[i] = ref_xs[i] - ref_x[ref_pair[0]]
                ref_ys[i] = ref_ys[i] - ref_y[ref_pair[0]]
            for i in range(len(ref_xs2)):
                ref_xs2[i] = ref_xs2[i] - ref_x2[ref_pair[1]]
                ref_ys2[i] = ref_ys2[i] - ref_y2[ref_pair[1]]
            for i in range(len(out_xs)):
                out_xs[i] = out_xs[i] - ref_x[ref_pair[0]]
                out_ys[i] = out_ys[i] - ref_y[ref_pair[0]]
            for i in range(len(out_xs2)):
                out_xs2[i] = out_xs2[i] - ref_x2[ref_pair[1]]
                out_ys2[i] = out_ys2[i] - ref_y2[ref_pair[1]]
            #Convert the angles and distances to the new coords
            angles, distances, out_angles, out_distances = to_angles(ref_xs, ref_ys, out_xs, out_ys)
            angles2, distances2, out_angles2, out_distances2 = to_angles(ref_xs2, ref_ys2, out_xs2, out_ys2)
            self.angles, self.distances, self.out_angles, self.out_distances = angles, distances, out_angles, out_distances
            other.angles, other.distances, other.out_angles, other.out_distances = angles2, distances2, out_angles2, out_distances2

        #Combine the ref cols and angles
        ref_cols = self.ref_cols + other.ref_cols
        angles = self.angles + other.angles
        distances = self.distances + other.distances
        out_cols = self.out_cols + other.out_cols
        out_angles = self.out_angles + other.out_angles
        out_distances = self.out_distances + other.out_distances
        return MultiRule(ref_cols, out_cols, angles, distances, out_angles, out_distances, self.task)
    
    #Allow rules to re-centre themselves around their first output
    def recenter(self):
        out_x, out_y = self.out_loc(0, 0, 0)
        ref_x, ref_y = self.ref_loc(-out_x[0], -out_y[0], 0)
        for i in range(len(self.ref_cols)):
            self.angles[i] = np.arctan2(ref_y[i], ref_x[i])
            self.distances[i] = np.sqrt(ref_x[i]**2 + ref_y[i]**2)
        #Do the same with the outputs, normalise them to the first array entry
        for i in range(len(self.out_cols)):
            out_x[i] = out_x[i] - out_x[0]
            out_y[i] = out_y[i] - out_y[0]
            self.out_angles[i] = np.arctan2(out_y[i], out_x[i])
            self.out_distances[i] = np.sqrt(out_x[i]**2 + out_y[i]**2)
        #Sort the rule again
        self.sort_rule()

    def rotate_45(self):
        #Rotate the rule by 45 degrees
        for i in range(len(self.ref_cols)):
            self.angles[i] = (self.angles[i] + np.pi/4) % (2 * np.pi)
        for i in range(len(self.out_cols)):
            self.out_angles[i] = (self.out_angles[i] + np.pi/4) % (2 * np.pi)
        #Sort the rule again
        self.sort_rule()

    def rotate_90(self):
        #Rotate the rule by 90 degrees
        for i in range(len(self.ref_cols)):
            self.angles[i] = (self.angles[i] + np.pi/2) % (2 * np.pi)
        for i in range(len(self.out_cols)):
            self.out_angles[i] = (self.out_angles[i] + np.pi/2) % (2 * np.pi)
        #Sort the rule again
        self.sort_rule()