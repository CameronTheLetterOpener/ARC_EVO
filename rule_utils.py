import numpy as np

def to_angles(ref_xs, ref_ys, out_xs, out_ys):
    angles = []
    distances = []
    out_x = 0
    out_y = 0
    for ref_x, ref_y in zip(ref_xs, ref_ys):
        angle = np.arctan2(ref_y - out_y, ref_x - out_x)
        distance = np.sqrt((out_x - ref_x)**2 + (out_y - ref_y)**2)
        angles.append(angle)
        distances.append(distance)
    out_angles = []
    out_distances = []
    for out_x, out_y in zip(out_xs, out_ys):
        angle = np.arctan2(out_y, out_x)
        distance = np.sqrt(out_x**2 + out_y**2)
        out_angles.append(angle)
        out_distances.append(distance)
    return angles, distances, out_angles, out_distances

def from_angles(angles, distances, out_angles, out_distances):
    ref_xs = []
    ref_ys = []
    out_xs = []
    out_ys = []
    for angle, distance in zip(angles, distances):
        ref_xs.append(int(round(np.cos(angle)*distance)))
        #Flipped due to image plane?
        ref_ys.append(int(round(np.sin(angle)*distance)))
    for angle, distance in zip(out_angles, out_distances):
        out_xs.append(int(round(np.cos(angle)*distance)))
        out_ys.append(int(round(np.sin(angle)*distance)))
    return ref_xs, ref_ys, out_xs, out_ys

class Reference:
    def __init__(self, col, angle, distance):
        #col, angle and distance can be "#" to indicate a wildcard
        self.col = col
        self.angle = angle
        self.distance = distance

    def get_xy(self, out_x, out_y, grid):
        #Get a list of the x and y coordinates of the reference pixels
        #There will be multiple if the angle and distance are not unique
        ref_xs = []
        ref_ys = []
        if self.distance == '#':
            #Get all the pixels in the grid along the line starting from out_x, out_y at the angle self.angle
            #This is a line of pixels in the direction of the angle'
            #Find where the line intersects the grid
            for i in range(grid.shape[1]):
                for j in range(grid.shape[0]):
                    if i == out_x and j == out_y:
                        continue
                    #Check if the pixel is on the line, make sure the direction is respected
                    #Compute the angle and distance from the output pixel to the reference pixel
                    angle = np.arctan2(j - out_y, i - out_x)
                    # if(self.angle == 0):
                    #     print('a')
                    #If the angles are equal within a small tolerance
                    if (angle - self.angle) % (2 * np.pi) < 0.01:
                            ref_xs.append(i)
                            ref_ys.append(j)
        elif self.angle == '#':
            #Get all the pixels in the grid at the distance self.distance from out_x, out_y
            for i in range(grid.shape[1]):
                for j in range(grid.shape[0]):
                    if np.sqrt((i - out_x)**2 + (j - out_y)**2) == self.distance:
                        ref_xs.append(i)
                        ref_ys.append(j)
        else:
            #Get the pixel at the angle and distance
            ref_x = int(round(out_x + np.cos(self.angle)*self.distance))
            ref_y = int(round(out_y + np.sin(self.angle)*self.distance))
            if(ref_x >= 0 and ref_x < grid.shape[1] and ref_y >= 0 and ref_y < grid.shape[0]):
                ref_xs.append(ref_x)
                ref_ys.append(ref_y)

        return ref_xs, ref_ys

class Output:
    def __init__(self, col, angle, distance):
        self.col = col
        self.angle = angle
        self.distance = distance

    def get_xy(self, out_x, out_y, grid):
        #For now, let's make outs simply
        out_xs = []
        out_ys = []
        #Get the pixel at the angle and distance
        out_x = int(round(out_x + np.cos(self.angle)*self.distance))
        out_y = int(round(out_y + np.sin(self.angle)*self.distance))
        if(out_x >= 0 and out_x < grid.shape[1] and out_y >= 0 and out_y < grid.shape[0]):
            out_xs.append(out_x)
            out_ys.append(out_y)
        return out_xs, out_ys
