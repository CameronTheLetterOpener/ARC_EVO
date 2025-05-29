import arckit
import numpy as np

# Create a search object
array = np.array([[4, 0, 0], [4, 4, 4], [0, 4, 0]])

#Load ARC
train_set, eval_set = arckit.load_data()

def grid_contains(grid, array):
    #Check if a grid contains a certain pattern
    contained = False
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            #Check if the pattern is contained in the grid
            #The pattern can be bigger than 1x1
            if(i + len(array) > len(grid) or j + len(array[0]) > len(grid[i])):
                continue
            contained = True
            for x in range(len(array)):
                for y in range(len(array[x])):
                    if(array[x][y] != grid[i + x][j + y]):
                        contained = False
                        break
                if(not contained):
                    break
            if(contained):
                break
        if(contained):
            break
    return contained

def search(set, array):
    #Search through every grid in the array and see if any array contains the same pattern as the input array
    #Output all tasks that contain the pattern
    matches = []
    for task in set:
        print(task)
        contained = False
        for io in task.train:
            for grid in io:
                if(grid_contains(grid, array)):
                    matches.append(task.id)
                    contained = True
                    break
            if(contained):
                break
        if(contained):
            continue
                    
        for io in task.test:
            for grid in io:
                if(grid_contains(grid, array)):
                    matches.append(task.id)
                    contained = True
                    break
            if(contained):
                break
    return matches
    
matches = search(train_set, array)
print(matches)