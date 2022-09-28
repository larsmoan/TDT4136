from Map import *
import random

#This is used as the heurestic for the h-cost as well as the g-cost from current node to its neighbours - this allows for the agent to walk diagonally
def distance(startpos, endpos):
    dy = abs(startpos[0] - endpos[0])
    dx = abs(startpos[1] - endpos[1])
    return min(dx,dy)*14 + abs(dx - dy)*10

class Node:
    def __init__(self, map_obj: Map_Obj, pos):
        self.parent = None
        self.pos = pos
        self.g = 0
        self.h = distance(self.pos, map_obj.get_goal_pos())

        self.f = 0
    
    def __repr__(self):
        return f"Node at position: {self.pos}: g={self.g} h={self.h} f={self.get_f()}"
    
    #The reason for this to be a function and not a member value is so that it is initialized after the g-cost has been set in get_neighbours()
    def get_f(self):
        return self.g + self.h
    
    def __eq__(self, other):
        return self.pos == other.pos
    

def aStar(task, tuning_param):
    map = Map_Obj(task)
    num_iter = 0

    def get_neighbours(node):
        neighbour_nodes = []
       
        #All 8 neighbours - including walks diagonally
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for y, x in directions:

            neighbour_node_pos = [node.pos[0] + y, node.pos[1] + x]
            #Skip the neighbour if it is outside the grid
            if neighbour_node_pos[0] < 0 or neighbour_node_pos[1] < 0 or neighbour_node_pos[0] >= len(map.get_maps()[0]) \
                or neighbour_node_pos[1] > len(map.get_maps()[0][1]):
                continue
            
            #Check wether or not it is a wall
            if map.get_cell_value(neighbour_node_pos) < 0:
                continue
            
            #Initialize the neighbour position as a node
            neighbour_node = Node(map, neighbour_node_pos)
            #All neighbours will be children of the current node
            neighbour_node.parent = node
            #This ensures that the g-cost takes the cell value into account - check tuning for why tuning param is multiplied in
            neighbour_node.g = node.g + distance(node.pos, neighbour_node.pos)*map.get_cell_value(neighbour_node_pos)*tuning_param
            neighbour_node.f = neighbour_node.g + neighbour_node.h

            neighbour_nodes.append(neighbour_node)
        return neighbour_nodes
    
    #The actual path
    starting_node = Node(map, map.get_start_pos())
    #Wether or not this should be end goal or goal is not yet known
    end_node = Node(map, map.get_goal_pos())
    open_list = [starting_node]
    closed_list = []

    while len(open_list) > 0:
        num_iter += 1
        open_list.sort(key=lambda node: node.f)

        current_node = open_list.pop(0)
        closed_list.append(current_node)
        #Visualisez all nodes that are "opened"
        map.replace_map_values(current_node.pos, 8, map.get_end_goal_pos())

        #Unncomment to step through the search
        #map.update_image()

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current)
                #Visualize the path
                map.replace_map_values(current.pos, 7, map.get_goal_pos())
                current = current.parent
            
            print("Solution has been found! \n")
            #Higlight the starting point as green
            map.replace_map_values(map.get_start_pos(), 5, map.get_goal_pos())
            map.update_image(True)
            return path[::-1] , num_iter
        
        neighbour_nodes = get_neighbours(current_node)
        for neighbour_node in neighbour_nodes:
            #Check if the neighbour is already opened
            if neighbour_node not in open_list and neighbour_node not in closed_list:

                #Adding a new unique node to the open list
                open_list.append(neighbour_node)
            else:
                for existing_node in open_list:
                    if neighbour_node == existing_node and neighbour_node.f < existing_node.f:
                        open_list.remove(existing_node)
                        open_list.append(neighbour_node)

    print("No solution found")
    return False




def tuning():
    #Setup for tuning the parameter for g vs h
    results_sum = [] 
    for i in range(50):
        tuning_param = random.random()
        result = [tuning_param, 0]
        for task in range(1,5):
            #Use the tuning param on all tasks, sum the number of iterations
            path, num_iter = aStar(task, tuning_param)
            result[1] += num_iter

        results_sum.append(result)

    results_sum.sort(key=lambda x: x[1])
    print(results_sum)

    #Plotting the results with matplotlib as a scatter plot
    import matplotlib.pyplot as plt
    for i in range(len(results_sum)):
        plt.scatter(results_sum[i][0], results_sum[i][1])

    plt.xlabel("Tuning parameter * g")
    plt.ylabel("Number of iterations")
    plt.savefig("tuning_param.png")
    plt.show()


if __name__ == "__main__":
    for i in range(1,5):
        aStar(i, 1)
