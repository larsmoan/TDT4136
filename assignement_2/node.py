from a_star import Node
from Map import *

#Create a node for the start position
map = Map_Obj(1)
starting_node = Node(map, map.get_start_pos())

#Create a succesor node
succesor_node = Node(map, [starting_node.pos[0] + 1, starting_node.pos[1]])
succesor_node.parent = starting_node
