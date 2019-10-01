#!/usr/bin/env python
# coding: utf8

#==============================================================================
#title           :patter_optimization.py
#description     :optimize a created automata from samples
#author          :POUSSEUR Hugo
#date            :20180705
#version         :0.1
#usage           :python patter_optimization.py
#python_version  :2.7
#==============================================================================


# ----------------- Imports -------------------------

import numpy as np
import random as rand
import matplotlib.pyplot as plt
from graphviz import Digraph

import scipy as sp
import scipy.stats
import scipy.io.wavfile

# ----------------- Global Variables -------------------------
# -- EDITABLE --

NUMBER_ANTS = 1000
NUMBER_LOOPS_PATTERN_SEARCH = 100
LIMIT_LEN_SUBSTRING = 3

HSV1 = [0.0,0.20,0.20]
HSV2 = [0.0,1.0,1.0]
STEPS_COLORS_ARRAY = 100

ALPHABET_TEST = ["a","b","c","d","e","f","g","h","i","j"]

NUMBER_WORDS = 7
LEN_WORLD = 20
NUMBER_TESTS = 10

DEMO = "demo"
STATS = "stats"
DEBUG = "debug"

FUNCTION_TO_START = DEBUG

NUMBER_MAX_ESTIMATION = 100
NUMBER_ANTS_PREDICTION = 1000

# -- NO EDITABLE --

# dictionnary variables
BEGIN_S1 = "begin_s1"
BEGIN_S2 = "begin_s2"
SUBSTRING = "substring"

NUMBER_BEFORE_OPTI = "number_before_opti"
NUMBER_AFTER_OPTI = "number_after_opti"
NUMBER_STATES_STATIONARY = "number_states_stationary"
LEN_AVERAGE_WORLD_CREATE_BY_ANTS = "len_world_average"

EDGES_AVAILABLE = "edges_available"
NUMBER_PATHS_LESS = "number_paths_less"

# path variables
PATH_SRC_FOLDER = "./src/graphviz"






# ----------------- Classes Definitions -------------------------

class Node:
        """
        Represents a state of the automate, each state can get edge children.
        This edge allows to point on another node.
        Each node gets informations about itself, like the cross frequency
        """
        def __init__(self):
            """
            :n_in: inputs number
            :_f_out: outputs number
            """
            self.child_edges = []
            self.state = Tree._COUNTER_NODE
            self.n_in = 0
            self._f_out = 0
            self.current_n_in = 0
            self.current_f_out = 0
            Tree._COUNTER_NODE += 1
            self.is_stationary = False


        def get_child_by_data(self,data):
            """
            research a child in according data given
            return None in the worst case
            """
            for edge in self.child_edges:
                if(str(edge.data) == str(data)):
                    return edge
            return None


        def add_path_in(self):
            """
            Update the inputs number
            """
            self.n_in += 1
            self.current_n_in += 1


        def add_path_out(self):
            """
            Update the outputs number
            """
            self._f_out += 1
            self.current_f_out += 1


        def add_new_child(self,edge):
            """
            Add new edge child at the current node
            """
            self.child_edges.append(edge)


        def get_path_stay_in(self):
            """
            In according the inputs and outputs number, it's easy to dedious
            the end world number at this state
            """
            return self.n_in - self._f_out


        def merge(self,node):
            """
            Merge a node given with this current node
            """
            #self.state += ","+str(node.state)
            self._f_out -= node.get_path_stay_in()

        def reset_freq(self):
            self.current_n_in = self.n_in
            self.current_f_out = self._f_out


        def __str__(self, level = 1):
            """
            Insure the tree console display
            """
            ret = '> ('+repr(self.state) \
                +") ["+ str(self.n_in)+"]" \
                +str(self._f_out)+"]"

            ret += "\n"

            for child_edge in self.child_edges:
                if(child_edge.child_node.state > self.state):
                        ret += "\t"*level+"-"+str(child_edge)
                        ret += child_edge.child_node.__str__(level+1)
            return ret


        def __repr__(self):
            return '<tree node representation>'




class Edge:
    """
    Represents a transition between 2 states, this transition stores data.
    Each data corresponds at a letter from the alphabet
    """

    NONE_CHAR = '#'


    def __init__(self,data,child_node,n=0):
        self.child_node = child_node
        self.data = data
        self.n = n
        self.n_random = 0

        self.current_n = n


    def add_path_on(self):
        """
        Update the use of itself
        """
        self.n += 1
        self.current_n += 1


    def merge(self,edge):
        """
        Allows to merge 2 edges, it's the edge given that be merged with
        the current edge.
        """
        if(edge.data != Edge.NONE_CHAR):
            self.data += edge.data
        self.child_node = edge.child_node


    def add_path_on_random(self):
        self.n_random += 1


    def reset_freq(self):
        self.current_n = self.n


    def __str__(self):
        return self.data+"["+str(self.n)+"]"




class Tree:
    """
    Represents the automata in whole. Accessible by the top node called root
    In this class, it's possible to build a tree from exemple and optimize it.
    """
    def __init__(self,set_of_samples):
        Tree._COUNTER_NODE = 0

        self.number_states = 0
        self.number_states_stationnary = 0
        self._max_path = 0

        self.root = self._create_tree_from_samples(set_of_samples)


    def _create_tree_from_samples(self,set_of_samples):
        """
        From a set of examples given this functions build a tree, called PTA
        (Prefix Tree Acceptor)
        """
        root = Node()
        for sample in set_of_samples:
            tmp = root
            for c in sample:

                if(c != Edge.NONE_CHAR):
                    edge_path = tmp.get_child_by_data(c)

                    if(edge_path != None):
                        tmp = edge_path.child_node
                    else:
                        tmp = self._add_data(c,tmp)
                        self.number_states += 1

        return root

    def goto(self,world):
        tmp = self.root
        tmp.n_in -= 1
        index = 0
        while index < len(world):
            i = 0
            c = world[index:index+i]
            if(c != Edge.NONE_CHAR):
                edge_path = None
                while(edge_path == None):
                    edge_path = tmp.get_child_by_data(c)
                    if(edge_path == None):
                        edge_path = self.root.get_child_by_data(c)
                    i+=1
                    c = world[index:index+i]

                tmp._f_out -= 1
                edge_path.n -= 1

                tmp.reset_freq()
                edge_path.reset_freq()

                tmp = edge_path.child_node
                tmp.n_in -= 1
            index +=i
        return tmp


    def update_tree_statistic_from_samples(self,set_of_samples):
            """
            From a set of examples, this function update the previous tree and
            add statistic in each node/edge (state/transition). At the end
            processus the tree is called PPTA (Probabilistic Prefix Tree Acceptor)
            """
            for sample in set_of_samples:
                print(sample)
                tmp = self.root
                index = 0
                tmp.add_path_in()
                for c in sample:
                    tmp = self._search_next_node(tmp,c)
                    index += 1


    def _search_next_node(self,current_node,data_search):
            """
            Research the next node, this function is used by
            the update_tree_statistic_from_samples to progress in the tree.
            Sometimes it's necessary to create new link between the current node
            and the root
            """
            tmp = current_node
            tmp.add_path_out()
            if(data_search != Edge.NONE_CHAR):
                edge_path = tmp.get_child_by_data(data_search)

                if(edge_path != None):
                    edge_path.add_path_on()
                    tmp = edge_path.child_node
                    tmp.add_path_in()
                    return tmp

                access_root = tmp.get_child_by_data(Edge.NONE_CHAR)
                if(access_root != None):
                    tmp = access_root.child_node
                    access_root.add_path_on()
                else:

                    tmp = self._add_data(data=Edge.NONE_CHAR,
                                        node_parent=tmp,
                                         node_child=self.root,
                                         n=1)

                tmp.add_path_in()
                return self._search_next_node(tmp,data_search)

            return None


    def update_stationary_states(self,node):
        """
        Throught the tree and check each state if this is stationary or not
        """
        is_stationary = (node != self.root and self._check_stationary_state(node))
        if(is_stationary):
            node.is_stationary = is_stationary
            self.number_states_stationnary += 1
        for edge in node.child_edges:
            child_node = edge.child_node
            if(child_node.state > node.state):
                self.update_stationary_states(child_node)


    def _check_stationary_state(self,node):
        """
        Allows to check if a node is stationary or not.
        i.e. : once throught this node it's impossible to root or
        back previous state
        """
        if(len(node.child_edges) > 0):
            return False
        for edge in node.child_edges:
            child_node = edge.child_node
            self.check_stationary_state(child_node)
        return True


    def _add_data(self,data,node_parent,node_child=None,n=0):
        """
        Allows to create new link between state
        """
        if(node_child == None):
            node_child = Node()
        new_edge = Edge(data=data,child_node=node_child,n=n)
        node_parent.add_new_child(new_edge)
        return node_child

    def display_graphviz(self,name,array_colors):
        self.create_draw_graphviz(array_colors)
        self.dot.render(name, view=True)


    def create_draw_graphviz(self,array_colors):
        """
        From the current tree, this function creates a graphvie
        """
        self.dot  = Digraph(comment='graph automata',format="png")
        self.dot.graph_attr['rankdir'] = 'LR'
        self._update_info_graphviz(self.root,array_colors)



    def _update_info_graphviz(self,node,array_colors):
        """
        Subfonction of create_draw_graphviz, that is recursive
        """
        shape_state = 'doublecircle'  if(node.get_path_stay_in() > 0) else 'circle'
        color_state = 'lightgrey' if(not node.is_stationary) else 'blue'
        self.dot.node(
                    name=str(node.state),
                    label=str(node.state) \
                      + " ["+str(node.n_in)+"|"+str(node.get_path_stay_in())+"]",
                    shape=shape_state,
                    color=color_state,
                    style='filled')

        for edge in node.child_edges:
            node_child = edge.child_node

            color_position = 0
            if(self._max_path != 0):
                color_position = int( ((edge.n_random*1.0)/self._max_path)*(STEPS_COLORS_ARRAY-1))
            HSV = array_colors[color_position]

            self.dot.edge(
                    str(node.state),
                    str(node_child.state),edge.data \
                            + " ["+ str(edge.n) +"] "+ str(edge.n_random),
                    color= str(HSV[0]) + ' '+ str(HSV[1]) + ' '+ str(HSV[2]))            #color=str(((100 - edge.n_random)/100))+ ' 0 0')
            if(node_child.state > node.state):
                self._update_info_graphviz(node_child,array_colors)



    def reduce_tree(self,cut_branch_useless=False):
        """
        At the end creation some path are used only one time. If it's the case
        this function reduce on one edge the successive edge.
        """
        self.number_states += 1
        self._reduce_node(self.root,cut_branch_useless)



    def _reduce_node(self,node,cut_branch_useless=False):
        """
        Subfonction of reduce_tree, used recursively
        """

        for edge in node.child_edges:

            node_child = edge.child_node
            number_childs = len(node_child.child_edges)

            if(cut_branch_useless and edge.n == 0):
                node.child_edges.remove(edge)

            elif(number_childs == 1 and
                    (node_child.get_path_stay_in() == 0 or
                     node_child.child_edges[0].data == Edge.NONE_CHAR or
                     len(node_child.child_edges) == 0 ) and False):

                    self._merge_nodes(node,edge)
                    self._reduce_node(node,cut_branch_useless)

            elif(node_child.state > node.state):
                self._reduce_node(node_child,cut_branch_useless)



    def _merge_nodes(self,nodeA,edgeToB):
        """
        Allows to merge 2 nodes, nodeB in the nodeA
        """
        self.number_states -= 1
        nodeB = edgeToB.child_node
        edge_child_nodeB = nodeB.child_edges[0]
        edgeToB.merge(edge_child_nodeB)
        edge_child_nodeB.child_node.merge(nodeB)


    def _merge_himself(self,nodeA,edgeToB):
        nodeB = edgeToB.node_child
        edgeToB.node_child = nodeA
        for edge in nodeB.child_edges:
            nodeA.add_new_child(edge)


    def _child_free(self,node,stack):
        """
        Between child and state already throught this function return child hasn't
        yet discovered
        """
        result = copy_list(node.child_edges)
        number_paths_less = 0
        if(stack != None):
            for edge in node.child_edges:
                for edge_path in stack:
                    if(edge == edge_path):
                        number_paths_less += edge.n
                        result.remove(edge)

        dictionnary = {}
        dictionnary[EDGES_AVAILABLE] = result
        dictionnary[NUMBER_PATHS_LESS] = number_paths_less
        return dictionnary


    def _choose_next_edge(self,node):
        """
        In according probability and state available
        """
        p = rand.random()
        p_check = 0

        for edge in node.child_edges:
            child_node = edge.child_node
            denum = node.current_n_in
            if(denum < 1):
                denum = 1
            p_check += edge.current_n*1.0 / denum
            if(p_check >= p):
                return edge

        return None


    def _reset_node_from_edge_stack(self,stack):
        for edge in stack:
            node = edge.child_node
            edge.reset_freq()
            node.reset_freq()


    def create_random_world(self,stack_save_path=None,node=None):
        """
        Let go an ant to throught the graph based on the probability. At each
        new decision a new probability is computes until find an end pointself.
        return the world created
        """
        node.reset_freq()
        if(node == None):
            node = self.root
        world = ""
        end = False
        node.current_n_in -= 1
        while(not end):
            print("n_in:"+str(node.current_n_in))
            edge_choose = self._choose_next_edge(node)


            if(edge_choose == None):
                 world += "" #"%"
                 break

            else:
                if(stack_save_path != None):
                    edge_choose.current_n -= 1
                    node.current_f_out -= 1
                    stack_save_path.append(edge_choose)

                edge_choose.add_path_on_random()
                if(edge_choose.n_random > self._max_path):
                    self._max_path = edge_choose.n_random

                world += edge_choose.data

                node = edge_choose.child_node
                node.current_n_in -= 1

            end = (len(node.child_edges) == 0 or len(world) > 20)
        #node.current_n_in -= 1
        self._reset_node_from_edge_stack(stack_save_path)
        return world


    def let_go_ants(self,n,node=None):
        sum = 0
        for k in range(n):
            print("ant:"+str(k))
            t = self.create_random_world(None,node)
            sum += len(t)
        avg = sum*1.0/n
        return avg

    def prediction(self,current_world,real_world):
        sum = 0
        current_node = tree.goto(current_world)
        coord = world_to_values(real_world,info_data[1],0)
        save_coord = []
        for k in range(NUMBER_ANTS_PREDICTION):
            t = tree.create_random_world([],current_node)
            coord = world_to_values(t,info_data[1],len(current_world))
            save_coord.append(coord[1])
            #plt.plot(coord[0],coord[1],'--')
            sum += len(t)
        avg = sum*1.0/NUMBER_ANTS_PREDICTION

        plt.plot(coord[0],average_matrix(save_coord),label="avg world created")
        coord_ref = world_to_values(real_world,info_data[1],0)
        plt.plot(coord_ref[0],coord_ref[1],'r-',label="reference world")
        Y1,Y2,Y3 = matrix_column_to_plt(matrix_to_column(save_coord))
        plt.plot(coord[0],Y1)
        plt.plot(coord[0],Y2)
        plt.plot(coord[0],Y3)
        plt.legend()
        plt.grid(True)
        plt.show()






# ----------------- Global Definition  -------------------------



def common_string(s1,s2):
    """
    Based on the dynamic programming, the functions return the more important
    substring between s1 and s2.
    """

    N = len(s1) + 1
    M = len(s2) + 1

    matrix = np.zeros((N, M)).astype(int)
    index_max_rep = [0,0]

    for i in range(1,N):
        for j in range(1,M):
            if(s1[i-1] == s2[j-1]):
                matrix[i][j] = matrix[i-1][j-1] + 1
                if (matrix[i][j] > matrix[index_max_rep[0],index_max_rep[1]]):
                    index_max_rep = [i,j]

    len_substring = matrix[index_max_rep[0],index_max_rep[1]]
    begin_substring_s1 = index_max_rep[0] - len_substring
    begin_substring_s2 = index_max_rep[1] - len_substring

    substring = s1[begin_substring_s1:begin_substring_s1+len_substring]

    dict_substring = {}
    dict_substring[BEGIN_S1] = begin_substring_s1
    dict_substring[BEGIN_S2] = begin_substring_s2
    dict_substring[SUBSTRING] = substring

    return dict_substring



def optimize_word_to_subword(set_of_samples,index):
    """
    In according a set of example and a example specify (index position),
    Based on common substring, this function allows to find sub set of
    exemple more optimize in the PPTA build.
    """
    substring_begin_my_sample = -1
    substring_begin_sample_chosen = -1
    substring_string  = ""
    substring_sample_chosen_position = -1

    sample_fixed = set_of_samples[index]

    dict = None

    for i in range(len(set_of_samples)): # loop inside the set of examples

        if(i != index):
            sample = set_of_samples[i]

            dict_substring = common_string(sample_fixed,sample)
            tmp_substring_begin_my_sample = dict_substring[BEGIN_S1]
            tmp_substring_begin_sample_chosen = dict_substring[BEGIN_S2]
            tmp_substring_string = dict_substring[SUBSTRING]

            is_more_longer = len(tmp_substring_string) > len(substring_string)
            is_same_longer = (len(tmp_substring_string) == len(substring_string))
            is_best_position = tmp_substring_begin_sample_chosen < substring_begin_sample_chosen

            # the substring chosed has updated if the length is more important
            if((is_more_longer) or (is_same_longer and is_best_position)):

                substring_begin_my_sample = tmp_substring_begin_my_sample
                substring_begin_sample_chosen = tmp_substring_begin_sample_chosen
                substring_string = tmp_substring_string
                substring_sample_chosen_position = i

                dict = dict_substring

    # the new sample has checked if it's useless or not to add it
    if(len(substring_string) >= LIMIT_LEN_SUBSTRING):

        if(index > substring_sample_chosen_position):
            word1_remove = set_of_samples.pop(index)
            word2_remove = set_of_samples.pop(substring_sample_chosen_position)
        else:
            word2_remove = set_of_samples.pop(substring_sample_chosen_position)
            word1_remove = set_of_samples.pop(index)


        set_of_samples.append(substring_string)

        if(substring_begin_sample_chosen > 0):
            set_of_samples.append(word2_remove[:substring_begin_sample_chosen])

        if(substring_begin_sample_chosen + len(substring_string) < len(word2_remove)):
            if(word2_remove[substring_begin_sample_chosen+len(substring_string)] == substring_string[0]):
                set_of_samples.append(word2_remove[substring_begin_sample_chosen:])
            else:
                set_of_samples.append(word2_remove[substring_begin_sample_chosen+len(substring_string):])

        if(substring_begin_my_sample > 0):
            set_of_samples.append(word1_remove[:substring_begin_my_sample])

        if(substring_begin_my_sample + len(substring_string) < len(word1_remove)):
            if(word1_remove[substring_begin_my_sample+len(substring_string)] == substring_string[0]):
                set_of_samples.append(word1_remove[substring_begin_my_sample:])
            else:
                set_of_samples.append(word1_remove[substring_begin_my_sample+len(substring_string):])


    return (set_of_samples)


def generate_colors_array():
    """
    From 2 colors given at the top page, this functions returns an degrade colors
    array from HSV1 to HSV2 size of STEPS
    """
    array_colors = []
    r1, g1, b1 = HSV1
    r2, g2, b2 = HSV2
    rdelta, gdelta, bdelta = (r2-r1)/STEPS_COLORS_ARRAY, (g2-g1)/STEPS_COLORS_ARRAY, (b2-b1)/STEPS_COLORS_ARRAY
    for step in range(STEPS_COLORS_ARRAY):
        r1 += rdelta
        g1 += gdelta
        b1 += bdelta
        array_colors.append((r1, g1, b1))
    return array_colors


def generator(length_list,length_world):
    """
    generate randomly a set of samples
    """
    I = []
    world = ""
    for i in range(length_list):
        world = ''.join([rand.choice(ALPHABET_TEST) for n in xrange(length_world)])
        I.append(world)
    return I


def copy_list(list):
    """
    In the aim to avoid side effect, it's safe to copy value in new array
    """
    new_list = []
    for e in list:
        new_list.append(e)
    return new_list


def optimization_samples(I,n):
    """
    From samples given, this function returns a set of samples better optimize
    based on patterns similar
    """
    Iprime = copy_list(I)
    for k in range(n):
        index = 0
        while index < len(Iprime):
            Iprime=optimize_word_to_subword(Iprime,index)
            index+=1
    return Iprime


def demo(I):
    """
    For a set of sample given, apply the process and show the graph got
    """
    Ior = copy_list(I)
    tree = Tree(Ior)
    tree.update_tree_statistic_from_samples(Ior)
    tree.display_graphviz(PATH_SRC_FOLDER+"/PPTA_Ior",array_colors)

    Iprime = optimization_samples(Ior,NUMBER_ANTS)
    tree = Tree(Iprime)
    tree.display_graphviz(PATH_SRC_FOLDER+"/PTA_Iprime",array_colors)
    tree.update_tree_statistic_from_samples(Ior)
    tree.display_graphviz(PATH_SRC_FOLDER+"/PPTA_Iprime",array_colors)
    print(tree.number_states)
    #display_graphviz(tree,PATH_SRC_FOLDER+"/Optimization PPTA")

    tree.reduce_tree(cut_branch_useless=True)
    tree.update_stationary_states(tree.root)
    avg_world = tree.let_go_ants(NUMBER_ANTS)
    tree.display_graphviz(PATH_SRC_FOLDER+"/Reduce Optimization PPTA",array_colors)


def stats_per_samples(I):
    """
    From a set of samples I given, return stats shown information about this way
    to optimize the initial PPTA
    """
    Ior = copy_list(I)

    tree = Tree(Ior)
    number_states_before_opti = tree.number_states

    Iprime = optimization_samples(Ior,NUMBER_ANTS)
    tree = Tree(Iprime)
    tree.update_tree_statistic_from_samples(Ior)
    tree.reduce_tree(cut_branch_useless=True)
    number_states_after_opti = tree.number_states

    tree.update_stationary_states(tree.root)
    number_states_stationnary = tree.number_states_stationnary

    avg_world = tree.let_go_ants(NUMBER_ANTS)

    stats = {}
    stats[NUMBER_BEFORE_OPTI] = number_states_before_opti
    stats[NUMBER_AFTER_OPTI] = number_states_after_opti
    stats[NUMBER_STATES_STATIONARY] = number_states_stationnary
    stats[LEN_AVERAGE_WORLD_CREATE_BY_ANTS] = avg_world

    return stats



def create_stats(length_list,length_word,size_stats):
    """
    Generate several sample in according arguments given and plot a charts with
    results from each sample
    """
    x = []
    y_bef_opti = []
    y_aft_opti = []
    y_states_sta = []
    y_states_percent = []
    y_len_world_avg = []
    y_delta_avg_word_sample = []

    for i in range(size_stats):
        print("####start:"+str(i+1)+"####")
        I = generator(length_list,length_word)
        dict = stats_per_samples(I)
        x.append(i)
        y_bef_opti.append(dict[NUMBER_BEFORE_OPTI])
        y_aft_opti.append(dict[NUMBER_AFTER_OPTI])
        y_states_sta.append(dict[NUMBER_STATES_STATIONARY])

        percent = int(dict[NUMBER_STATES_STATIONARY])*1.0 / int(dict[NUMBER_AFTER_OPTI]) * 100
        y_states_percent.append(percent)
        y_len_world_avg.append(dict[LEN_AVERAGE_WORLD_CREATE_BY_ANTS])
        y_delta_avg_word_sample.append((abs(int(dict[LEN_AVERAGE_WORLD_CREATE_BY_ANTS]) - length_word)))

    plt.plot(x,y_bef_opti,'r-',label='Number states PPTA Ior')
    plt.plot(x,y_aft_opti,'b-',label='Number states PPTA after opti')
    plt.plot(x,y_states_sta,'g-',label='Number stationary states')
    plt.plot(x,y_states_percent,'b--',label='Percent of stationary states')
    plt.plot(x,y_len_world_avg,'g--',label='Len world average (gene ants)')
    plt.plot(x,y_delta_avg_word_sample,'r--',label='Delta world (sample <-> avg ant)')
    plt.xlabel('test number')
    plt.grid(True)

    plt.legend()
    plt.show()


class Stats:

    @staticmethod
    def moyenne(tableau):
        return sum(tableau, 0.0) / len(tableau)

    @staticmethod
    def variance(tableau):
        m=Stats.moyenne(tableau)
        return Stats.moyenne([(x-m)**2 for x in tableau])

    @staticmethod
    def ecartype(tableau):
        return Stats.variance(tableau)**0.5

    @staticmethod
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0*np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
        return m, m-h, m+h



# ----------------- Main -------------------------

# Some set of samples used during the developpement
I1 = ["acdd","er","aze","zeret","sdfsdfsdf","qksljdlkqsjkdqqdsq","zavezvaszn","dsfhskdjfs","qsdqsdqsdqsqsdq"]
I2 = ['dceebc', 'bedded', 'eebabd', 'abcaeb', 'ebbaca', 'ccdacd', 'abdcbd', 'ceebde', 'aabeea', 'dcedcb']
I3  = ["abcd","bcef","ghbc"]
I4 = ["#","#","#","#","#","#","#","#","#","110","0","00","00","10110","100"]
I5 = ["bcacdcdcdcab","bbcadddccbbbaa","babadbabbdcc"]#,"0","00","00"]
I6 =["a","bb","bba","baab","baaaba"]
I7 = ["abcd","ghcde"]
I8 = ['dbdcbcaeaeecceaacaae', 'cdeaabdaeaceebabbacc', 'ceaaeeedbcbbbbdceceb', 'aeebecbecccbeaabbdad', 'accadaaabaecbcccdebb', 'cecbdcceabbaaebeadeb', 'bddbacaeaeacbaeacdbe', 'cdabceadaeecacadeede', 'cababcecacebcbacbcea', 'baabcebbedcaabbebabe']
I9 = ['edeoc','ededc', 'edcba']
I10 = ['aadaa', 'dadad', 'bdecc', 'abccb', 'baddb']
I11 = ['aeabd', 'ccaeb', 'abaac', 'adccb', 'aeacb']
I12 = ['iigbcdifeh', 'jfdbbgaeai', 'jcghfibjff', 'gedbffcbbf', 'adbfbijdbc', 'edbfefgfjc', 'cdibjagida', 'ghejddggac', 'ehgajfahfg', 'gbfdiaidhf']
I13 = ['cejbehfejhfdgdifieccaiadabfhfh', 'cbjchggjhgecbihjddjajgajhbbhbe', 'jiijfiejicijghcecgdfcahgicahea', 'eahdjjibhbacfhjdbfhfjeeahbdhgf', 'aehfjagffifjghdaaghahecdifdhah', 'fibfieihfgjadiibgdbdgfbjfjihgd', 'fcffdfibgacddhidjagicfbjcfegcf', 'cbbdbhfjchahffeabjheeigicfccij', 'aajaeccbjjghdbedadegdgiffjjhag', 'hfhjjcffgdadcjefghcifdcgadddba']


data = [[0.0, 0, 0], [0.0, 0, 1], [0.0, 20, 2], [0.0, 0, 3], [0.5, 0, 0], [0.5, 0, 1], [0.5, 16, 2], [0.5, 0, 3], [1.0, 0, 0], [1.0, 0, 1], [1.0, 16, 2], [1.0, 0, 3], [1.5, 0, 0], [1.5, 0, 1], [1.5, 16, 2], [1.5, 0, 3], [2.0, 0, 0], [2.0, 0, 1], [2.0, 12, 2], [2.0, 0, 3], [2.5, 4, 0], [2.5, 4, 1], [2.5, 12, 2], [2.5, 0, 3], [3.0, 4, 0], [3.0, 4, 1], [3.0, 12, 2], [3.0, 0, 3], [3.5, 4, 0], [3.5, 4, 1], [3.5, 12, 2], [3.5, 0, 3], [4.0, 4, 0], [4.0, 4, 1], [4.0, 12, 2], [4.0, 0, 3], [4.5, 4, 0], [4.5, 4, 1], [4.5, 12, 2], [4.5, 4, 3], [5.0, 4, 0], [5.0, 4, 1], [5.0, 12, 2], [5.0, 4, 3], [5.5, 4, 0], [5.5, 4, 1], [5.5, 8, 2], [5.5, 4, 3], [6.0, 8, 0], [6.0, 4, 1], [6.0, 8, 2], [6.0, 4, 3], [6.5, 8, 0], [6.5, 4, 1], [6.5, 8, 2], [6.5, 4, 3], [7.0, 8, 0], [7.0, 8, 1], [7.0, 8, 2], [7.0, 4, 3], [7.5, 8, 0], [7.5, 8, 1], [7.5, 8, 2], [7.5, 4, 3], [8.0, 8, 0], [8.0, 8, 1], [8.0, 8, 2], [8.0, 8, 3], [8.5, 8, 0], [8.5, 8, 1], [8.5, 8, 2], [8.5, 8, 3], [9.0, 8, 0], [9.0, 8, 1], [9.0, 8, 2], [9.0, 8, 3], [9.5, 8, 0], [9.5, 8, 1], [9.5, 8, 2], [9.5, 8, 3], [10.0, 8, 0], [10.0, 8, 1], [10.0, 8, 2], [10.0, 8, 3], [10.5, 8, 0], [10.5, 8, 1], [10.5, 8, 2], [10.5, 8, 3], [11.0, 8, 0], [11.0, 8, 1], [11.0, 8, 2], [11.0, 8, 3], [11.5, 8, 0], [11.5, 8, 1], [11.5, 8, 2], [11.5, 8, 3], [12.0, 8, 0], [12.0, 12, 1], [12.0, 8, 2], [12.0, 8, 3], [12.5, 8, 0], [12.5, 12, 1], [12.5, 8, 2], [12.5, 8, 3], [13.0, 8, 0], [13.0, 12, 1], [13.0, 8, 2], [13.0, 12, 3], [13.5, 8, 0], [13.5, 12, 1], [13.5, 8, 2], [13.5, 12, 3], [14.0, 8, 0], [14.0, 16, 1], [14.0, 8, 2], [14.0, 12, 3], [14.5, 4, 0], [14.5, 16, 1], [14.5, 8, 2], [14.5, 12, 3], [15.0, 4, 0], [15.0, 20, 1], [15.0, 8, 2], [15.0, 12, 3], [15.5, 4, 0], [15.5, 20, 1], [15.5, 4, 2], [15.5, 12, 3], [16.0, 4, 0], [16.0, 20, 1], [16.0, 4, 2], [16.0, 12, 3], [16.5, 4, 0], [16.5, 24, 1], [16.5, 4, 2], [16.5, 12, 3], [17.0, 4, 0], [17.0, 28, 1], [17.0, 4, 2], [17.0, 12, 3], [17.5, 4, 0], [17.5, 28, 1], [17.5, 4, 2], [17.5, 12, 3], [18.0, 0, 0], [18.0, 32, 1], [18.0, 4, 2], [18.0, 12, 3], [18.5, 0, 0], [18.5, 36, 1], [18.5, 0, 2], [18.5, 12, 3], [19.0, 0, 0], [19.0, 40, 1], [19.0, 0, 2], [19.0, 12, 3], [19.5, 0, 0], [19.5, 44, 1], [19.5, 0, 2], [19.5, 12, 3]]



def create_samples_from_data(datas):
    samples = {}
    alphabet = {}
    alphabet_inverse = {}
    next_letter = 97

    for l in datas:

        value = l[1]
        id_function = l[2]

        current_data = alphabet.get(value)
        if(current_data == None):
            current_data = chr(next_letter)
            next_letter+=1
            alphabet[value] = current_data
            alphabet_inverse[current_data] = value


        current_world = samples.get(id_function)
        if(current_world == None):
            samples[id_function] = ""

        samples[id_function] += current_data

    print(alphabet)
    print(alphabet_inverse)
    print("##################")
    return [samples,alphabet_inverse]


def world_to_values(world,alphabet,begin):
    X = []
    values = []
    index = begin
    for c in world:
        if(c != Edge.NONE_CHAR):
            X.append(index)
            values.append(alphabet.get(c))
            index += 1
    for x in range(index-begin,NUMBER_MAX_ESTIMATION):
        X.append(index)
        values.append(0)
        index += 1
    print(len(values))
    return [X,values]


def average_matrix(matrix):
    result = []
    for j in range(NUMBER_MAX_ESTIMATION):
        result.append(0)
        for i in range(len(matrix)):
            result[j] += matrix[i][j]
        result[j] /= len(matrix)
    return result



def matrix_to_column(matrix):
    columns = []
    for j in range(NUMBER_MAX_ESTIMATION):
        column_local = []
        for i in range(len(matrix)):
            column_local.append(matrix[i][j])
        columns.append(column_local)
    return columns


def matrix_column_to_plt(matrix):
    Y_mean = []
    Y_confidence_inf = []
    Y_confidence_sup = []
    for list in matrix:
        result = Stats.mean_confidence_interval(list)
        Y_mean.append(result[0])
        Y_confidence_inf.append(result[1])
        Y_confidence_sup.append(result[2])

    return Y_mean,Y_confidence_inf,Y_confidence_sup


# At the begining the array_colors is computes in according global variable
# available in the top page
array_colors = generate_colors_array()


color_dict = {
             "a": (255,255,255),
             "b": (255,0,255),
             "c": (255,255,0),
             "d": (0,255,255),
             "e": (255,100,100),
             "f": (100,100,255),
             "g": (100,50,255),
             "h": (0,50,0),
             "i": (255,50,100),
             "j": (255,100,255),
             "k": (100,50,255),
             "l": (50,255,100),
             "m": (255,50,255),
             "n": (100,50,50),
             "#": (0,0,0)
             }

freq_dict = {
             "a": 130.81,
             "b": 146.83,
             "c": 164.81,
             "d": 174.61,
             "e": 196.00,
             "f": 220.00,
             "g": 246.94,
             "h": 261.63,
             "i": 293.66,
             "j": 329.63,
             "k": 349.23,
             "l": 392.00,
             "m": 440.00,
             "n": 493.88,
             "#": 32.70
             }

# ----------------- Begin Test Zone -------------------------

info_data = create_samples_from_data(data)
I = info_data[0].values()

current_world = "aaaaaeeeeeeeeeffffffffffddddc"
#aaaaaeeeeeeeeeffffffffffddddccbbbghhijkl
#aaaaaeeeeeeefffffffffffffffffeeeeeeeaaaa
real_world = "aaaaaeeeeeeeeeffffffffffddddccbbbghhijkl"

Ior = copy_list(I)
tree = Tree(Ior)
tree.update_tree_statistic_from_samples(Ior)

Iprime = optimization_samples(Ior,NUMBER_ANTS)
tree = Tree(Iprime)
tree.update_tree_statistic_from_samples(Ior)
tree.update_stationary_states(tree.root)
tree.reduce_tree(cut_branch_useless=True)
tree.prediction(current_world,real_world)


# ----------------- End Test Zone -------------------------




if(FUNCTION_TO_START == DEMO):
    IN = generator(NUMBER_WORDS,LEN_WORLD)
    demo(IN)
elif(FUNCTION_TO_START == STATS):
    create_stats(NUMBER_WORDS,LEN_WORLD,NUMBER_TESTS)
else:
    print("ERROR: function not found")
