# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 14:15:55 2019

@author: victor
"""

#!/usr/bin/env python

import matplotlib.pyplot as plt
import math
import sys
import numpy as np
import random as rd
sys.setrecursionlimit(10000)  # permet d'augmenter la limite de stack récursive

DEBUG = True
def DISPLAY(tag,value):
    if(DEBUG):
        print(tag+":"+str(value))

class Node:  # Classe basique noeud
        NONE_CHAR = '#'
        _COUNTER_NODE = 0

        def __init__(self,number_inputs = 0):
            self.child_edges = []
            self.n = number_inputs
            self.f_out = 0
            self.f_int = 0
            self.state = Node._COUNTER_NODE
            Node._COUNTER_NODE += 1

        def add_child_edge(self,edge):
            self.child_edges.append(edge)
            #self.child_edges.sort(key=lambda x: x.data, reverse=False)


        def _add_child(self,new_branch_data):
            edge = Edge(new_branch_data)
            self.add_child_edge(edge)
            return edge

        def get_freq(self):
            return self.n - self.f_out

        def get_children_creation(self,data_search):
            if(data_search == Node.NONE_CHAR):
                return self
            self.f_out += 1
            for edge in self.child_edges:
                if(edge.data == data_search):
                    return edge.get_node_creation()

            new_edge = self._add_child(data_search)
            return new_edge.get_node_creation()

        def get_children(self,data_search):
            for edge in self.child_edges:
                if(edge.data == data_search):
                    return edge
            return None

        def is_own_child(self,node):
            for edge in self.child_edges:
                if(node == edge.child_node):
                    return True
            return False

        def sum_input_from_node(self,node):
            self.n += node.n
            self.f_out += node.f_out

        def remove_edge(self,edge):
            self.child_edges.remove(edge)

        def research_node(self,node):
            for edge in self.child_edges:
                if(edge.child_node == node):
                    return edge
            return None

        def str_child(self):
            string = ""
            for edge in self.child_edges:
                string += str(edge)+" "
            return string


        def __str__(self, level = 1):  
            ret = '-> ('+repr(self.state) \
                +")"+str(self.f_int)

            for child_edge in self.child_edges:
                if(child_edge.child_node == self):
                    ret += " <--"+ str(child_edge.n) +"-->"

           

            ret += "\n"

            for child_edge in self.child_edges:
                if(child_edge.child_node != self):
                    ret += "\t"*level+"-"+str(child_edge.n)
                    if(child_edge.child_node.state > self.state):
                        ret += child_edge.child_node.__str__(level+1)

            return ret

        def __repr__(self):
            return '<tree node representation>'

        def to_Markov(self, node):  # Transformation de l'arbre en HMM.
            a=(len(node.child_edges)+1)
            print(str(len(node.child_edges)))
            
            node.f_int=[0]*(a)  # Creation liste vide
            for j in range(a):
                node.f_int[j]=[0]*2    # creation liste imbriquée
            node.f_int[a-1][0]='#'     #creation de l'élement arret
            for i in range (0,len(node.child_edges)):
            
                node.f_int[i][0]=str(node.child_edges[i].data)   # remplissage des data
                node.f_int[i][1]=((node.child_edges[i].n)/(node.n))  # remplissage des probas d'apparition
                
                if(node.get_freq()!=0):
                    node.child_edges[i].n=(node.child_edges[i].n)/(node.n-node.get_freq())  # remplissage des probas
                else:
                    node.child_edges[i].n=(node.child_edges[i].n)/(node.n)  
                print(node.child_edges[i].n)
                
            if(node.get_freq()!=0):
                node.f_int[a-1][1]=node.get_freq()/node.n
                
            else:
                node.f_int[a-1][1]=0
            print(node.f_int)
            for k in node.child_edges:
                if(k.child_node==node):  # cas d'une boucle.
                    return;
                else:
                    self.to_Markov(k.child_node)  #appel recursif
        


        def antman(self,node):   #Une fourmie : deplacement aleatoire
            fichier_ant=open('antman.txt',"a")
            
            a=rd.uniform(0,1)
            print(a)
            i=0
            test=1
            sum=0
            
            if(a<node.f_int[i][1]):
                if(node.f_int[i][0]=="#"):  # cas d'un arret
                  fichier_ant.write(node.f_int[i][0])  # écrite dans le fichier 
                  return;    # stop
                else:   #  non-arrêt
                    fichier_ant.write(node.f_int[i][0])
                    test=0
            else:
                
                sum+=node.f_int[i][1]
                i=1
   
            while((test==1)&(i<len(node.child_edges))):
                if(node.f_int[i][0]=="#"):
                   fichier_ant.write(node.f_int[i][0])
                   
                   return;
                   
                if(a<sum +node.f_int[i][1]):
                    fichier_ant.write(node.f_int[i][0])
                    test=0
                else:
                    sum=sum +node.f_int[i][1]
                    i+=1
            b=rd.uniform(0,1)
            j=0
            sum2=0
            test2=1
            if(b<node.child_edges[j].n):  # premier cas 
                self.antman(node.child_edges[j].child_node)   # appel recursif 
                test2=0
            else:
                sum2+=node.child_edges[j].n
                j=1
            while((test2==1)&(j<len(node.child_edges))):  # autre cas
                if(b<sum2+node.child_edges[j].n):
                    
                    self.antman(node.child_edges[j].child_node)
                    print(node.child_edges[j].child_node.state)
                    test2=0
                else:
                    sum2=sum2+ node.child_edges[j].n
                    j+=1
           
            
            
                
            
class Edge:   # Classe basique de liaison entre les noeuds.
    def __init__(self,data='#',child=None,edge=None):
        self.child_node = None

        if(edge == None):
            self.n = 0
            self.data = data
            self.back = False
        else:
            self.n = edge.n
            self.data = edge.data
            self.back = True

        if(child == None):
            self._add_child();
        else:
            self.child_node = child

    def _add_child(self):
        new_child_node = Node()
        self.child_node = new_child_node

    def get_node_creation(self):
        self.n += 1
        self.child_node.n += 1
        return self.child_node

    def get_node(self):
        return self.child_node

    def sum_input_from_edge(self,edge):
        self.n += edge.n

    def __str__(self):
        return self.data+"["+str(self.n)+"]"



class Tree:            # Class basique d'arbre et fonction d'ajout, d'initialisation.
    def __init__(self,set_of_samples):
        self.root = Node(number_inputs = len(set_of_samples))
        self.add_samples(set_of_samples);

    def add_samples(self,set_of_samples):
        DISPLAY("set_of_samples",set_of_samples)
        for i in set_of_samples:
            self.add_word_to_tree(i)

    def add_word_to_tree(self,sample):
        current_node = self.root
        for l in sample:
            current_node = current_node.get_children_creation(l)



class Alergia:

    ALPHA_DEFAULT =10*(1/300)

    def __init__(self,tree):
        self.tree = tree
        self.number_of_states = 0
        self.number_end_paths = 0

    def is_hoeffding(self,f,n,fp,np,alpha=ALPHA_DEFAULT):   # CALCUL D'HOEFFDING
        f = f * 1.0
        fp = fp * 1.0
        abs = math.fabs((f/n) - (fp/np))
        root = math.sqrt(1.0/2*math.log(2.0/alpha))
        sum  = (1/math.sqrt(n))+(1/math.sqrt(np))
        hoeffding = root*sum
        #DISPLAY("abs",abs)
        #DISPLAY("root",root)
        #DISPLAY("sum",sum)
        #DISPLAY("hoef",hoeffding)
        return abs < hoeffding

    def is_hoeffding_transition(self,nodeA,nodeB,edgeA,edgeB,alpha=ALPHA_DEFAULT):
        f = edgeA.n if edgeA != None else 0
        n = nodeA.n
        fp = edgeB.n if edgeB != None else 0
        np = nodeB.n
        return self.is_hoeffding(f,n,fp,np,alpha)

    def is_hoeffding_state(self,nodeA,nodeB,alpha=ALPHA_DEFAULT):   # calcul des paramètres
        n = nodeA.n
        f = nodeA.get_freq()
        np = nodeB.n
        fp = nodeB.get_freq()
        return self.is_hoeffding(f,n,fp,np,alpha)


    def is_compatible(self,nodeA,nodeB,alpha=ALPHA_DEFAULT):   # Fonction permettant de faire les test de compatibilité de fusion
        state_compatible = self.is_hoeffding_state(nodeA,nodeB,alpha)
        if(state_compatible):
            for data in self.states_in_commun(nodeA,nodeB):
                edgeA = nodeA.get_children(data);
                edgeB = nodeB.get_children(data);
                compatible = self.is_hoeffding_transition(nodeA,nodeB,
                                                                edgeA,edgeB,alpha)  # APPEL du test d'Hoeffding 
                if(compatible):
                    nextNodeA = edgeA.child_node if edgeA != None else nodeA
                    nextNodeB = edgeB.child_node if edgeB != None else nodeB
                    self.is_compatible(nextNodeA,nextNodeB,alpha)

                else:
                    return False

            return True
        else:
            return False

    def states_in_commun(self,nodeA,nodeB):   # Algo qui permet de mettre en évidence les états en communs de deux noeuds.
        data_list = []
        for edgeA in nodeA.child_edges:
            for edgeB in nodeB.child_edges:
                if(edgeA.data == edgeB.data):
                    data_list.append(edgeA.data)
        return set(data_list)



    def break_link(self,nodeA,parent_node_B,edge_link):
        parent_node_B.remove_edge(edge_link)
        edge_link = Edge(child=nodeA,edge=edge_link)
        parent_node_B.add_child_edge(edge_link)


    def merge_node(self,node_tmp_A,node_tmp_B):         # Process de merge des noeuds : on recupère l'entrant et le sortant de chaque.
        node_tmp_A.sum_input_from_node(node_tmp_B)
        for data in self.states_in_commun(node_tmp_B,node_tmp_A):
            edgeA = node_tmp_A.get_children(data);
            edgeB = node_tmp_B.get_children(data);
            if(edgeA != None and edgeB != None):
                edgeA.sum_input_from_edge(edgeB)
                node_tmp_B.remove_edge(edgeB)
                node_tmp_A = edgeA.child_node
                node_tmp_B = edgeB.child_node
                self.merge_node(node_tmp_A,node_tmp_B)



    def merge_lists(self,listA,listBlack):   # Merge des lists pour les fusion d'état.
        list_final = []
        for elem in listA:
            sum_seen = 0
            for elem_black in listBlack:
                if(elem == elem_black):
                    sum_seen += 1
                    break
            if(sum_seen == 0):
                list_final.append(elem)
        return list_final


    def process(self,node,parent_node,alpha=ALPHA_DEFAULT):   # process ALERGIA/MERGING
         for edge in parent_node.child_edges:
             node_child = edge.child_node
             if(not edge.back):
                 if(self.is_compatible(node,node_child,alpha)):
                     DISPLAY("Compatible","done")
                     self.break_link(node,parent_node,edge)
                     self.merge_node(node,node_child)
                 print(self.tree.root)

                 self.process(node_child,node_child)
                 self.process(node,node_child)





    def evaluate_number_states(self,node):              # Parcours permettant de calculer le nombre d'état total
          self.number_of_states += 1
          for edge in node.child_edges:
              node_child = edge.child_node
              if(not edge.back):
                self.evaluate_number_states(node_child)

    def evaluate_number_end_paths(self,node):           # Fonction permettant de calculer le nombre d'état finaux: check valeur back
          self.number_end_paths += node.get_freq()
          for edge in node.child_edges:
              node_child = edge.child_node
              if(not edge.back):
                self.evaluate_number_end_paths(node_child)







    def evaluate_perform(self,I):   # Calcul du nombre d'états en fonction d'alpha.
        X_alpha = []
        Y_nb_paths_end = []
        Y_nb_states = []
        for a in range(1,100):
            tree_tmp = Tree(I)
            print(tree_tmp.root)
            alpha = a*1.0/100
            print(alpha)
            root = tree_tmp.root
            alergia.process(root,root,alpha)
            alergia.evaluate_number_states(root)
            alergia.evaluate_number_end_paths(root)

            X_alpha.append(alpha)
            Y_nb_states.append(self.number_of_states)
            Y_nb_paths_end.append(self.number_end_paths)

            self.number_end_paths = 0
            self.number_of_states = 0

        plt.plot(X_alpha,Y_nb_paths_end,'-r')
        plt.plot(X_alpha,Y_nb_states,'+b')
        #plt.plot(X_alpha,[len(I)]*len(X_alpha),'--g')
        plt.show()




class Home_Opitmization:

        def __init__(self,tree):
            self.tree = tree





def step_by_step():    # Alergia étape par étape  Je n'utilisais pas cette fonction. Intéressante pour comprendre comment allergia opère.
    nodeA = tree.root
    nodeB = tree.root.child_edges[1].child_node

    #alergia.process(tree.root,tree.root)


    print(alergia.is_compatible(nodeA,nodeB))

    alergia.break_link(nodeA,nodeA,tree.root.child_edges[1])
    alergia.merge_node(nodeA,nodeB)

    nodeBB_parent = tree.root.child_edges[0].child_node
    nodeBB_edge = nodeBB_parent.child_edges[1]
    nodeBB = nodeBB_edge.child_node

    print(tree.root)


    print(alergia.is_compatible(nodeA,nodeBB))

    alergia.break_link(nodeA,nodeBB_parent,nodeBB_edge)
    alergia.merge_node(nodeA,nodeBB)

    nodeAAA =  tree.root.child_edges[0].child_node
    nodeBBB_parent = tree.root.child_edges[0].child_node
    nodeBBB_edge = nodeBBB_parent.child_edges[0]
    nodeBBB = nodeBBB_edge.child_node

    print(alergia.is_compatible(nodeAAA,nodeBBB))

    alergia.break_link(nodeAAA,nodeBBB_parent,nodeBBB_edge)
    alergia.merge_node(nodeAAA,nodeBBB)


    print(tree.root)



 
        
    
# plein de dataset !
#I = ["#","#","#","#","#","#","#","#","#","110","0","00","00","10110","100"]
#I = ["a","bb","bba","baab","baaaba"][]
I = ["abcb","bcba","e","abc"]
#I = ['#', '#', '#', '#', '#', '#', '#', '#', '#', '110', '0', '00', '00', '10', '1']
#I = ['1', '0', '#', '00']
with open("D:/document/ESILV/parcoursRecherche/random/Recup.txt",'r') as f: # recupération et lecture du fichier contenant les datas
    fichier_entier= f.read()
    files=fichier_entier.split("\n")
tree = Tree(files)
print(tree.root)

alergia = Alergia(tree)
alergia.evaluate_perform(I)

#nodeA = tree.root
#nodeB = tree.root.child_edges[1].child_node
#
alergia.process(tree.root,tree.root)
print(tree.root)
alergia.evaluate_number_states(tree.root)
print(alergia.number_of_states)
alergia.evaluate_number_end_paths(tree.root)
print(alergia.number_end_paths)

Node.to_Markov(tree.root, tree.root)
#for i in range (100):
#    Node.antman(tree.root,tree.root)
#    fichier_ant=open('antman.txt','a')
#    fichier_ant.write("\n")
#    fichier_ant.close()
print(tree.root)



#---------------------------------------------------------------------------------------------------------

    
    