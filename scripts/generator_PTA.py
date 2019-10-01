


class Node:

    NONE_CHAR = '#'

    def __init__(self, data,inputs=1,outputs=0):
        self.data = data
        self.children = []
        self.n = inputs;
        self.number_of_outputs = outputs;

    def add_child(self, obj):
        self.children.append(obj)

    def add_input(self):
        self.n += 1

    def get_children(self,data_search):
        self.number_of_outputs += 1
        new_child = self
        if(data_search != Node.NONE_CHAR):
            for n in self.children:
                if(n.data == data_search):
                    n.add_input()
                    return n
            new_child = Node(data_search)
            self.add_child(new_child)
        return new_child

    def get_freq(self):
        return self.n - self.number_of_outputs


    def __str__(self, level = 0):
        ret = "\t="*level+'> ('+repr(self.data) \
            +") ["+ str(self.n)+"/" \
            +str(self.n - self.number_of_outputs)+"]\n"

        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def __repr__(self):
        return '<tree node representation>'







class Tree:
    def __init__(self,set_of_samples):
        self.root = Node(None,inputs=0)
        self.add_samples(set_of_samples);

    def add_samples(self,set_of_samples):
        for i in set_of_samples:
            self.add_word_to_tree(i)

    def add_word_to_tree(self,sample):
        current_node = self.root
        for l in sample:
            current_node = current_node.get_children(l)

    def __str__(self):
        return str(self.root)




class Alergia:
    def __init__(self,tree):
        self.tree = tree

    def is_hoeffding(self,nodeA,nodeB,alpha=0.85):
        n = nodeA.n
        f = nodeA.get_freq();
        np = nodeB.n
        fp = nodeB.get_freq();

        abs = math.abs((f/n) - (fp/np))
        root = math.sqrt(1/2*math.log(2/alpha))
        sum  = (1/math.sqrt(n))+(1/math.sqrt(np))

        hoeffding = root*sum
        print("abs:"+str(abs)+" hoef:"+hoeffding)
        return abs < hoeffding


listTot=[]
with open("D:/document/ESILV/parcoursRecherche/random/Recup.txt",'r') as f:
    fichier_entier= f.read()
    files=fichier_entier.split("\n")

#I = ["#","#","poml","dfe","vbc","ert"]
I = ["abcb","bcba","e","abc"]
tree = Tree(files)

print(tree)
