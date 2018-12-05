#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include <queue>
#define N 100
#define NULL 0
#define M 170

using namespace std;

struct Node{
    string ville;
    Node *child[N];
    float proba;
};

Node *newNode(string Ville)
{
    Node *temp= new Node;
    temp->ville=Ville;
    temp->proba=(float)1/M;
    for(int i =0 ;i<N;i++)
    {
        temp->child[i]=NULL;

    }
    return temp;
}
int cont(Node* newN)
{
    int i=0;
    while(newN->child[i]!=NULL)
    {
        i++;
    }
    return i;
}
const vector<string> explode(const string& s, const char& c)
{
	string buff{""};
	vector<string> v;

	for(auto n:s)
	{
		if(n != c) buff+=n; else
		if(n == c && buff != "") { v.push_back(buff); buff = ""; }
	}
	if(buff != "") v.push_back(buff);

	return v;
}
int test(string v ,Node *tree)
{
    int i=0;
    int index=-1;

    while(tree->child[i]!=NULL)
    {

        if(v==tree->child[i]->ville)
        {
            index=i;

        }
        i+=1;
    }
    return index;
}
int ajout(vector<string> v,Node *tree, int n)
{

    if(n==v.size())
    {
        return 0;
    }
    int val=test(v[n],tree);
    if(val!=(-1))
    {
        tree->child[val]->proba+=(float)1/M;
        ajout(v,tree->child[n],n+=1);





    }
    else
    {
        int i=0;
        while(tree->child[i]!=NULL)
        {
            i++;
        }


        tree->child[i]=newNode(v[n]);

        ajout(v,tree->child[i],n+=1);



    }
}

/*void affichage(Node *tree)
{
    Node *neutre=newNode(" ");


    queue<Node*>file;
    queue<Node*>file2;
    queue<int> elem;
    Node *s;
    int i=0;
    while(tree->child[i]!=NULL)
    {
        file.push(tree->child[i]);
        i++;
    }
    elem.push(i);

    while(!file.empty())
    {
        s=file.back();
        file.pop();
        int j=0;
        cout<<s->ville;
        while (s->child[j]!=NULL)
        {
            file2.push(s->child[j]);
        }
        file2.push(neutre);
        cout<<s->ville<<" ";
        if(file.empty())
        {
            file=file2;
             while(!file2.empty())
            {
            file2.pop();
            }
            cout<<"\n";
        }




    }



}*/

void affichage (Node *tree)
{

    int i=0;
    while(tree->child[i]!=NULL)
    {
        cout<<tree->child[i]->ville<<" "<<tree->child[i]->proba<<" || ";
        i++;

    }
    cout<<"\n \n \n";
    int j=0;
    while(tree->child[j]!=NULL)
    {
        affichage(tree->child[j]);
        j++;

    }
}

void lecture_csv()
{
    std::ifstream ifs{"D:/document/ESILV/parcoursRecherche/MadridSpot.csv"};
    std::string tmp{};
    Node* tree=newNode(" ");

    for(int i=0;i<M;i++)
    {
        std::getline(ifs,tmp);

        vector<string> v{explode(tmp, ',')};



        ajout(v,tree,0);
    }
     affichage(tree);
}
int main()
{


    lecture_csv();
}
