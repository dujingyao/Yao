#include<iostream>
#include<vector>
using namespace std;
typedef struct treenode{
    int info;
    int before;
}treenode;
int n;
int main(){
    cin>>n;
    treenode tree[n+2];
    for(int i=1;i<n+1;i++){
        cin>>tree[i].info;
        tree[i].before=-1;
    }
    int k=n-1;
    while(k--){
        int i,j;
        cin>>i>>j;
        tree[j].before=i;
    }
    int sum=0;
    int max=sum;
    vector<int> a;
    while(n){
        sum=0;
        for(int x=n;x>=1;x--){
            int flag=0;
            if(a.size()>0){
                for(int i=0;i<a.size();i++){
                    if(a[i]==x) flag=1;
                }
            }
            if(flag==1) continue;
            sum+=tree[x].info;
            a.push_back(tree[x].before);;
        }
        if(sum>max) max=sum;
        n--;
    }
    cout<<max<<endl;
    return 0;
}