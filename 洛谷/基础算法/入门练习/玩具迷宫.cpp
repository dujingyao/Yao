#include<iostream>
#include<string>
using namespace std;
const int MAX=1e6+5;
struct node{
    int head;
    string name;
}number[MAX];
int main(){
    
    int n,m;
    cin>>n>>m;
    for(int i=0;i<n;i++){
        cin>>number[i].head>>number[i].name;
    }

    int f=0;
    while(m--){
        int x,y;
        cin>>x>>y;
        //头朝外,向左
        if(number[f].head==0&&x==0){
            f=(f+n-y)%n;
        }
        //头朝外,向右
        else if(number[f].head==0&&x==1){
            f=(f+y)%n;
        }
        //头朝内,向左
        else if(number[f].head==1&&x==0){
            f=(f+y)%n;
        }
        //头朝内,向右
        else if(number[f].head==1&&x==1){
            f=(f+n-y)%n;
        }
    }
    cout<<number[f].name<<endl;

    return 0;
}