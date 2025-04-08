#include<bits/stdc++.h>
using namespace std;
const int N=1e5+10;
priority_queue<int,vector<int>,greater<int>> minHeap;
int n;
int ans;
int main(){
    cin>>n;
    int m;
    //输入并排序
    for(int i=1;i<=n;i++){
        cin>>m;
        minHeap.push(m);
    }
    if(n==1){
        cout<<minHeap.top();
    }
    while(minHeap.size()>1){
        int z=0;
        int x,y;
        x=minHeap.top();
        minHeap.pop();
        y=minHeap.top();
        minHeap.pop();c
        z+=(x+y);
        ans+=z;
        minHeap.push(z);
    }
    cout<<ans;
    return 0;
}