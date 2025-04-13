#include<bits/stdc++.h>
using namespace std;

const int N=1e5+10;

int h[N],backup[N];
int T,n;
//两种情况
//1.两个相同的挨着
//2.两个相同的中间隔了一个
int main(){
    cin>>T;
    while(T--){
        cin>>n;
        vector<int> a;
        for(int i=1;i<=n;i++){
            cin>>h[i];
        }
        if(n==2){
            if(h[1]==h[2]) cout<<h[1]<<endl;
            else cout<<-1<<endl;
            continue;
        }
        for(int i=1;i<=n-2;i++){
            if(h[i]==h[i+1]) a.push_back(h[i]);
            else if(h[i+1]==h[i+2]) a.push_back(h[i+1]);
            else if(h[i]==h[i+2]) a.push_back(h[i]); 
        }
        if(a.empty()){
            cout<<-1<<endl;
        }
        else{
            set<int> b(a.begin(),a.end());
            for(auto it:b){
                cout<<it<<' ';
            }
            cout<<endl;
        }
    
    }
    return 0;
}