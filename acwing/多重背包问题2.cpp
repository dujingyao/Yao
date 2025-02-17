#include<iostream>
#include<algorithm>
using namespace std;

const int N=12010,V=2010;
int f[N];
int v[N],w[N];

int main(){
    
    int n,m;
    cin>>n>>m;
    int cnt=0;
    //循环n次
    for(int i=1;i<=n;i++){
        int a,b,s;
        cin>>a>>b>>s;
        //分块
        int k=1;//假定当前选择的物品个数为k个
        while(k<=s){
            cnt++;
            v[cnt]=a*k;
            w[cnt]=b*k;
            s-=k;
            k*=2;
        }
        //若s还有剩余
        if(s>0){
            cnt++;
            v[cnt]=a*s;
            w[cnt]=b*s;
        }
    }
    n=cnt;
    for(int i=1;i<=n;i++){
        for(int j=m;j>=v[i];j--){
            f[j]=max(f[j],f[j-v[i]]+w[i]);
        }
    }
    cout<<f[m]<<endl;
    

    return 0;
}