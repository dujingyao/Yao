#include<bits/stdc++.h>
using namespace std;

const int N=1e5+10;

typedef long long ll;

int num[4][N];

int main(){
    
    int n;
    cin>>n;
    for(int i=1;i<=3;i++){
        for(int j=1;j<=n;j++){
            cin>>num[i][j];
        }
    }
    ll res=0;
    //排序
    for(int i=1;i<=3;i++){
        sort(num[i]+1,num[i]+n+1);
    }
    //枚举b，寻找a满足的个数
    int a=1,c=1;
    for(int i=1;i<=n;i++){
        int key=num[2][i];
        while(a<=n&&num[1][a]<key) a++;
        while(c<=n&&num[3][c]<=key) c++;

        res+=(ll)(a-1)*(n-c+1);
    }

    cout<<res<<endl;

    return 0;
}