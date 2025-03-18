#include<bits/stdc++.h>
using namespace std;
typedef pair<int,int> PII;
const int N=1e6+10;
int t,n,f[N]; // 保存每个建筑的访问次数
PII a[N]; // 存储建筑编号和访问次数
int x[N]; // 存储每个建筑的坐标

bool cmp(PII x, PII y){
    return x.second > y.second; // 降序排序
}

int main(){
    cin>>t;
    while(t--){
        cin>>n;
        for(int i=1;i<=n;i++){
            cin>>f[i];
            a[i] = {i, f[i]};
        }
        x[0] = 0;
        sort(a+1, a+n+1, cmp); // 按访问次数降序排列
        int pos=1;
        for(int i=1;i<=n;i++){
            if(i%2 == 1){
                x[a[i].first] = pos;
            } else {
                x[a[i].first] = -pos;
                pos++;
            }
        }
        long long dis=0;
        for(int j=1;j<=n;j++){ // 遍历每个建筑编号计算总距离
            dis += 2LL * f[j] * abs(x[j]);
        }
        cout<<dis<<endl;
        for(int i=0;i<=n;i++){
            cout<<x[i]<<' ';
        }
        cout<<endl;
    }
    return 0;
}