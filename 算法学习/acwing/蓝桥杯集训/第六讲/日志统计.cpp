#include<bits/stdc++.h>
using namespace std;

const int N=1e5+10;
typedef pair<int,int> PII;
PII a[N];
int cnt[N];//计数
bool check[N];//判断是否达到标准

int main(){
    int n,d,k;
    cin>>n>>d>>k;
    for(int i=0;i<n;i++) cin>>a[i].first>>a[i].second;
    //双指针，i在前,j在后
    for(int i=0,j=0;i<n;i++){
        int t=a[i].second;
        cnt[t]++;
        while(a[i].first-a[j].first>=d){
            cnt[a[j].second]--;
            j++;
        }
        if(cnt[t]==k) check[i]=true;
    }
    for(int i=1;i<=10000;i++){
        if(check(i)) cout<<i<<endl;
    }
    return 0;
}