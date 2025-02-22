#include<iostream>
using namespace std;

const int N=1e6+10;

int a[N],n;//输入
int ct1[N];//数字i在数组中出现的次数
int ct2[N];//倒叙
int uct[N];//前i个数字中不同数字的个数

long long int ans;

int main(){
    cin>>n;
    cin>>a[0];
    ct1[a[0]]++;
    uct[0]=1;
    for(int i=1;i<n;i++){
        cin>>a[i];
        uct[i]=uct[i-1];
        ct1[a[i]]++;
        if(ct1[a[i]]==1) uct[i]++;
    }

    for(int i=n-1;i>=0;i--){
        ct2[a[i]]++;
        if(ct2[a[i]]==2){
            ans+=uct[i-1];
            if(ct1[a[i]]>2) ans--;
        }
    }
    cout<<ans<<endl;

    return 0;
}