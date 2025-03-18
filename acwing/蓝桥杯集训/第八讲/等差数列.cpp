#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N=1e5+10;
ll a[N];
int n;
//辗转相除法求最大公约数
int gcd(int a,int b){
    while(b){
        int c=a%b;
        a=b;
        b=c;
    }
    return a;
}

int main(){
    cin>>n;
    for(int i=1;i<=n;i++) cin>>a[i];
    if(n==2){
        cout<<2<<endl;
        return 0;
    }
    sort(a+1,a+1+n);
    //求后面各项与第一项的最大公约数
    int d=0;
    for(int i=2;i<=n;i++){
        d=gcd(d,a[i]-a[1]);
    }
    if(!d) cout<<n<<endl;
    //求多少项的公式
    else cout<<(a[n]-a[1])/d+1<<endl;
    return 0;
}