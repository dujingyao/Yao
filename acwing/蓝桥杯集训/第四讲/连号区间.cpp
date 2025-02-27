#include<bits/stdc++.h>
using namespace std;

int n;
vector<int> a,b;

int main(){
    //本身就包含每一个数，不重不漏
    cin>>n;
    for(int i=1;i<=n;i++){
        int x;
        cin>>x;
        a.push_back(x);
    }
    int res=0;
    for(int i=0;i<n;i++){
        int max1=0,min1=11000;
        for(int j=i;j<n;j++){//
            max1=max(a[j],max1);
            min1=min(a[j],min1);
            if(max1-min1==j-i) res++;
        }
    }
    cout<<res<<endl;

    return 0;
}