#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

int main(){
    int n;
    cin>>n;
    while(n--){
        vector<int> a;
        int x,y,res=0;
        cin>>x>>y;
        while(x>0){
            a.push_back(x%2);
            x/=2;
        }
        while(y--){
            a.push_back(0);
            int i=0;
            while(a[i]+1>=2){
                res++;
                a[i]=0;
                i++;
            }
            a[i]++;
            res++;
        }
        cout<<res<<endl;
    }
    return 0;
}