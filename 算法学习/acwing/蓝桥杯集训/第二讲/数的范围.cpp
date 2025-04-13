#include<iostream>
using namespace std;

const int N=100010;

int n,q;
int a[N];

int main(){
    cin>>n>>q;
    for(int i=1;i<=n;i++){
        cin>>a[i];
    }
    while(q--){
        int goal;
        cin>>goal;
        int l=1,r=n;
        //找左
        while(l<r){
            int mid=(l+r)/2;
            if(a[mid]>=goal) r=mid;
            else l=mid+1;
        }
        if(a[l]!=goal) cout<<"-1 -1"<<endl;
        else{
            //找右
            cout<<l-1<<' ';
            l=1,r=n;
            while(l<r){
                int mid=(l+r+1)/2;
                if(a[mid]<=goal) l=mid;
                else r=mid-1;
            }
            cout<<l-1<<endl;
        }
    }
    return 0;
}