#include<iostream>
using namespace std;

int main(){
    int n,q;
    scanf("%d %d",&n,&q);
    int a[n];
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
    }
    while(q--){
        int x;
        scanf("%d",&x);
        int l=0,r=n-1;
        while(l<r){
            int mid=(l+r)/2;
            if(a[mid]>=x) r=mid;
            else l=mid+1;
        }
        if(a[l]==x) printf("%d ",l+1);
        else printf("-1 ");
    }
    return 0;
}