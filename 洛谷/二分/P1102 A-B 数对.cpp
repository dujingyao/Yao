#include<iostream>
#include<algorithm>
using namespace std;

int main(){
    int m,n;
    scanf("%d %d",&m,&n);
    int a[m];
    for(int i=0;i<m;i++){
        scanf("%d",&a[i]);
    }
    sort(a,a+m);
    int sum=0;
    int j=m-1;
    for(;j>=0;j--){
        int l=0,r=j;
        while(l<r){
            int mid=(l+r)/2;
            if(a[j]-a[mid]==n) sum++;
            if(a[j]-a[mid]<=n) r=mid;
            else l=mid+1;
        }
    }
    printf("%d",sum);
    return 0;
}