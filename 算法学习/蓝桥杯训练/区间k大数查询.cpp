#include<iostream>
#include<algorithm>
#include <vector>
using namespace std;
int main(){
    int m,n;
    scanf("%d",&m);
    vector<int> a(m+1);
    for(int i=1;i<=m;i++){
        scanf("%d",&a[i]);
    }
    scanf("%d",&n);
    while(n--){
        int x,y,k;
        scanf("%d %d %d",&x,&y,&k);
        vector<int> b(a.begin()+x,a.begin()+y+1);
        nth_element(b.begin(),b.begin()+k-1,b.end(),greater<int>());
        printf("%d\n",b[k-1]);
    }

    return 0;
}