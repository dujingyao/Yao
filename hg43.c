#include<stdio.h>
#include<math.h>
int main(){
    int n,x=0;
    scanf("%d",&n);
    for(int i=2;i<sqrt(n);i++){
        if(n%i==0){
            printf("NO");
            x=1;
            break;
        }
    }
    if(x==0) printf("YES");
    return 0;
}