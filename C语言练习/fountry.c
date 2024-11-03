#include<stdio.h>

int main(){
    int N,Q;
    scanf("%d",&N);
    int a[2][N];
    int i,j;
    for(i=0;i<=1;i++){
        for(j=0;j<N;j++){
            scanf("%d",&a[i][j]);
        }
    }
    scanf("%d",&Q);
    int m,n;
    int f1,f2,sum;
    while(Q--){
        sum=0;
        scanf("%d %d",&m,&n);
        for(i=0;i<N-1;i++){
            if(m>a[0][i]&&a[0][i+1]>m) f1=i;
            if(n>a[0][i]&&a[0][i+1]>m) f2=i;
        }
        for(i=f1;i<=f2;i++){
            sum+=a[1][i];
        }
        printf("%d\n",sum);
    }
    return 0;
}