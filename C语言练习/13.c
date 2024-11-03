#include <stdio.h>
int n[10]={6,2,5,5,4,5,6,3,7,6};
int num(int x);
int main(){
    int res[25]={0};
    int c,sum;
    for(int i=0;i<=10000;i++){
        for(int j=0;j<=10000;j++){
            c=i+j;
            sum=num(i)+num(j)+num(c)+4;
            if(sum<=24) res[sum]++;
		}
	}
    int n;
    scanf("%d",&n);
    printf("%d",res[n]);
    return 0;
}
int num(int x){
    int sum=0;
    //if(x==0) return n[0];
    while(x){
        sum+=n[x%10];
        x/=10;
	}
    return (x==0)?n[0]:sum;
}
