#include<stdio.h>

int main(){
    int flag=1;
    double sum=0.0;
    for(int i=1,j=2;i<=99;i++,j++){
            sum+=1.0*flag*i/j;
            flag=-flag;
    }
    printf("%.4f",sum);
    return 0;
}