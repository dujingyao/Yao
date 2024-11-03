#include<stdio.h>

int main(){
    for(int i=100;i<=200;i++){
        int n=0;
        for(int j=2;j<i;j++){
            if(i%j!=0) n++;
        }
        if(i-2==n) printf("%d\n",i);
    }
    return 0;
}