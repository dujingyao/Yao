#include<stdio.h>

int main(){
    for(int i=1949;i<=2019;i++){
        if(i%4==0&&i%400!=0||i%400==0) printf("%d\n",i);
    }
    return 0;
}