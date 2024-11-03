#include<stdio.h>
int sumOddLengthSubarrays(int* arr, int arrSize){
    int sum=0;
    int i,j;
    for(i=0;i<arrSize;i++){
        for(j=1;j<=arrSize-i;j+=2){
            for(int m=i;m<i+j;m++){
                sum+=arr[m];
            }
        }
    }
    return sum;
}
int main(){
    int a[3]={1,2,3};
    int x=sumOddLengthSubarrays(a,3);
    printf("%d",x);
    return 0;
}