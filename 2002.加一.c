#include<stdio.h>
int* plusOne(int* digits, int digitsSize, int* returnSize) {
    int i;
    for(i=digitsSize-1;i>=0;i--){
        if(digits[i]<9){
            digits[i]++;
            *returnSize=digitsSize;
            return digits;
        }else{
            digits[i]=0;
        }
    }
    int *result=(int*)malloc(sizeof(int)*(digitsSize+1));
    result[0]=1;
    for(i=1;i<=digitsSize;i++){
        result[i]=digits[i-1];
    }
    *returnSize=digitsSize+1;
    return result;
}
int main(){
    
    return 0;
}