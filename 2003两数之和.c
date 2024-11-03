#include<stdio.h>
#include<stdlib.h>
int* twoSum(int* nums, int numsSize, int target, int* returnSize) {
    int i,j;
    int* q = (int*)malloc(2 * sizeof(int));
    for(i=0;i<numsSize;i++){
        for(j=i+1;j<numsSize;j++){
            if(nums[i]+nums[j]==target){
                q[0]=i;
                q[1]=j;
                *returnSize=2;
                return q;     
            }
        }
    }
    return NULL;
}
int main()
{
    int nums[]={3,2,4};
    int returnSize;
    int *q=twoSum(nums,3,6,&returnSize);
    if(q!=NULL){
        for(int i=0;i<2;i++){
            printf("%d ",q[i]);
        }
        free(q);
    }else{
        printf("未找到相应的值\n");
    }
    return 0;
}