#include<stdio.h>
#include<stdlib.h>
#include<string.h>
void func(int a[],int n,int & size);
int main(){
    int a[5000],n,size=0;
    scanf("%d",&n);
    for (int i=0;i<5000;++i) {
        a[i]=0;
    }
    func(a,n,size);
    for(int i=size-1;i>=0;i--){
        printf("%d",a[i]);
    }
    return 0;
}
void func(int a[],int n,int & size){
    a[0]=1;
    int temp;
    int flag=0;
    size=1;
    for(int x=2;x<=n;x++){
        flag=0;
        for(int i=0;i<size;i++){
            temp=a[i]*x+flag;
            flag=temp/10;
            a[i]=temp%10;
        }
        while(flag){
            a[size++]=flag%10;
            flag/=10;
        }
    }
}
