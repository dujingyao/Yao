#include<stdio.h>
#include<string.h>
int f(int m);
int main(){
	int a[200]={0};
	int n,k,sum;
	scanf("%d",&n);
	for(int i=0;i<10000;i++){
		for(int j=0;j<10000;j++){
			k=i+j;
			sum=f(i)+f(j)+f(k)+4;
			if(sum<=24) a[sum]++;
		}
	}
	for(int i=0;i<25;i++){
		printf("%d ",a[i]);
	}
	return 0;
}
int f(int m){
	int b[10]={6,2,5,5,4,5,6,3,7,6};
	int sum=0,x,flag=0;
	while(1){
		if(flag==0&&m==0){
			sum=6;
			break;
		}
		flag=1;
		x=m%10;
		if(m<=0) break;
		sum=sum+b[x];
		m/=10;
	}
	return sum;
}