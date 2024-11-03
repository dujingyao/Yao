#include<stdio.h>
int main()
{
	int n,x,sum=0,i,j=1;
	scanf("%d",&n);
	for(i=0;n>0;i++){
		x=n%10;
		if(x>=4) x--;
		n=n/10;
		sum=sum+x*j;
		j=j*9;
	}
	printf("%d",sum);
	
	return 0;
}
