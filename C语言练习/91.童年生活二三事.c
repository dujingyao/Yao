#include<stdio.h>
int main()
{
	int a[41];
	int i,n;
	a[1]=1;a[2]=2;
	for(i=3;i<=40;i++)
	{
		a[i]=a[i-2]+a[i-1];
	}
	while((scanf("%d",&n)==1)&&(n!=0))
		printf("%d\n",a[n]);
		
	return 0;
}
