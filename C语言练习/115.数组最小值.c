#include<stdio.h>
int main()
{
	int n,i;
	scanf("%d",&n);
	int a[n];
	for(i=0;i<n;i++)
	{
		scanf("%d",&a[i]);
	}
	int j,c=a[0],h=0;
	for(j=1;j<n;j++)
	{
		if(c>a[j]) c=a[j],h=j; 
	}
	printf("%d %d",c,h);
	
	return 0;
}
