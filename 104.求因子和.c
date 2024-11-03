#include<stdio.h>
int FacSum(int n)
{
	int i,y=0;
	for(i=1;i<n;i++)
	{
		if(n%i==0) y=y+i;
	}
	return y;
}
int main()
{
	int n;
	scanf("%d",&n);
	printf("%d",FacSum(n));
	
	return 0;
}
