#include<stdio.h>
int facsum(int n)
{
	int i,j=0;
	for(i=1;i<n;i++)
	{
		if(n%i==0) j=j+i;
	}
	return j;
}
int main()
{
	int x,y,find=0,t;
	scanf("%d %d",&x,&y);
	for(int i=x;i<=y;i++)
	{
		t=facsum(i);
		if(facsum(t)==i&&i<t)
		{
			printf("%d %d\n",i,t);
			find++;
		}
	}
	if(find==0)
		printf("No answer");
	
	return 0;
}
