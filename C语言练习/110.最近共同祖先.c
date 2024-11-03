#include<stdio.h>
int common(int x,int y)
{
	if(x==y) return x;
	while(1)
	{
		if(x>y) x=x/2;
		if(y>x) y=y/2;
		if(x==y) return x;
	}
}
int main()
{
	int x,y;
	scanf("%d %d",&x,&y);
	printf("%d",common(x,y));
	
	return 0;
}
