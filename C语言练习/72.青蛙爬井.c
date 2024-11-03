#include<stdio.h>
int main()
{
	int high,up,down,x=0,day=1;
	scanf("%d %d %d",&high,&up,&down);
	while(x<=high)
	{
		x+=up;
		if(x>=high) break;
		x-=down;
		day++;
	}
	printf("%d",day);
	
	return 0;
}
