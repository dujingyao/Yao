#include<stdio.h>
int main()
{
	int heads,feet;
	scanf("%d %d",&heads,&feet);
	int x=(4*heads-feet)/2,y=(feet-2*heads)/2;
	printf("%d %d",x,y);
	
	return 0;
}
