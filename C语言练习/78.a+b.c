#include<stdio.h>
int main()
{
	int n;
	int i,A,B,C;
	scanf("%d",&n);
	for(i=1;i<=n;i++)
	{
		scanf("%d %d",&A,&B);
		C=A+B;
		printf("%d\n",C);
	}
	
	return 0;
}
