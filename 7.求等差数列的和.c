#include<stdio.h>
int main()
{
	int a,b,c;
	scanf("%d %d %d",&a,&b,&c);
	int n=(b-a)/c+1;
	int s=(a+b)*n/2;
	printf("%d",s);
	
	
	return 0;
}
