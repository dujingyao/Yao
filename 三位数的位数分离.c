#include<stdio.h>
int main()
{int n;
	scanf("%d",&n);
	int a=n/100,b=n%100/10,c=n%10;
	printf("%d %d %d",c,b,a);
	
	
	return 0;
}
