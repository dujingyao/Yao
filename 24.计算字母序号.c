#include<stdio.h>
int main()
{
	char a,b;
	scanf("%c",&a);
	if(a>=97&&a<=122) b=a-96;
	if(a>=65&&a<=90) b=a-64;
	printf("%d\n",b);
	
	return 0;
}
