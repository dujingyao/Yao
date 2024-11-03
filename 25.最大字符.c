#include<stdio.h>
int main()
{
	char a,b,c;
	scanf("%c %c %c",&a,&b,&c);
	if(a>b&&a>c)
		printf("%c",a);
	if(b>a&&b>c)
		printf("%c",b);
	if(c>b&&c>a)
		printf("%c",c);
	
	return 0;
}
