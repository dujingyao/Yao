#include<stdio.h>
int main()
{
	char ch;
	scanf("%c",&ch);
	if(ch>='a'&&ch<='z'){
	ch-=32;
	printf("%c",ch);
	}else printf("%c",ch);
	
	return 0;
}
