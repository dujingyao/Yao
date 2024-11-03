#include<stdio.h>
int main()
{
	char ch;
	int i;
	for(i=0;;i++)
	{
		scanf("%c",&ch);
		if(ch=='\n') break;
	}printf("%d",i);
	
	return 0;
}
